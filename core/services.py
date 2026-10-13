from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from zoneinfo import ZoneInfo
from django.db import transaction
from django.utils import timezone

from core.exceptions import (
    IdempotencyConflictError,
    InsufficientStockError,
    InvalidTransitionError,
    NotFoundError,
    OrderLockedError,
    ValidationError,
)
from core.models import (
    Dealer,
    IdempotencyRecord,
    Inventory,
    Order,
    OrderDailySequence,
    OrderItem,
    OrderStatus,
    Product,
)

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
GST_RATE = Decimal("0.18")
CENTS = Decimal("0.01")


def hash_request_payload(data: dict) -> str:
    """Computes a deterministic SHA-256 hash of a JSON payload."""
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def generate_order_number() -> str:
    """
    Atomically generates a unique order number formatted as ORD-YYYYMMDD-NNNN
    in Asia/Kolkata timezone with a daily sequence starting at 0001.
    Uses select_for_update() on OrderDailySequence to guarantee uniqueness under concurrency.
    """
    now = timezone.now().astimezone(KOLKATA_TZ)
    date_str = now.strftime("%Y%m%d")

    seq_obj, _ = OrderDailySequence.objects.select_for_update().get_or_create(
        date_str=date_str,
        defaults={"last_seq": 0},
    )
    seq_obj.last_seq += 1
    seq_obj.save(update_fields=["last_seq"])

    return f"ORD-{date_str}-{seq_obj.last_seq:04d}"


def validate_and_compute_order_lines(items_data: list) -> tuple[list[dict], Decimal, Decimal, Decimal]:
    """
    Validates items input and computes exact line totals and order totals using current catalogue prices.
    Uses ROUND_HALF_UP for 18% GST rounding per line to 2 decimal places.
    """
    if not isinstance(items_data, list) or len(items_data) == 0:
        raise ValidationError(
            message="Order must contain at least one item.",
            details=[{"field": "items", "error": "Must not be empty"}],
        )

    seen_skus = set()
    sku_list = []
    qty_map = {}

    for idx, item in enumerate(items_data):
        if not isinstance(item, dict):
            raise ValidationError(
                message=f"Item at index {idx} must be an object.",
                details=[{"field": f"items[{idx}]", "error": "Invalid item format"}],
            )

        sku = item.get("sku")
        if not sku or not isinstance(sku, str):
            raise ValidationError(
                message=f"Item at index {idx} must have a valid 'sku'.",
                details=[{"field": f"items[{idx}].sku", "error": "SKU is required"}],
            )
        sku = sku.strip()

        if sku in seen_skus:
            raise ValidationError(
                message=f"Duplicate SKU '{sku}' in order. The same SKU must not appear twice.",
                details=[{"field": f"items[{idx}].sku", "error": f"Duplicate SKU '{sku}'"}],
            )
        seen_skus.add(sku)
        sku_list.append(sku)

        quantity = item.get("quantity")
        if quantity is None or not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
            raise ValidationError(
                message=f"Quantity for SKU '{sku}' must be an integer >= 1.",
                details=[{"field": f"items[{idx}].quantity", "error": "Must be integer >= 1"}],
            )
        qty_map[sku] = quantity

    # Query all products in one go
    products = {p.sku: p for p in Product.objects.filter(sku__in=sku_list)}
    missing_skus = [s for s in sku_list if s not in products]
    if missing_skus:
        raise ValidationError(
            message=f"Unknown SKU(s): {', '.join(missing_skus)}.",
            details=[{"field": "sku", "error": f"Unknown SKU: {s}"} for s in missing_skus],
        )

    lines = []
    subtotal = Decimal("0.00")
    tax_total = Decimal("0.00")

    for sku in sku_list:
        product = products[sku]
        qty = qty_map[sku]
        unit_price = Decimal(str(product.price)).quantize(CENTS, rounding=ROUND_HALF_UP)
        line_subtotal = (unit_price * qty).quantize(CENTS, rounding=ROUND_HALF_UP)
        line_tax = (line_subtotal * GST_RATE).quantize(CENTS, rounding=ROUND_HALF_UP)
        line_total = (line_subtotal + line_tax).quantize(CENTS, rounding=ROUND_HALF_UP)

        subtotal += line_subtotal
        tax_total += line_tax

        lines.append(
            {
                "product": product,
                "sku": product.sku,
                "name": product.name,
                "quantity": qty,
                "unit_price": unit_price,
                "line_subtotal": line_subtotal,
                "line_tax": line_tax,
                "line_total": line_total,
            }
        )

    grand_total = subtotal + tax_total
    return lines, subtotal, tax_total, grand_total


def create_draft_order(
    dealer_id: int,
    items_data: list,
    idempotency_key: str | None = None,
    raw_payload: dict | None = None,
) -> tuple[Order, bool]:
    """
    Creates a new DRAFT order with full validation and current catalogue prices.
    Handles Idempotency-Key if provided:
    - Same key + same payload -> returns existing order (created=False).
    - Same key + different payload -> raises IdempotencyConflictError (409).
    """
    if dealer_id is None:
        raise ValidationError(
            message="dealer_id is required.",
            details=[{"field": "dealer_id", "error": "dealer_id is required"}],
        )

    try:
        dealer = Dealer.objects.get(id=dealer_id)
    except Dealer.DoesNotExist:
        raise ValidationError(
            message=f"Dealer with id {dealer_id} does not exist.",
            details=[{"field": "dealer_id", "error": f"Unknown dealer_id {dealer_id}"}],
        )

    req_hash = hash_request_payload(raw_payload) if (idempotency_key and raw_payload) else ""

    with transaction.atomic():
        if idempotency_key:
            # Check if idempotency record exists with lock
            rec = (
                IdempotencyRecord.objects.select_for_update()
                .filter(key=idempotency_key)
                .first()
            )
            if rec:
                if rec.request_hash == req_hash:
                    return rec.order, False
                raise IdempotencyConflictError(
                    message="Idempotency key reused with a different request body.",
                    details=[{"idempotency_key": idempotency_key}],
                )

        lines, subtotal, tax_total, grand_total = validate_and_compute_order_lines(items_data)
        order_number = generate_order_number()

        order = Order.objects.create(
            order_number=order_number,
            dealer=dealer,
            status=OrderStatus.DRAFT,
            subtotal=subtotal,
            tax_total=tax_total,
            grand_total=grand_total,
        )

        order_items = [
            OrderItem(
                order=order,
                product=l["product"],
                sku=l["sku"],
                name=l["name"],
                quantity=l["quantity"],
                unit_price=l["unit_price"],
                line_subtotal=l["line_subtotal"],
                line_tax=l["line_tax"],
                line_total=l["line_total"],
            )
            for l in lines
        ]
        OrderItem.objects.bulk_create(order_items)

        if idempotency_key:
            IdempotencyRecord.objects.create(
                key=idempotency_key,
                request_hash=req_hash,
                order=order,
            )

        return order, True


def update_draft_order(order_number: str, items_data: list) -> Order:
    """
    Replaces all items in a DRAFT order.
    Only allowed while order is DRAFT, otherwise raises OrderLockedError (409).
    """
    with transaction.atomic():
        try:
            order = Order.objects.select_for_update().get(order_number=order_number)
        except Order.DoesNotExist:
            raise NotFoundError(message=f"Order '{order_number}' not found.")

        if order.status != OrderStatus.DRAFT:
            raise OrderLockedError(
                message=f"Order '{order_number}' is not in DRAFT status and cannot be edited.",
                details=[{"order_number": order_number, "status": order.status}],
            )

        lines, subtotal, tax_total, grand_total = validate_and_compute_order_lines(items_data)

        # Remove old items
        order.items.all().delete()

        # Create new items
        order_items = [
            OrderItem(
                order=order,
                product=l["product"],
                sku=l["sku"],
                name=l["name"],
                quantity=l["quantity"],
                unit_price=l["unit_price"],
                line_subtotal=l["line_subtotal"],
                line_tax=l["line_tax"],
                line_total=l["line_total"],
            )
            for l in lines
        ]
        OrderItem.objects.bulk_create(order_items)

        order.subtotal = subtotal
        order.tax_total = tax_total
        order.grand_total = grand_total
        order.save(update_fields=["subtotal", "tax_total", "grand_total", "updated_at"])

        return order


def confirm_order(order_number: str) -> Order:
    """
    Transitions DRAFT -> CONFIRMED.
    Validates and deducts stock atomically with row locks (select_for_update).
    All-or-nothing: if any line exceeds stock, returns 409 INSUFFICIENT_STOCK with details and rolls back.
    Locks prices permanently.
    """
    with transaction.atomic():
        try:
            order = Order.objects.select_for_update().get(order_number=order_number)
        except Order.DoesNotExist:
            raise NotFoundError(message=f"Order '{order_number}' not found.")

        if order.status != OrderStatus.DRAFT:
            raise InvalidTransitionError(
                message=f"Cannot transition order '{order_number}' from {order.status} to CONFIRMED.",
                details=[{"order_number": order_number, "current_status": order.status, "target_status": "CONFIRMED"}],
            )

        items = list(order.items.all())
        if not items:
            raise ValidationError(message="Cannot confirm an empty order.")

        # Sort SKUs to prevent deadlocks across concurrent confirm transactions
        sorted_skus = sorted([item.sku for item in items])
        inventories = {
            inv.product_id: inv
            for inv in Inventory.objects.select_for_update().filter(product_id__in=sorted_skus)
        }

        short_lines = []
        message_parts = []

        for item in items:
            inv = inventories.get(item.sku)
            available = inv.quantity if inv else 0
            if item.quantity > available:
                short_lines.append(
                    {
                        "sku": item.sku,
                        "requested": item.quantity,
                        "available": available,
                    }
                )
                message_parts.append(
                    f"Insufficient stock for {item.name}. Available: {available}, Requested: {item.quantity}"
                )

        if short_lines:
            raise InsufficientStockError(
                message=". ".join(message_parts),
                details=short_lines,
            )

        # Deduct stock
        for item in items:
            inv = inventories[item.sku]
            inv.quantity -= item.quantity
            inv.save(update_fields=["quantity", "updated_at"])

        order.status = OrderStatus.CONFIRMED
        order.stock_deducted = True
        order.save(update_fields=["status", "stock_deducted", "updated_at"])

        return order


def dispatch_order(order_number: str) -> Order:
    """Transitions CONFIRMED -> DISPATCHED."""
    with transaction.atomic():
        try:
            order = Order.objects.select_for_update().get(order_number=order_number)
        except Order.DoesNotExist:
            raise NotFoundError(message=f"Order '{order_number}' not found.")

        if order.status != OrderStatus.CONFIRMED:
            raise InvalidTransitionError(
                message=f"Cannot transition order '{order_number}' from {order.status} to DISPATCHED.",
                details=[{"order_number": order_number, "current_status": order.status, "target_status": "DISPATCHED"}],
            )

        order.status = OrderStatus.DISPATCHED
        order.save(update_fields=["status", "updated_at"])
        return order


def deliver_order(order_number: str) -> Order:
    """Transitions DISPATCHED -> DELIVERED."""
    with transaction.atomic():
        try:
            order = Order.objects.select_for_update().get(order_number=order_number)
        except Order.DoesNotExist:
            raise NotFoundError(message=f"Order '{order_number}' not found.")

        if order.status != OrderStatus.DISPATCHED:
            raise InvalidTransitionError(
                message=f"Cannot transition order '{order_number}' from {order.status} to DELIVERED.",
                details=[{"order_number": order_number, "current_status": order.status, "target_status": "DELIVERED"}],
            )

        order.status = OrderStatus.DELIVERED
        order.save(update_fields=["status", "updated_at"])
        return order


def cancel_order(order_number: str) -> Order:
    """
    Transitions DRAFT -> CANCELLED or CONFIRMED -> CANCELLED.
    If CONFIRMED, restores stock exactly once.
    Rejects any transition from DISPATCHED, DELIVERED, or CANCELLED.
    """
    with transaction.atomic():
        try:
            order = Order.objects.select_for_update().get(order_number=order_number)
        except Order.DoesNotExist:
            raise NotFoundError(message=f"Order '{order_number}' not found.")

        if order.status not in [OrderStatus.DRAFT, OrderStatus.CONFIRMED]:
            raise InvalidTransitionError(
                message=f"Cannot cancel order '{order_number}' in {order.status} status.",
                details=[{"order_number": order_number, "current_status": order.status, "target_status": "CANCELLED"}],
            )

        if order.status == OrderStatus.CONFIRMED and order.stock_deducted:
            items = list(order.items.all())
            sorted_skus = sorted([item.sku for item in items])
            inventories = {
                inv.product_id: inv
                for inv in Inventory.objects.select_for_update().filter(product_id__in=sorted_skus)
            }
            for item in items:
                inv = inventories.get(item.sku)
                if inv:
                    inv.quantity += item.quantity
                    inv.save(update_fields=["quantity", "updated_at"])
            order.stock_deducted = False

        order.status = OrderStatus.CANCELLED
        order.save(update_fields=["status", "stock_deducted", "updated_at"])
        return order
