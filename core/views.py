from decimal import Decimal
from django.db import IntegrityError, models, transaction
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from core.exceptions import (
    DuplicateError,
    NotFoundError,
    ValidationError,
)
from core.models import Dealer, Inventory, Order, Product
from core.pagination import StandardPagination
from core.serializers import (
    DealerSerializer,
    OrderSerializer,
    ProductSerializer,
    ProductUpdateSerializer,
)
from core import assistant, reorder_service, services, supplier_sync


def health(request):
    """GET /api/health/ — the grader polls this until your app is ready."""
    return JsonResponse({"status": "ok"})


# ==============================================================================
# Products Endpoints
# ==============================================================================


@api_view(["GET", "POST"])
def product_list(request):
    if request.method == "GET":
        qs = Product.objects.select_related("inventory").all()

        search = request.query_params.get("search")
        if search:
            qs = qs.filter(
                models.Q(sku__icontains=search)
                | models.Q(name__icontains=search)
                | models.Q(brand__icontains=search)
            )

        category = request.query_params.get("category")
        if category:
            qs = qs.filter(category=category)

        vehicle = request.query_params.get("vehicle")
        if vehicle:
            qs = qs.filter(vehicle_fitment__icontains=vehicle)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        serializer = ProductSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == "POST":
        serializer = ProductSerializer(data=request.data)
        if not serializer.is_valid():
            errors = serializer.errors
            if "sku" in errors and any("exists" in str(e).lower() for e in errors["sku"]):
                raise DuplicateError(
                    message=f"Product with SKU '{request.data.get('sku')}' already exists.",
                    details=[{"field": "sku", "error": "Already exists"}],
                )
            raise ValidationError(
                message="Invalid product input.",
                details=[{"field": f, "error": str(e)} for f, e in errors.items()],
            )

        sku = serializer.validated_data["sku"]
        if Product.objects.filter(sku=sku).exists():
            raise DuplicateError(
                message=f"Product with SKU '{sku}' already exists.",
                details=[{"field": "sku", "error": "Already exists"}],
            )

        with transaction.atomic():
            product = serializer.save()
            Inventory.objects.create(product=product, quantity=0)

        return Response(ProductSerializer(product).data, status=status.HTTP_201_CREATED)


@api_view(["GET", "PATCH"])
def product_detail(request, sku: str):
    try:
        product = Product.objects.select_related("inventory").get(sku=sku)
    except Product.DoesNotExist:
        raise NotFoundError(message=f"Product with SKU '{sku}' not found.")

    if request.method == "GET":
        return Response(ProductSerializer(product).data)

    elif request.method == "PATCH":
        # Disallow updating SKU
        patch_data = request.data.copy()
        if "sku" in patch_data and patch_data["sku"] != sku:
            raise ValidationError(
                message="SKU cannot be changed.",
                details=[{"field": "sku", "error": "Immutable field"}],
            )

        serializer = ProductUpdateSerializer(product, data=patch_data, partial=True)
        if not serializer.is_valid():
            raise ValidationError(
                message="Invalid product update.",
                details=[{"field": f, "error": str(e)} for f, e in serializer.errors.items()],
            )

        product = serializer.save()
        return Response(ProductSerializer(product).data)


# ==============================================================================
# Inventory Endpoints
# ==============================================================================


@api_view(["PUT"])
def inventory_detail(request, sku: str):
    try:
        product = Product.objects.get(sku=sku)
    except Product.DoesNotExist:
        raise NotFoundError(message=f"Product with SKU '{sku}' not found.")

    qty = request.data.get("quantity")
    if qty is None or not isinstance(qty, int) or isinstance(qty, bool) or qty < 0:
        raise ValidationError(
            message="Quantity must be an integer >= 0.",
            details=[{"field": "quantity", "error": "Must be integer >= 0"}],
        )

    with transaction.atomic():
        inventory, _ = Inventory.objects.select_for_update().get_or_create(
            product=product,
            defaults={"quantity": qty},
        )
        inventory.quantity = qty
        inventory.save(update_fields=["quantity", "updated_at"])

    return Response({"sku": sku, "quantity": inventory.quantity}, status=status.HTTP_200_OK)


# ==============================================================================
# Dealers Endpoints
# ==============================================================================


@api_view(["GET", "POST"])
def dealer_list(request):
    if request.method == "GET":
        qs = Dealer.objects.all().order_by("id")
        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        serializer = DealerSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == "POST":
        serializer = DealerSerializer(data=request.data)
        if not serializer.is_valid():
            errors = serializer.errors
            if "dealer_code" in errors and any("exists" in str(e).lower() for e in errors["dealer_code"]):
                raise DuplicateError(
                    message=f"Dealer with code '{request.data.get('dealer_code')}' already exists.",
                    details=[{"field": "dealer_code", "error": "Already exists"}],
                )
            raise ValidationError(
                message="Invalid dealer data.",
                details=[{"field": f, "error": str(e)} for f, e in errors.items()],
            )

        dealer_code = serializer.validated_data.get("dealer_code")
        if Dealer.objects.filter(dealer_code=dealer_code).exists():
            raise DuplicateError(
                message=f"Dealer with code '{dealer_code}' already exists.",
                details=[{"field": "dealer_code", "error": "Already exists"}],
            )

        try:
            dealer = serializer.save()
        except IntegrityError:
            raise DuplicateError(
                message=f"Dealer with code '{dealer_code}' already exists.",
                details=[{"field": "dealer_code", "error": "Already exists"}],
            )

        return Response(DealerSerializer(dealer).data, status=status.HTTP_201_CREATED)


@api_view(["GET"])
def dealer_detail(request, pk: int):
    try:
        dealer = Dealer.objects.get(id=pk)
    except Dealer.DoesNotExist:
        raise NotFoundError(message=f"Dealer with id {pk} not found.")

    return Response(DealerSerializer(dealer).data)


# ==============================================================================
# Orders Endpoints
# ==============================================================================


@api_view(["GET", "POST"])
def order_list(request):
    if request.method == "GET":
        qs = Order.objects.prefetch_related("items").all()

        dealer_id = request.query_params.get("dealer_id")
        if dealer_id:
            qs = qs.filter(dealer_id=dealer_id)

        status_param = request.query_params.get("status")
        if status_param:
            qs = qs.filter(status=status_param)

        paginator = StandardPagination()
        page = paginator.paginate_queryset(qs, request)
        serializer = OrderSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    elif request.method == "POST":
        data = request.data
        if not isinstance(data, dict):
            raise ValidationError(message="Request body must be a JSON object.")

        dealer_id = data.get("dealer_id")
        items = data.get("items")
        idempotency_key = request.headers.get("Idempotency-Key")

        order, created = services.create_draft_order(
            dealer_id=dealer_id,
            items_data=items,
            idempotency_key=idempotency_key,
            raw_payload=data,
        )

        resp_status = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return Response(OrderSerializer(order).data, status=resp_status)


@api_view(["GET", "PUT"])
def order_detail(request, order_number: str):
    if request.method == "GET":
        try:
            order = Order.objects.prefetch_related("items").get(order_number=order_number)
        except Order.DoesNotExist:
            raise NotFoundError(message=f"Order '{order_number}' not found.")
        return Response(OrderSerializer(order).data)

    elif request.method == "PUT":
        items = request.data.get("items")
        order = services.update_draft_order(order_number=order_number, items_data=items)
        return Response(OrderSerializer(order).data)


@api_view(["POST"])
def order_confirm(request, order_number: str):
    order = services.confirm_order(order_number)
    return Response(OrderSerializer(order).data)


@api_view(["POST"])
def order_dispatch(request, order_number: str):
    order = services.dispatch_order(order_number)
    return Response(OrderSerializer(order).data)


@api_view(["POST"])
def order_deliver(request, order_number: str):
    order = services.deliver_order(order_number)
    return Response(OrderSerializer(order).data)


@api_view(["POST"])
def order_cancel(request, order_number: str):
    order = services.cancel_order(order_number)
    return Response(OrderSerializer(order).data)


# ==============================================================================
# Forecasting & Reorder Suggestions Endpoints
# ==============================================================================


@api_view(["GET"])
def forecast_detail(request, sku: str):
    data = reorder_service.get_forecast_for_sku(sku)
    return Response(data)


@api_view(["GET"])
def reorder_suggestions(request):
    data = reorder_service.get_reorder_suggestions()
    return Response(data)


# ==============================================================================
# Assistant Endpoints
# ==============================================================================


@api_view(["POST"])
def assistant_chat(request):
    data = request.data
    if not isinstance(data, dict):
        raise ValidationError(message="Request body must be a JSON object.")

    session_id = data.get("session_id")
    dealer_id = data.get("dealer_id")
    message = data.get("message")

    if not message or not isinstance(message, str):
        raise ValidationError(
            message="message is required.",
            details=[{"field": "message", "error": "Must not be empty"}],
        )

    res = assistant.chat(session_id=session_id, dealer_id=dealer_id, message=message)
    return Response(res, status=status.HTTP_200_OK)


# ==============================================================================
# Supplier Sync Endpoints
# ==============================================================================


@api_view(["POST"])
def sync_supplier(request):
    data = supplier_sync.run_supplier_sync()
    return Response(data, status=status.HTTP_200_OK)
