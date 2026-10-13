from decimal import Decimal
from django.db import models
from django.utils import timezone


class Product(models.Model):
    sku = models.CharField(max_length=64, primary_key=True)
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=100, db_index=True)
    brand = models.CharField(max_length=100, db_index=True)
    vehicle_fitment = models.CharField(max_length=255, db_index=True)
    price = models.DecimalField(max_digits=12, decimal_places=2)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["sku"]
        indexes = [
            models.Index(fields=["sku"]),
            models.Index(fields=["name"]),
            models.Index(fields=["brand"]),
            models.Index(fields=["category"]),
            models.Index(fields=["vehicle_fitment"]),
        ]

    def __str__(self):
        return f"{self.sku} - {self.name}"

    @property
    def stock(self) -> int:
        if hasattr(self, "inventory"):
            return self.inventory.quantity
        return 0


class Inventory(models.Model):
    product = models.OneToOneField(
        Product,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="inventory",
    )
    quantity = models.IntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Inventories"
        constraints = [
            models.CheckConstraint(
                check=models.Q(quantity__gte=0),
                name="inventory_quantity_non_negative",
            )
        ]

    def __str__(self):
        return f"{self.product_id}: {self.quantity}"


class Dealer(models.Model):
    dealer_code = models.CharField(max_length=64, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    city = models.CharField(max_length=100, blank=True, default="")
    phone = models.CharField(max_length=50, blank=True, default="")
    email = models.CharField(max_length=255, blank=True, default="")
    gstin = models.CharField(max_length=50, blank=True, default="")
    credit_limit = models.DecimalField(
        max_digits=14, decimal_places=2, default=Decimal("0.00")
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["id"]
        indexes = [
            models.Index(fields=["dealer_code"]),
        ]

    def __str__(self):
        return f"{self.dealer_code} - {self.name}"


class OrderStatus(models.TextChoices):
    DRAFT = "DRAFT", "Draft"
    CONFIRMED = "CONFIRMED", "Confirmed"
    DISPATCHED = "DISPATCHED", "Dispatched"
    DELIVERED = "DELIVERED", "Delivered"
    CANCELLED = "CANCELLED", "Cancelled"


class Order(models.Model):
    order_number = models.CharField(max_length=64, unique=True, db_index=True)
    dealer = models.ForeignKey(
        Dealer,
        on_delete=models.PROTECT,
        related_name="orders",
    )
    status = models.CharField(
        max_length=30,
        choices=OrderStatus.choices,
        default=OrderStatus.DRAFT,
        db_index=True,
    )
    subtotal = models.DecimalField(
        max_digits=14, decimal_places=2, default=Decimal("0.00")
    )
    tax_total = models.DecimalField(
        max_digits=14, decimal_places=2, default=Decimal("0.00")
    )
    grand_total = models.DecimalField(
        max_digits=14, decimal_places=2, default=Decimal("0.00")
    )
    stock_deducted = models.BooleanField(
        default=False,
        help_text="Tracks whether inventory has been decremented for this order",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["order_number"]),
            models.Index(fields=["dealer", "status"]),
        ]

    def __str__(self):
        return f"{self.order_number} ({self.status})"


class OrderItem(models.Model):
    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name="items",
    )
    product = models.ForeignKey(
        Product,
        on_delete=models.PROTECT,
        related_name="order_items",
    )
    sku = models.CharField(max_length=64)
    name = models.CharField(max_length=255)
    quantity = models.PositiveIntegerField()
    unit_price = models.DecimalField(max_digits=12, decimal_places=2)
    line_subtotal = models.DecimalField(max_digits=14, decimal_places=2)
    line_tax = models.DecimalField(max_digits=14, decimal_places=2)
    line_total = models.DecimalField(max_digits=14, decimal_places=2)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"{self.sku} x {self.quantity} on {self.order.order_number}"


class OrderDailySequence(models.Model):
    date_str = models.CharField(max_length=8, primary_key=True)  # YYYYMMDD in Asia/Kolkata
    last_seq = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"{self.date_str}: {self.last_seq}"


class IdempotencyRecord(models.Model):
    key = models.CharField(max_length=255, primary_key=True)
    request_hash = models.CharField(max_length=64)
    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name="idempotency_records",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.key} -> {self.order.order_number}"


class AssistantSession(models.Model):
    session_id = models.CharField(max_length=128, primary_key=True)
    dealer = models.ForeignKey(
        Dealer,
        on_delete=models.CASCADE,
        related_name="assistant_sessions",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Session {self.session_id} (Dealer {self.dealer_id})"


class AssistantMessage(models.Model):
    session = models.ForeignKey(
        AssistantSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=20)  # user, assistant, system, tool
    content = models.TextField(blank=True, default="")
    tool_calls = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role} in {self.session_id}"
