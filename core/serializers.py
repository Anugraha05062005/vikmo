from decimal import Decimal
import re
from rest_framework import serializers

from core.models import Dealer, Order, OrderItem, Product

MONEY_REGEX = re.compile(r"^\d+(\.\d{1,2})?$")


def format_money(value: Decimal | None) -> str:
    if value is None:
        return "0.00"
    return f"{Decimal(value):.2f}"


class ProductSerializer(serializers.ModelSerializer):
    price = serializers.CharField()
    stock = serializers.SerializerMethodField()

    class Meta:
        model = Product
        fields = [
            "sku",
            "name",
            "category",
            "brand",
            "vehicle_fitment",
            "price",
            "stock",
            "description",
        ]

    def get_stock(self, obj) -> int:
        return obj.stock

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret["price"] = format_money(Decimal(instance.price))
        return ret

    def validate_price(self, value):
        str_val = str(value).strip()
        if not MONEY_REGEX.match(str_val):
            raise serializers.ValidationError("Price must be a positive number with at most 2 decimal places.")
        try:
            dec_val = Decimal(str_val)
        except Exception:
            raise serializers.ValidationError("Invalid price format.")
        if dec_val <= Decimal("0.00"):
            raise serializers.ValidationError("Price must be greater than 0.")
        return dec_val


class ProductUpdateSerializer(serializers.ModelSerializer):
    price = serializers.CharField(required=False)

    class Meta:
        model = Product
        fields = [
            "name",
            "category",
            "brand",
            "vehicle_fitment",
            "price",
            "description",
        ]

    def validate_price(self, value):
        str_val = str(value).strip()
        if not MONEY_REGEX.match(str_val):
            raise serializers.ValidationError("Price must be a positive number with at most 2 decimal places.")
        try:
            dec_val = Decimal(str_val)
        except Exception:
            raise serializers.ValidationError("Invalid price format.")
        if dec_val <= Decimal("0.00"):
            raise serializers.ValidationError("Price must be greater than 0.")
        return dec_val


class DealerSerializer(serializers.ModelSerializer):
    credit_limit = serializers.CharField(required=False)

    class Meta:
        model = Dealer
        fields = [
            "id",
            "dealer_code",
            "name",
            "city",
            "phone",
            "email",
            "gstin",
            "credit_limit",
        ]
        read_only_fields = ["id"]

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret["credit_limit"] = format_money(instance.credit_limit)
        return ret

    def validate_credit_limit(self, value):
        if value is None or value == "":
            return Decimal("0.00")
        str_val = str(value).strip()
        if not MONEY_REGEX.match(str_val):
            raise serializers.ValidationError("Credit limit must be a positive amount with at most 2 decimal places.")
        return Decimal(str_val)


class OrderItemSerializer(serializers.ModelSerializer):
    unit_price = serializers.SerializerMethodField()
    line_subtotal = serializers.SerializerMethodField()
    line_tax = serializers.SerializerMethodField()
    line_total = serializers.SerializerMethodField()

    class Meta:
        model = OrderItem
        fields = [
            "sku",
            "name",
            "quantity",
            "unit_price",
            "line_subtotal",
            "line_tax",
            "line_total",
        ]

    def get_unit_price(self, obj) -> str:
        return format_money(obj.unit_price)

    def get_line_subtotal(self, obj) -> str:
        return format_money(obj.line_subtotal)

    def get_line_tax(self, obj) -> str:
        return format_money(obj.line_tax)

    def get_line_total(self, obj) -> str:
        return format_money(obj.line_total)


class OrderSerializer(serializers.ModelSerializer):
    dealer_id = serializers.IntegerField(source="dealer.id")
    items = OrderItemSerializer(many=True, read_only=True)
    subtotal = serializers.SerializerMethodField()
    tax_total = serializers.SerializerMethodField()
    grand_total = serializers.SerializerMethodField()
    created_at = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%S%z")
    updated_at = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%S%z")

    class Meta:
        model = Order
        fields = [
            "order_number",
            "dealer_id",
            "status",
            "items",
            "subtotal",
            "tax_total",
            "grand_total",
            "created_at",
            "updated_at",
        ]

    def get_subtotal(self, obj) -> str:
        return format_money(obj.subtotal)

    def get_tax_total(self, obj) -> str:
        return format_money(obj.tax_total)

    def get_grand_total(self, obj) -> str:
        return format_money(obj.grand_total)
