from decimal import Decimal
import json
from pathlib import Path
from django.conf import settings
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from core.models import Dealer, Inventory, Order, OrderItem, OrderStatus, Product
from core import services


class DealerDeskAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        # Create test products
        self.p1 = Product.objects.create(
            sku="TST-1001",
            name="Front Brake Pads - Pulsar 150",
            category="Brakes",
            brand="Endurance",
            vehicle_fitment="Bajaj Pulsar 150",
            price=Decimal("100.00"),
            description="High quality test brake pads",
        )
        self.inv1 = Inventory.objects.create(product=self.p1, quantity=10)

        self.p2 = Product.objects.create(
            sku="TST-1002",
            name="Air Filter - Pulsar 150",
            category="Filters",
            brand="Bosch",
            vehicle_fitment="Bajaj Pulsar 150",
            price=Decimal("50.00"),
            description="Test air filter",
        )
        self.inv2 = Inventory.objects.create(product=self.p2, quantity=5)

        # Create test dealer
        self.dealer = Dealer.objects.create(
            dealer_code="DLR-TEST",
            name="Test Dealer Ltd",
            city="Bengaluru",
            credit_limit=Decimal("50000.00"),
        )

    def test_health_endpoint(self):
        resp = self.client.get("/api/health/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_product_list_and_filters(self):
        resp = self.client.get("/api/products/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("count", resp.json())
        self.assertIn("results", resp.json())

        # Search filter
        resp_s = self.client.get("/api/products/?search=Brake")
        self.assertEqual(resp_s.status_code, status.HTTP_200_OK)
        self.assertTrue(any(p["sku"] == "TST-1001" for p in resp_s.json()["results"]))

        # Category filter
        resp_c = self.client.get("/api/products/?category=Brakes")
        self.assertEqual(resp_c.status_code, status.HTTP_200_OK)
        self.assertTrue(all(p["category"] == "Brakes" for p in resp_c.json()["results"]))

    def test_product_detail_and_404(self):
        resp = self.client.get("/api/products/TST-1001/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["sku"], "TST-1001")
        self.assertEqual(resp.json()["stock"], 10)

        resp_404 = self.client.get("/api/products/NOPE-9999/")
        self.assertEqual(resp_404.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(resp_404.json()["error"]["code"], "NOT_FOUND")

    def test_product_create_and_duplicate(self):
        new_prod = {
            "sku": "TST-9999",
            "name": "Spark Plug",
            "category": "Electrical",
            "brand": "NGK",
            "vehicle_fitment": "Universal",
            "price": "150.00",
            "description": "Premium spark plug",
        }
        resp = self.client.post("/api/products/", new_prod, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(resp.json()["stock"], 0)

        # Duplicate SKU -> 409 DUPLICATE
        resp_dup = self.client.post("/api/products/", new_prod, format="json")
        self.assertEqual(resp_dup.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(resp_dup.json()["error"]["code"], "DUPLICATE")

    def test_inventory_update(self):
        resp = self.client.put(
            "/api/inventory/TST-1001/",
            {"quantity": 42, "reason": "test restock"},
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json(), {"sku": "TST-1001", "quantity": 42})

        # Check reflected in product detail
        resp_p = self.client.get("/api/products/TST-1001/")
        self.assertEqual(resp_p.json()["stock"], 42)

        # Negative quantity -> 400
        resp_neg = self.client.put(
            "/api/inventory/TST-1001/",
            {"quantity": -5},
            format="json",
        )
        self.assertEqual(resp_neg.status_code, status.HTTP_400_BAD_REQUEST)

    def test_order_creation_and_gst_calculation(self):
        # 2 units of p1 (100.00) + 1 unit of p2 (50.00)
        # subtotal: 200 + 50 = 250.00
        # tax line 1: 200 * 0.18 = 36.00
        # tax line 2: 50 * 0.18 = 9.00
        # tax total = 45.00
        # grand total = 295.00
        order_data = {
            "dealer_id": self.dealer.id,
            "items": [
                {"sku": "TST-1001", "quantity": 2},
                {"sku": "TST-1002", "quantity": 1},
            ],
        }
        resp = self.client.post("/api/orders/", order_data, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        data = resp.json()
        self.assertEqual(data["status"], "DRAFT")
        self.assertEqual(data["subtotal"], "250.00")
        self.assertEqual(data["tax_total"], "45.00")
        self.assertEqual(data["grand_total"], "295.00")
        self.assertTrue(data["order_number"].startswith("ORD-"))

    def test_order_validation_errors(self):
        # Empty items
        r1 = self.client.post("/api/orders/", {"dealer_id": self.dealer.id, "items": []}, format="json")
        self.assertEqual(r1.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(r1.json()["error"]["code"], "VALIDATION_ERROR")

        # Duplicate SKU in order
        r2 = self.client.post(
            "/api/orders/",
            {
                "dealer_id": self.dealer.id,
                "items": [
                    {"sku": "TST-1001", "quantity": 1},
                    {"sku": "TST-1001", "quantity": 2},
                ],
            },
            format="json",
        )
        self.assertEqual(r2.status_code, status.HTTP_400_BAD_REQUEST)

        # Invalid quantity
        r3 = self.client.post(
            "/api/orders/",
            {
                "dealer_id": self.dealer.id,
                "items": [{"sku": "TST-1001", "quantity": 0}],
            },
            format="json",
        )
        self.assertEqual(r3.status_code, status.HTTP_400_BAD_REQUEST)

    def test_idempotency_key(self):
        order_payload = {
            "dealer_id": self.dealer.id,
            "items": [{"sku": "TST-1001", "quantity": 1}],
        }
        headers = {"HTTP_IDEMPOTENCY_KEY": "test-key-123"}
        r1 = self.client.post("/api/orders/", order_payload, format="json", **headers)
        self.assertEqual(r1.status_code, status.HTTP_201_CREATED)
        order_num = r1.json()["order_number"]

        # Duplicate with same payload -> returns original order
        r2 = self.client.post("/api/orders/", order_payload, format="json", **headers)
        self.assertIn(r2.status_code, [status.HTTP_200_OK, status.HTTP_201_CREATED])
        self.assertEqual(r2.json()["order_number"], order_num)

        # Duplicate with different payload -> 409 IDEMPOTENCY_CONFLICT
        diff_payload = {
            "dealer_id": self.dealer.id,
            "items": [{"sku": "TST-1001", "quantity": 5}],
        }
        r3 = self.client.post("/api/orders/", diff_payload, format="json", **headers)
        self.assertEqual(r3.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(r3.json()["error"]["code"], "IDEMPOTENCY_CONFLICT")

    def test_order_lifecycle_and_stock(self):
        # Initial stock: TST-1001 has 10
        order_payload = {
            "dealer_id": self.dealer.id,
            "items": [{"sku": "TST-1001", "quantity": 3}],
        }
        r = self.client.post("/api/orders/", order_payload, format="json")
        order_num = r.json()["order_number"]

        # Draft order has not touched stock
        self.assertEqual(Inventory.objects.get(product=self.p1).quantity, 10)

        # Confirm order
        r_conf = self.client.post(f"/api/orders/{order_num}/confirm/")
        self.assertEqual(r_conf.status_code, status.HTTP_200_OK)
        self.assertEqual(r_conf.json()["status"], "CONFIRMED")
        self.assertEqual(Inventory.objects.get(product=self.p1).quantity, 7)

        # Cancel confirmed order -> stock restored to 10
        r_cancel = self.client.post(f"/api/orders/{order_num}/cancel/")
        self.assertEqual(r_cancel.status_code, status.HTTP_200_OK)
        self.assertEqual(r_cancel.json()["status"], "CANCELLED")
        self.assertEqual(Inventory.objects.get(product=self.p1).quantity, 10)

        # Cancel again or confirm cancelled -> 409 INVALID_TRANSITION
        r_invalid = self.client.post(f"/api/orders/{order_num}/confirm/")
        self.assertEqual(r_invalid.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(r_invalid.json()["error"]["code"], "INVALID_TRANSITION")

    def test_insufficient_stock_rollback(self):
        # TST-1002 has 5 units available
        order_payload = {
            "dealer_id": self.dealer.id,
            "items": [{"sku": "TST-1002", "quantity": 10}],
        }
        r = self.client.post("/api/orders/", order_payload, format="json")
        order_num = r.json()["order_number"]

        # Attempt to confirm -> 409 INSUFFICIENT_STOCK
        r_conf = self.client.post(f"/api/orders/{order_num}/confirm/")
        self.assertEqual(r_conf.status_code, status.HTTP_409_CONFLICT)
        err = r_conf.json()["error"]
        self.assertEqual(err["code"], "INSUFFICIENT_STOCK")
        self.assertEqual(err["details"][0]["sku"], "TST-1002")
        self.assertEqual(err["details"][0]["available"], 5)
        self.assertEqual(err["details"][0]["requested"], 10)

        # Stock unchanged
        self.assertEqual(Inventory.objects.get(product=self.p2).quantity, 5)

    def test_assistant_chat(self):
        resp = self.client.post(
            "/api/assistant/chat/",
            {
                "session_id": "test-session-1",
                "dealer_id": self.dealer.id,
                "message": "Do you have brake pads for a Pulsar 150?",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertIn("reply", data)
        self.assertIn("cited_skus", data)
        self.assertIn("tool_calls", data)
        self.assertIn("actions", data)
        self.assertIsInstance(data["reply"], str)
        self.assertIsInstance(data["cited_skus"], list)

    def test_reorder_suggestions_endpoint(self):
        resp = self.client.get("/api/reorder-suggestions/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        results = resp.json().get("results")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 40)
        for k in ("sku", "current_stock", "lead_time_weeks", "lead_time_demand", "safety_stock", "reorder_point", "reorder_qty"):
            self.assertIn(k, results[0])

    def test_forecast_detail_endpoint(self):
        resp = self.client.get("/api/forecast/FIL-1016/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        data = resp.json()
        self.assertEqual(data["sku"], "FIL-1016")
        self.assertEqual(len(data["forecast"]), 4)

        # 404 on unknown SKU
        resp_404 = self.client.get("/api/forecast/UNKNOWN-999/")
        self.assertEqual(resp_404.status_code, status.HTTP_404_NOT_FOUND)

    def test_exact_gst_calculation_with_round_half_up(self):
        # Specific price testing round_half_up: 1459.50 * 0.18 = 262.71
        p_odd = Product.objects.create(
            sku="ODD-1001",
            name="Odd Price Item",
            category="Brakes",
            brand="Brembo",
            vehicle_fitment="Universal",
            price=Decimal("1459.50"),
        )
        Inventory.objects.create(product=p_odd, quantity=10)

        order_data = {
            "dealer_id": self.dealer.id,
            "items": [{"sku": "ODD-1001", "quantity": 1}],
        }
        resp = self.client.post("/api/orders/", order_data, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        data = resp.json()
        self.assertEqual(data["subtotal"], "1459.50")
        self.assertEqual(data["tax_total"], "262.71")
        self.assertEqual(data["grand_total"], "1722.21")

    def test_multiline_atomic_confirmation_rollback(self):
        # Line 1: TST-1001 (has 10, requests 5)
        # Line 2: TST-1002 (has 5, requests 10 -> exceeds!)
        order_data = {
            "dealer_id": self.dealer.id,
            "items": [
                {"sku": "TST-1001", "quantity": 5},
                {"sku": "TST-1002", "quantity": 10},
            ],
        }
        resp = self.client.post("/api/orders/", order_data, format="json")
        order_num = resp.json()["order_number"]

        # Confirm -> must fail atomically
        resp_conf = self.client.post(f"/api/orders/{order_num}/confirm/")
        self.assertEqual(resp_conf.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(resp_conf.json()["error"]["code"], "INSUFFICIENT_STOCK")

        # Crucial check: TST-1001 was NOT deducted!
        self.assertEqual(Inventory.objects.get(product=self.p1).quantity, 10)
        self.assertEqual(Inventory.objects.get(product=self.p2).quantity, 5)

    def test_price_locking_after_confirm(self):
        order_data = {
            "dealer_id": self.dealer.id,
            "items": [{"sku": "TST-1001", "quantity": 1}],
        }
        resp = self.client.post("/api/orders/", order_data, format="json")
        order_num = resp.json()["order_number"]

        # Confirm order
        self.client.post(f"/api/orders/{order_num}/confirm/")

        # Change catalogue price from 100.00 to 200.00
        self.p1.price = Decimal("200.00")
        self.p1.save()

        # Confirmed order must retain locked price 100.00
        resp_order = self.client.get(f"/api/orders/{order_num}/")
        self.assertEqual(resp_order.json()["subtotal"], "100.00")
        self.assertEqual(resp_order.json()["items"][0]["unit_price"], "100.00")

    def test_order_edit_and_lock(self):
        order_data = {
            "dealer_id": self.dealer.id,
            "items": [{"sku": "TST-1001", "quantity": 1}],
        }
        resp = self.client.post("/api/orders/", order_data, format="json")
        order_num = resp.json()["order_number"]

        # PUT replaces items
        new_items = {"items": [{"sku": "TST-1002", "quantity": 3}]}
        resp_put = self.client.put(f"/api/orders/{order_num}/", new_items, format="json")
        self.assertEqual(resp_put.status_code, status.HTTP_200_OK)
        self.assertEqual(len(resp_put.json()["items"]), 1)
        self.assertEqual(resp_put.json()["items"][0]["sku"], "TST-1002")

        # Confirm order
        self.client.post(f"/api/orders/{order_num}/confirm/")

        # PUT on confirmed order -> 409 ORDER_LOCKED
        resp_locked = self.client.put(f"/api/orders/{order_num}/", new_items, format="json")
        self.assertEqual(resp_locked.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(resp_locked.json()["error"]["code"], "ORDER_LOCKED")

    def test_invalid_status_transitions(self):
        order_data = {
            "dealer_id": self.dealer.id,
            "items": [{"sku": "TST-1001", "quantity": 1}],
        }
        resp = self.client.post("/api/orders/", order_data, format="json")
        order_num = resp.json()["order_number"]

        # Cannot dispatch draft order
        r_disp = self.client.post(f"/api/orders/{order_num}/dispatch/")
        self.assertEqual(r_disp.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(r_disp.json()["error"]["code"], "INVALID_TRANSITION")

        # Confirm order
        self.client.post(f"/api/orders/{order_num}/confirm/")

        # Cannot deliver confirmed order without dispatch
        r_del = self.client.post(f"/api/orders/{order_num}/deliver/")
        self.assertEqual(r_del.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(r_del.json()["error"]["code"], "INVALID_TRANSITION")

    def test_reorder_qty_changes_when_stock_changes(self):
        p_fil, _ = Product.objects.get_or_create(
            sku="FIL-1016",
            defaults={
                "name": "Oil Filter - Pulsar",
                "category": "Filters",
                "brand": "Bosch",
                "vehicle_fitment": "Bajaj Pulsar",
                "price": Decimal("100.00"),
            },
        )
        inv_fil, _ = Inventory.objects.get_or_create(product=p_fil, defaults={"quantity": 10})

        # Check initial reorder qty for FIL-1016
        r1 = self.client.get("/api/reorder-suggestions/")
        item1 = next(x for x in r1.json()["results"] if x["sku"] == "FIL-1016")
        initial_stock = item1["current_stock"]

        # Update stock by adding 1000 units
        resp_put = self.client.put("/api/inventory/FIL-1016/", {"quantity": initial_stock + 1000}, format="json")
        self.assertEqual(resp_put.status_code, status.HTTP_200_OK)

        # Check reorder suggestions again -> reorder_qty must now be 0
        r2 = self.client.get("/api/reorder-suggestions/")
        item2 = next(x for x in r2.json()["results"] if x["sku"] == "FIL-1016")
        self.assertEqual(item2["current_stock"], initial_stock + 1000)
        self.assertEqual(item2["reorder_qty"], 0)
