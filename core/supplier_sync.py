"""
VIKMO DealerDesk — Supplier Synchronization Module
Pulls catalogue from supplier API with pagination, retry handling (429 Retry-After, 5xx),
duplicate SKU resolution by latest updated_at, invalid item isolation, and zero stock modification.
"""
from decimal import Decimal
import re
import time
from typing import Any
from django.conf import settings
from django.db import transaction
import requests
from requests.exceptions import RequestException

from core.exceptions import SupplierUnavailableError
from core.models import Inventory, Product
from core.retrieval import invalidate_index

MONEY_REGEX = re.compile(r"^\d+(\.\d{1,2})?$")
MAX_ATTEMPTS = 5


def parse_timestamp(ts_str: str) -> str:
    return ts_str or ""


def is_valid_price(price_val: Any) -> bool:
    if price_val is None:
        return False
    str_val = str(price_val).strip()
    if not MONEY_REGEX.match(str_val):
        return False
    try:
        dec = Decimal(str_val)
        return dec > Decimal("0.00")
    except Exception:
        return False


def run_supplier_sync() -> dict[str, Any]:
    api_url = getattr(settings, "SUPPLIER_API_URL", "").rstrip("/")
    api_key = getattr(settings, "SUPPLIER_API_KEY", "")

    if not api_url or not api_key:
        raise SupplierUnavailableError(
            message="SUPPLIER_API_URL or SUPPLIER_API_KEY not configured.",
            details=[],
        )

    session = requests.Session()
    headers = {"X-Api-Key": api_key}

    cursor = None
    all_raw_items: list[dict[str, Any]] = []
    pages_fetched = 0
    requests_made = 0

    endpoint = f"{api_url}/v1/products"

    # 1. Fetch all pages with retry logic
    while True:
        params: dict[str, Any] = {"limit": 20}
        if cursor:
            params["cursor"] = cursor

        success = False
        data = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            requests_made += 1
            try:
                resp = session.get(endpoint, headers=headers, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    pages_fetched += 1
                    success = True
                    break
                elif resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait_sec = float(retry_after) if retry_after else 1.0
                    except (ValueError, TypeError):
                        wait_sec = 1.0
                    time.sleep(max(0.5, wait_sec))
                    continue
                elif 500 <= resp.status_code <= 599:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                else:
                    # 401, 400, 404, etc. -> unrecoverable failure
                    raise SupplierUnavailableError(
                        message=f"Supplier returned status {resp.status_code}.",
                        details=[{"status_code": resp.status_code, "text": resp.text[:200]}],
                    )
            except RequestException:
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue

        if not success or data is None:
            raise SupplierUnavailableError(
                message="Supplier API unreachable after maximum retries.",
                details=[{"requests_made": requests_made, "pages_fetched": pages_fetched}],
            )

        items = data.get("items", [])
        all_raw_items.extend(items)

        cursor = data.get("next_cursor")
        if not cursor:
            break

    # 2. Process items: filter invalid items into rejected
    rejected = []
    valid_items = []

    for item in all_raw_items:
        supplier_sku = item.get("supplier_sku")
        sku = item.get("sku")
        price = item.get("price")

        if not sku or not str(sku).strip():
            rejected.append(
                {"supplier_sku": supplier_sku, "reason": "missing sku"}
            )
            continue

        if not is_valid_price(price):
            rejected.append(
                {"supplier_sku": supplier_sku, "reason": "invalid price"}
            )
            continue

        valid_items.append(item)

    # 3. Deduplicate SKUs by latest updated_at
    latest_by_sku: dict[str, dict[str, Any]] = {}
    for item in valid_items:
        sku = str(item["sku"]).strip()
        updated_at = parse_timestamp(item.get("updated_at", ""))
        if sku not in latest_by_sku:
            latest_by_sku[sku] = item
        else:
            prev_updated = parse_timestamp(latest_by_sku[sku].get("updated_at", ""))
            if updated_at > prev_updated:
                latest_by_sku[sku] = item

    # 4. Apply changes atomically in database
    created = 0
    updated = 0
    unchanged = 0

    with transaction.atomic():
        existing_products = {
            p.sku: p
            for p in Product.objects.filter(sku__in=list(latest_by_sku.keys()))
        }

        new_products = []
        new_inventories = []

        for sku, item in latest_by_sku.items():
            name = str(item.get("name", "")).strip()
            category = str(item.get("category", "")).strip()
            brand = str(item.get("brand", "")).strip()
            fitment = str(item.get("vehicle_fitment", "")).strip()
            description = str(item.get("description", "")).strip()
            price = Decimal(str(item["price"]).strip())

            if sku not in existing_products:
                prod = Product(
                    sku=sku,
                    name=name,
                    category=category,
                    brand=brand,
                    vehicle_fitment=fitment,
                    price=price,
                    description=description,
                )
                new_products.append(prod)
                new_inventories.append(Inventory(product=prod, quantity=0))
                created += 1
            else:
                existing = existing_products[sku]
                # Check if any field differs
                differs = (
                    existing.name != name
                    or existing.category != category
                    or existing.brand != brand
                    or existing.vehicle_fitment != fitment
                    or existing.description != description
                    or existing.price != price
                )

                if differs:
                    existing.name = name
                    existing.category = category
                    existing.brand = brand
                    existing.vehicle_fitment = fitment
                    existing.description = description
                    existing.price = price
                    existing.save(
                        update_fields=[
                            "name",
                            "category",
                            "brand",
                            "vehicle_fitment",
                            "description",
                            "price",
                            "updated_at",
                        ]
                    )
                    updated += 1
                else:
                    unchanged += 1

        if new_products:
            Product.objects.bulk_create(new_products)
            Inventory.objects.bulk_create(new_inventories)
            invalidate_index()

    return {
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "rejected": rejected,
        "pages_fetched": pages_fetched,
        "requests_made": requests_made,
    }
