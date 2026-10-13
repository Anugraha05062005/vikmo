import csv
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Dealer, Inventory, Order, Product
from core.retrieval import invalidate_index


class Command(BaseCommand):
    help = "Load catalogue.csv and dealers.csv into PostgreSQL idempotently, cleaning transient test records."

    def handle(self, *args, **options):
        data_dir = Path(settings.DATA_DIR)
        catalogue_path = data_dir / "catalogue.csv"
        dealers_path = data_dir / "dealers.csv"

        if not catalogue_path.exists():
            self.stderr.write(self.style.ERROR(f"Catalogue file not found at {catalogue_path}"))
            return

        if not dealers_path.exists():
            self.stderr.write(self.style.ERROR(f"Dealers file not found at {dealers_path}"))
            return

        self.stdout.write(f"Seeding data from {data_dir}...")

        with transaction.atomic():
            # Clean non-catalogue orders and transient test dealers
            Order.objects.all().delete()

            # Seed Dealers
            dealer_codes_in_csv = set()
            dealer_count = 0
            with open(dealers_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dealer_code = row["dealer_code"].strip()
                    dealer_codes_in_csv.add(dealer_code)
                    credit_limit = Decimal(row.get("credit_limit", "0.00").strip() or "0.00")

                    Dealer.objects.update_or_create(
                        dealer_code=dealer_code,
                        defaults={
                            "name": row["name"].strip(),
                            "city": row.get("city", "").strip(),
                            "phone": row.get("phone", "").strip(),
                            "email": row.get("email", "").strip(),
                            "gstin": row.get("gstin", "").strip(),
                            "credit_limit": credit_limit,
                        },
                    )
                    dealer_count += 1

            # Remove test dealers created outside CSV
            Dealer.objects.exclude(dealer_code__in=dealer_codes_in_csv).delete()

            # Seed Products & Inventory
            product_skus_in_csv = set()
            product_count = 0
            with open(catalogue_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sku = row["sku"].strip()
                    product_skus_in_csv.add(sku)
                    price = Decimal(row["price"].strip())
                    stock_qty = int(row["stock"].strip())

                    product, _ = Product.objects.update_or_create(
                        sku=sku,
                        defaults={
                            "name": row["name"].strip(),
                            "category": row["category"].strip(),
                            "brand": row["brand"].strip(),
                            "vehicle_fitment": row["vehicle_fitment"].strip(),
                            "price": price,
                            "description": row.get("description", "").strip(),
                        },
                    )

                    Inventory.objects.update_or_create(
                        product=product,
                        defaults={"quantity": stock_qty},
                    )
                    product_count += 1

            # Remove products created outside CSV (e.g. from supplier sync tests)
            extra_products = Product.objects.exclude(sku__in=product_skus_in_csv)
            extra_products.delete()

            invalidate_index()

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully seeded {product_count} products (with inventory) and {dealer_count} dealers."
            )
        )
