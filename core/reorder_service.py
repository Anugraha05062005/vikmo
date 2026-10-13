import csv
import math
from pathlib import Path
from django.conf import settings
import pandas as pd

from core.exceptions import NotFoundError
from core.models import Inventory

_FORECAST_CACHE = None
_LEAD_TIMES_CACHE = None
_SIGMA_CACHE = None


def load_reorder_metadata():
    global _FORECAST_CACHE, _LEAD_TIMES_CACHE, _SIGMA_CACHE
    if _FORECAST_CACHE is not None and _LEAD_TIMES_CACHE is not None and _SIGMA_CACHE is not None:
        return _FORECAST_CACHE, _LEAD_TIMES_CACHE, _SIGMA_CACHE

    base_dir = Path(settings.BASE_DIR)
    data_dir = Path(settings.DATA_DIR)
    forecast_path = base_dir / "ml" / "forecast.csv"
    lead_path = data_dir / "lead_times.csv"
    sales_path = data_dir / "sales_history.csv"

    # 1. Lead times
    lead_times = {}
    with open(lead_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lead_times[row["sku"].strip()] = int(row["lead_time_weeks"].strip())

    # 2. Forecasts
    forecasts = {}
    if forecast_path.exists():
        with open(forecast_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sku = row["sku"].strip()
                w = row["week_start"].strip()
                u = float(row["forecast_units"].strip())
                if sku not in forecasts:
                    forecasts[sku] = []
                forecasts[sku].append({"week_start": w, "forecast_units": u})

    # Sort each SKU's forecast by week_start
    for sku in forecasts:
        forecasts[sku].sort(key=lambda x: x["week_start"])

    # 3. Sigmas over most recent 12 weeks of sales_history.csv
    sigmas = {}
    if sales_path.exists():
        df_sales = pd.read_csv(sales_path)
        df_sales["week_start"] = pd.to_datetime(df_sales["week_start"])
        recent_weeks = sorted(df_sales["week_start"].unique())[-12:]
        df_recent = df_sales[df_sales["week_start"].isin(recent_weeks)]

        for sku in lead_times.keys():
            s = df_recent[df_recent["sku"] == sku]["units_sold"]
            if len(s) < 2:
                s = df_sales[df_sales["sku"] == sku]["units_sold"]
            if len(s) > 1:
                sigmas[sku] = float(s.std(ddof=1))
            else:
                sigmas[sku] = 0.0

    _FORECAST_CACHE = forecasts
    _LEAD_TIMES_CACHE = lead_times
    _SIGMA_CACHE = sigmas
    return _FORECAST_CACHE, _LEAD_TIMES_CACHE, _SIGMA_CACHE


def get_forecast_for_sku(sku: str) -> dict:
    forecasts, lead_times, _ = load_reorder_metadata()
    if sku not in lead_times or sku not in forecasts:
        raise NotFoundError(message=f"Forecast for SKU '{sku}' not found.")

    return {
        "sku": sku,
        "forecast": forecasts[sku],
    }


def get_reorder_suggestions() -> dict:
    forecasts, lead_times, sigmas = load_reorder_metadata()

    # Query live available stock from PostgreSQL
    skus = list(lead_times.keys())
    stock_map = {
        inv.product_id: inv.quantity
        for inv in Inventory.objects.filter(product_id__in=skus)
    }

    results = []
    for sku in skus:
        L = lead_times[sku]
        sku_fc = forecasts.get(sku, [])
        # sum of forecast_units for the first L forecast weeks
        first_l_fc = sku_fc[:L]
        ltd_sum = sum(item["forecast_units"] for item in first_l_fc)
        lead_time_demand = round(ltd_sum, 1)

        sigma = sigmas.get(sku, 0.0)
        safety_stock = math.ceil(1.65 * sigma * math.sqrt(L))
        reorder_point = round(lead_time_demand + safety_stock, 1)

        current_stock = stock_map.get(sku, 0)
        reorder_qty = max(0, math.ceil(reorder_point - current_stock))

        results.append(
            {
                "sku": sku,
                "current_stock": current_stock,
                "lead_time_weeks": L,
                "lead_time_demand": lead_time_demand,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "reorder_qty": reorder_qty,
            }
        )

    return {"results": results}
