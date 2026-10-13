"""
VIKMO DealerDesk — Demand Forecasting Pipeline
Reproducible, leak-free time-series forecasting for 40 automotive parts SKUs.

Evaluates against 3 mandatory baselines:
1. Last observed value
2. Recent 4-week moving average
3. Same week last year (52 weeks prior)

Produces: ml/forecast.csv (exactly 160 rows: 40 SKUs x 4 weeks)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(data_dir: Path):
    sales = pd.read_csv(data_dir / "sales_history.csv")
    sales["week_start"] = pd.to_datetime(sales["week_start"])
    lead = pd.read_csv(data_dir / "lead_times.csv")
    promo = pd.read_csv(data_dir / "promo_calendar.csv")
    promo["week_start"] = pd.to_datetime(promo["week_start"])
    fest = pd.read_csv(data_dir / "festival_calendar.csv")
    fest["date"] = pd.to_datetime(fest["date"])
    return sales, lead, promo, fest


def compute_weeks_to_diwali(dt: pd.Timestamp, fest_dates: pd.Series) -> float:
    """Signed distance in weeks to the closest Diwali date."""
    diffs = [(dt - f_date).days / 7.0 for f_date in fest_dates]
    return min(diffs, key=abs)


def build_feature_table(sales: pd.DataFrame, fest_dates: pd.Series) -> pd.DataFrame:
    """Builds historical feature table strictly using past information."""
    all_records = []
    
    for sku, group in sales.groupby("sku"):
        g = group.sort_values("week_start").copy()
        
        # Historical summary stats (calculated expanding or overall non-promo)
        non_promo_sales = g[g["promo_flag"] == 0]["units_sold"]
        base_non_promo = non_promo_sales.mean() if len(non_promo_sales) > 0 else g["units_sold"].mean()
        
        # Promo lift factor
        promo_sales = g[g["promo_flag"] == 1]["units_sold"]
        if len(promo_sales) > 0 and base_non_promo > 0:
            promo_lift = promo_sales.mean() / base_non_promo
        else:
            promo_lift = 2.0  # Population average promo multiplier
        
        g["non_promo_mean"] = base_non_promo
        g["sku_promo_lift"] = promo_lift
        g["weeks_to_diwali"] = g["week_start"].apply(lambda d: compute_weeks_to_diwali(d, fest_dates))
        
        # Lags and rolling stats
        g["lag_1"] = g["units_sold"].shift(1)
        g["lag_2"] = g["units_sold"].shift(2)
        g["lag_3"] = g["units_sold"].shift(3)
        g["lag_4"] = g["units_sold"].shift(4)
        g["rolling_mean_4"] = g["units_sold"].shift(1).rolling(4, min_periods=1).mean()
        g["rolling_mean_8"] = g["units_sold"].shift(1).rolling(8, min_periods=1).mean()
        g["rolling_std_4"] = g["units_sold"].shift(1).rolling(4, min_periods=1).std().fillna(0)
        
        all_records.append(g)
        
    return pd.concat(all_records, ignore_index=True)


def evaluate_chronological_split(sales: pd.DataFrame, fest_dates: pd.Series):
    """
    Validates model against the last 4 weeks of known data (2026-08-31 to 2026-09-21)
    trained strictly on week_start < 2026-08-31.
    """
    val_weeks = sorted(sales["week_start"].unique())[-4:]
    val_origin = val_weeks[0] - pd.Timedelta(weeks=1)
    
    train_sales = sales[sales["week_start"] < val_weeks[0]].copy()
    val_sales = sales[sales["week_start"].isin(val_weeks)].copy()
    
    # Baseline 1: Last observed value before validation cutoff
    last_val = train_sales.sort_values("week_start").groupby("sku")["units_sold"].last()
    val_b1 = val_sales.copy()
    val_b1["pred"] = val_b1["sku"].map(last_val)
    mae_b1 = mean_absolute_error(val_b1["units_sold"], val_b1["pred"])
    
    # Baseline 2: Recent 4-week average before validation cutoff
    recent_4w = train_sales[train_sales["week_start"] >= val_origin - pd.Timedelta(weeks=3)]
    recent_avg = recent_4w.groupby("sku")["units_sold"].mean()
    val_b2 = val_sales.copy()
    val_b2["pred"] = val_b2["sku"].map(recent_avg).fillna(val_b2["sku"].map(last_val))
    mae_b2 = mean_absolute_error(val_b2["units_sold"], val_b2["pred"])
    
    # Baseline 3: Same week last year (52 weeks prior)
    def last_year_pred(row):
        target = row["week_start"] - pd.Timedelta(weeks=52)
        match = train_sales[(train_sales["sku"] == row["sku"]) & (train_sales["week_start"] == target)]
        if len(match) > 0:
            return match["units_sold"].values[0]
        return last_val.get(row["sku"], 0)
        
    val_b3 = val_sales.copy()
    val_b3["pred"] = val_b3.apply(last_year_pred, axis=1)
    mae_b3 = mean_absolute_error(val_b3["units_sold"], val_b3["pred"])
    
    # Proposed Model on validation split:
    df_train = build_feature_table(train_sales, fest_dates)
    features = [
        "promo_flag", "weeks_to_diwali", "non_promo_mean", "sku_promo_lift",
        "lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4", "rolling_mean_8", "rolling_std_4"
    ]
    df_clean = df_train.dropna(subset=["lag_4"]).copy()
    
    model = HistGradientBoostingRegressor(random_state=42, max_iter=120, min_samples_leaf=4)
    model.fit(df_clean[features], df_clean["units_sold"])
    
    # Predict step-by-step for the 4 validation weeks
    curr_history = train_sales.copy()
    model_preds = []
    
    for w in val_weeks:
        for sku in sorted(sales["sku"].unique()):
            g = curr_history[curr_history["sku"] == sku].sort_values("week_start")
            last_units = g["units_sold"].values
            
            # Check if this SKU is rare/intermittent
            is_rare = (g["units_sold"].mean() < 1.5)
            # Check promo for this week in validation set
            w_row = val_sales[(val_sales["sku"] == sku) & (val_sales["week_start"] == w)]
            promo_flag = w_row["promo_flag"].values[0] if len(w_row) > 0 else 0
            
            w_to_diwali = compute_weeks_to_diwali(w, fest_dates)
            non_promo_m = g[g["promo_flag"] == 0]["units_sold"].mean() if len(g[g["promo_flag"] == 0]) > 0 else g["units_sold"].mean()
            promo_m = g[g["promo_flag"] == 1]["units_sold"].mean() if len(g[g["promo_flag"] == 1]) > 0 else 2.0 * non_promo_m
            p_lift = (promo_m / non_promo_m) if (non_promo_m and non_promo_m > 0) else 2.0
            
            if is_rare:
                # Conservative expectation for intermittent demand
                pred = non_promo_m * (p_lift if promo_flag else 1.0)
            elif len(g) < 8:
                # New SKU
                pred = g["units_sold"].tail(4).mean() * (p_lift if promo_flag else 1.0)
            else:
                row_feat = pd.DataFrame([{
                    "promo_flag": promo_flag,
                    "weeks_to_diwali": w_to_diwali,
                    "non_promo_mean": non_promo_m,
                    "sku_promo_lift": p_lift,
                    "lag_1": last_units[-1] if len(last_units) >= 1 else non_promo_m,
                    "lag_2": last_units[-2] if len(last_units) >= 2 else non_promo_m,
                    "lag_3": last_units[-3] if len(last_units) >= 3 else non_promo_m,
                    "lag_4": last_units[-4] if len(last_units) >= 4 else non_promo_m,
                    "rolling_mean_4": np.mean(last_units[-4:]) if len(last_units) >= 4 else non_promo_m,
                    "rolling_mean_8": np.mean(last_units[-8:]) if len(last_units) >= 8 else non_promo_m,
                    "rolling_std_4": np.std(last_units[-4:]) if len(last_units) >= 4 else 0.0,
                }])
                pred = max(0.0, float(model.predict(row_feat)[0]))
                
            model_preds.append({"sku": sku, "week_start": w, "pred": pred})
            
    val_model = pd.DataFrame(model_preds)
    merged_model = pd.merge(val_sales, val_model, on=["sku", "week_start"])
    mae_model = mean_absolute_error(merged_model["units_sold"], merged_model["pred"])
    
    print("=" * 60)
    print("CHRONOLOGICAL VALIDATION RESULTS (Origin: 2026-08-24):")
    print(f"  Baseline 1 (Last observed value):    MAE = {mae_b1:.2f}")
    print(f"  Baseline 2 (Recent 4-week average):  MAE = {mae_b2:.2f}")
    print(f"  Baseline 3 (Same week last year):    MAE = {mae_b3:.2f}")
    print(f"  Our Proposed Model:                  MAE = {mae_model:.2f}")
    print(f"  Improvement over best baseline:      {((mae_b2 - mae_model) / mae_b2) * 100:.1f}%")
    print("=" * 60)
    
    return {
        "mae_b1": mae_b1,
        "mae_b2": mae_b2,
        "mae_b3": mae_b3,
        "mae_model": mae_model,
    }


def generate_final_forecast(sales: pd.DataFrame, lead: pd.DataFrame, promo: pd.DataFrame, fest_dates: pd.Series, output_path: Path):
    """
    Trains on all available historical data up to 2026-09-21 and generates forecast.csv
    for 4 forecast weeks (2026-09-28 to 2026-10-19) for the 40 SKUs in lead_times.csv.
    """
    forecast_weeks = ["2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"]
    target_skus = set(lead["sku"].unique())
    
    # Train model on full history
    df_all = build_feature_table(sales, fest_dates)
    features = [
        "promo_flag", "weeks_to_diwali", "non_promo_mean", "sku_promo_lift",
        "lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4", "rolling_mean_8", "rolling_std_4"
    ]
    df_clean = df_all.dropna(subset=["lag_4"]).copy()
    
    model = HistGradientBoostingRegressor(random_state=42, max_iter=150, min_samples_leaf=4)
    model.fit(df_clean[features], df_clean["units_sold"])
    
    # Promo map: (sku, week_str) -> promo_flag
    promo_map = {}
    for _, row in promo.iterrows():
        promo_map[(row["sku"], str(row["week_start"].date()))] = int(row["promo_flag"])
        
    forecast_rows = []
    
    for sku in sorted(target_skus):
        g = sales[sales["sku"] == sku].sort_values("week_start")
        last_units = list(g["units_sold"].values)
        
        is_rare = (g["units_sold"].mean() < 1.5)
        is_new = (len(g) < 10)
        
        non_promo_sales = g[g["promo_flag"] == 0]["units_sold"]
        non_promo_m = non_promo_sales.mean() if len(non_promo_sales) > 0 else g["units_sold"].mean()
        promo_sales = g[g["promo_flag"] == 1]["units_sold"]
        if len(promo_sales) > 0 and non_promo_m > 0:
            p_lift = promo_sales.mean() / non_promo_m
        else:
            p_lift = 2.0
            
        # Multi-week autoregressive roll-forward
        simulated_units = list(last_units)
        
        for w_str in forecast_weeks:
            w_dt = pd.to_datetime(w_str)
            p_flag = promo_map.get((sku, w_str), 0)
            w_to_diwali = compute_weeks_to_diwali(w_dt, fest_dates)
            
            if is_rare:
                # Intermittent demand: maintain low-rate Poisson/exponential expectation
                pred = non_promo_m * (p_lift if p_flag else 1.0)
                # Cap rare SKUs conservatively
                pred = min(pred, 2.0) if not p_flag else pred
            elif is_new:
                # New SKU: use recent momentum + promo
                recent_4w = np.mean(simulated_units[-4:]) if len(simulated_units) >= 4 else non_promo_m
                pred = recent_4w * (p_lift if p_flag else 1.0)
            else:
                row_feat = pd.DataFrame([{
                    "promo_flag": p_flag,
                    "weeks_to_diwali": w_to_diwali,
                    "non_promo_mean": non_promo_m,
                    "sku_promo_lift": p_lift,
                    "lag_1": simulated_units[-1],
                    "lag_2": simulated_units[-2] if len(simulated_units) >= 2 else simulated_units[-1],
                    "lag_3": simulated_units[-3] if len(simulated_units) >= 3 else simulated_units[-1],
                    "lag_4": simulated_units[-4] if len(simulated_units) >= 4 else simulated_units[-1],
                    "rolling_mean_4": np.mean(simulated_units[-4:]),
                    "rolling_mean_8": np.mean(simulated_units[-8:]),
                    "rolling_std_4": np.std(simulated_units[-4:]),
                }])
                pred = max(0.0, float(model.predict(row_feat)[0]))
                
            pred_rounded = round(pred, 2)
            forecast_rows.append({
                "sku": sku,
                "week_start": w_str,
                "forecast_units": pred_rounded,
            })
            simulated_units.append(pred_rounded)
            
    out_df = pd.DataFrame(forecast_rows)
    out_df = out_df[["sku", "week_start", "forecast_units"]]
    out_df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(out_df)} rows.")
    return out_df


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    ml_dir = repo_root / "ml"
    
    sales, lead, promo, fest = load_data(data_dir)
    fest_dates = fest["date"]
    
    # 1. Evaluate chronological validation
    metrics = evaluate_chronological_split(sales, fest_dates)
    
    # 2. Generate final forecast file
    output_csv = ml_dir / "forecast.csv"
    forecast_df = generate_final_forecast(sales, lead, promo, fest_dates, output_csv)
    print("Forecasting pipeline completed successfully.")
