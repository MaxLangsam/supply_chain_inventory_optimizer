import numpy as np
import pandas as pd

def generate_synthetic_sales(
    n_departments=2, n_categories=4, n_items=10, n_stores=3, n_days=1095, seed=42
):
    np.random.seed(seed)
    # Hierarchy
    departments = [f"dept_{i+1}" for i in range(n_departments)]
    categories = [f"cat_{i+1}" for i in range(n_categories)]
    items = [f"item_{i+1}" for i in range(n_items)]
    stores = [f"store_{i+1}" for i in range(n_stores)]
    # Assign categories to departments, items to categories
    cat_to_dept = {cat: np.random.choice(departments) for cat in categories}
    item_to_cat = {item: np.random.choice(categories) for item in items}
    item_to_dept = {item: cat_to_dept[item_to_cat[item]] for item in items}
    # Supplier lead times (random per item)
    item_to_supplier = {item: f"supplier_{np.random.randint(1,4)}" for item in items}
    supplier_lead_time = {f"supplier_{i+1}": np.random.randint(5,15) for i in range(3)}
    # Price changes (random walk)
    base_prices = {item: np.random.uniform(10, 100) for item in items}
    # Weather (affects demand)
    weather = np.random.normal(0, 1, n_days)
    # Store capacity
    store_capacity = {store: np.random.randint(200, 400) for store in stores}
    # Holidays and promotions
    holidays = np.random.choice(np.arange(n_days), size=10, replace=False)
    promotions = np.random.choice(np.arange(n_days), size=20, replace=False)
    # Data generation
    data = []
    for item in items:
        for store in stores:
            base = np.random.randint(20, 50)
            seasonality = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
            noise = np.random.normal(0, 3, n_days)
            spikes = np.zeros(n_days)
            spike_days = np.random.choice(n_days, size=5, replace=False)
            spikes[spike_days] = np.random.randint(30, 60, size=5)
            # Promotions boost
            promo_boost = np.zeros(n_days)
            promo_boost[promotions] = np.random.randint(10, 30)
            # Holidays drop
            holiday_drop = np.zeros(n_days)
            holiday_drop[holidays] = -np.random.randint(5, 15)
            # Price effect (random walk)
            price = base_prices[item] + np.cumsum(np.random.normal(0, 0.2, n_days))
            price_effect = 1 - (price - price.mean()) / (2 * price.std())
            # Weather effect
            weather_effect = 1 + 0.05 * weather
            # Capacity limit
            demand = base + seasonality + noise + spikes + promo_boost + holiday_drop
            demand = demand * price_effect * weather_effect
            demand = np.clip(np.round(demand), 0, store_capacity[store])
            row = [
                item, item_to_cat[item], item_to_dept[item], store, 
                item_to_supplier[item], supplier_lead_time[item_to_supplier[item]]
            ] + price.tolist() + demand.tolist()
            data.append(row)
    columns = (
        ["item_id", "category_id", "department_id", "store_id", "supplier_id", "lead_time"] +
        [f"price_d_{i+1}" for i in range(n_days)] +
        [f"d_{i+1}" for i in range(n_days)]
    )
    sales_df = pd.DataFrame(data, columns=columns)
    # Save auxiliary info
    aux = {
        "holidays": holidays,
        "promotions": promotions,
        "weather": weather,
        "store_capacity": store_capacity
    }
    return sales_df, n_days, aux

def melt_sales(sales_df, n_days):
    id_vars = ["item_id", "category_id", "department_id", "store_id", "supplier_id", "lead_time"]
    price_cols = [f"price_d_{i+1}" for i in range(n_days)]
    demand_cols = [f"d_{i+1}" for i in range(n_days)]
    sales_long = sales_df.melt(
        id_vars=id_vars + price_cols, value_vars=demand_cols,
        var_name="d", value_name="demand"
    )
    # Extract d_num, allow for NaN, then drop rows with NaN before converting to int
    sales_long["d_num"] = sales_long["d"].str.extract(r'd_(\d+)').astype('Int64')
    sales_long = sales_long.dropna(subset=["d_num"])
    sales_long["d_num"] = sales_long["d_num"].astype(int)
    # Extract price for each day
    sales_long["price"] = sales_long.apply(lambda row: row[f"price_d_{row['d_num']}"] if pd.notnull(row["d_num"]) else np.nan, axis=1)
    sales_long = sales_long[id_vars + ["d_num", "demand", "price"]]
    return sales_long

def generate_calendar(n_days, start_date="2017-01-01", aux=None):
    dates = pd.date_range(start_date, periods=n_days)
    calendar = pd.DataFrame({
        "d_num": np.arange(1, n_days+1),
        "date": dates,
        "weekday": dates.day_name()
    })
    if aux:
        calendar["holiday"] = calendar["d_num"].isin(aux["holidays"])
        calendar["promotion"] = calendar["d_num"].isin(aux["promotions"])
        calendar["weather"] = aux["weather"]
    return calendar

def join_sales_calendar(sales_long, calendar):
    sales_long = sales_long.merge(calendar, on="d_num", how="left")
    return sales_long

if __name__ == "__main__":
    sales_df, n_days, aux = generate_synthetic_sales()
    sales_long = melt_sales(sales_df, n_days)
    calendar = generate_calendar(n_days, aux=aux)
    sales_long = join_sales_calendar(sales_long, calendar)
    sales_long.to_csv("sales_long.csv", index=False)
    calendar.to_csv("calendar.csv", index=False)
