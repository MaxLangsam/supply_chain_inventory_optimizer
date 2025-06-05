from data_generation import generate_synthetic_sales, melt_sales, generate_calendar, join_sales_calendar
from forecasting import compare_models, get_best_model, sarimax_forecast
from inventory_optimization import calculate_rop, simulate_inventory, scenario_test, classify_volatility, cluster_products
from visualization import plot_forecast, plot_model_comparison, plot_inventory, plot_costs, plot_clusters
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # 1. Data Generation with smaller parameters for testing
    print("Generating synthetic data...")
    sales_df, n_days, aux = generate_synthetic_sales(
        n_departments=2,
        n_categories=4,
        n_items=5,  # Reduced number of items
        n_stores=2,  # Reduced number of stores
        n_days=365,  # One year of data
        seed=42
    )
    
    print("Melting sales data...")
    sales_long = melt_sales(sales_df, n_days)
    
    print("Generating calendar...")
    calendar = generate_calendar(n_days, aux=aux)
    
    print("Joining sales and calendar data...")
    sales_long = join_sales_calendar(sales_long, calendar)
    
    # Ensure date column is datetime
    sales_long['date'] = pd.to_datetime(sales_long['date'])
    
    # Save the data
    print("Saving data to CSV files...")
    sales_long.to_csv("sales_long.csv", index=False)
    calendar.to_csv("calendar.csv", index=False)
    
    # Print data summary
    print("\nData Summary:")
    print(f"Number of unique items: {len(sales_long['item_id'].unique())}")
    print(f"Number of unique stores: {len(sales_long['store_id'].unique())}")
    print(f"Total number of records: {len(sales_long)}")
    print(f"Date range: {sales_long['date'].min()} to {sales_long['date'].max()}")
    
    # 2. Forecasting
    print("\nStarting forecasting analysis...")
    # Initialize variables
    df_target = None
    item_id = None
    store_id = None
    
    # Find a valid item-store combination with sufficient data
    for item in sales_long['item_id'].unique():
        for store in sales_long['store_id'].unique():
            temp_df = sales_long[(sales_long['item_id'] == item) & (sales_long['store_id'] == store)].copy()
            if len(temp_df) >= 30:
                df_target = temp_df
                item_id = item
                store_id = store
                print(f"Selected item {item_id} at store {store_id} with {len(temp_df)} days of data")
                break
        if df_target is not None:
            break
    
    # Check if we found valid data
    if df_target is None or len(df_target) < 30:
        print("No item-store combination found with sufficient data (minimum 30 days).")
        exit()
    
    # Continue with the rest of the analysis
    print("\nPerforming forecasting analysis...")
    train = df_target.iloc[:-30].copy()
    test = df_target.iloc[-30:].copy()
    
    print("Comparing models...")
    results = compare_models(train, test)
    plot_model_comparison(results)
    
    best_model, _ = get_best_model(results)
    print(f"Best model: {best_model}")
    
    print("Generating SARIMAX forecast...")
    pred, conf_int = sarimax_forecast(train, test)
    plot_forecast(train, test, pred, conf_int, item_id, store_id, model_name="SARIMAX")
    
    # 3. Inventory Optimization
    print("\nPerforming inventory optimization...")
    ROP, safety_stock = calculate_rop(train)
    inventory, reorder_alerts, service_level, stockouts, tci = simulate_inventory(pred, test['date'], ROP, ROP)
    plot_inventory(test['date'], inventory, ROP, reorder_alerts)
    print(f"ROP: {ROP:.2f}, Service Level: {service_level*100:.1f}%, Stockout Days: {stockouts}, TCI: {tci:.2f}")
    
    # 4. Scenario Testing
    print("\nPerforming scenario testing...")
    inv_delay, alerts_delay = scenario_test(pred, test['date'], ROP, ROP, scenario='delay')
    inv_surge, _, _, _, tci_surge = scenario_test(pred, test['date'], ROP, ROP, scenario='demand_surge')
    plot_costs([tci, tci_surge], ['Base', 'Demand Surge'])
    
    # 5. ML: Volatility & Clustering
    print("\nPerforming ML analysis...")
    print(classify_volatility(sales_long))
    item_stats = cluster_products(sales_long)
    plot_clusters(item_stats)
