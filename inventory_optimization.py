import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

def calculate_rop(train, lead_time=7, service_level=0.95):
    daily_demand_mean = train['demand'][-lead_time:].mean()
    lead_time_demand = train['demand'][-lead_time:].sum()
    safety_stock = 1.65 * train['demand'][-lead_time:].std() * np.sqrt(lead_time)
    ROP = lead_time_demand + safety_stock
    return ROP, safety_stock

def simulate_inventory(pred, test_dates, ROP, reorder_qty, initial_inventory=None, holding_cost=1, stockout_cost=10, order_cost=50):
    if len(pred) == 0:
        return [], [], np.nan, 0, 0
    if initial_inventory is None:
        initial_inventory = ROP * 2
    inventory = [initial_inventory]
    reorder_alerts = []
    total_holding = 0
    total_stockout = 0
    total_orders = 0
    for i, demand in enumerate(pred):
        inv = inventory[-1] - demand
        if inv < ROP:
            inv += reorder_qty
            reorder_alerts.append(test_dates.iloc[i])
            total_orders += 1
        holding = max(inv, 0) * holding_cost
        stockout = abs(min(inv, 0)) * stockout_cost
        total_holding += holding
        total_stockout += stockout
        inventory.append(inv)
    inventory = inventory[1:]
    stockouts = sum(np.array(inventory) <= 0)
    if len(inventory) == 0:
        service_level = np.nan
    else:
        service_level = 1 - stockouts/len(inventory)
    tci = total_holding + total_stockout + total_orders * order_cost
    return inventory, reorder_alerts, service_level, stockouts, tci

def scenario_test(pred, test_dates, ROP, reorder_qty, scenario="delay"):
    if scenario == "delay":
        # Simulate a supplier delay: skip one reorder
        if len(pred) == 0:
            return [], []
        skip_idx = np.random.randint(0, len(pred))
        inventory = [ROP * 2]
        reorder_alerts = []
        for i, demand in enumerate(pred):
            inv = inventory[-1] - demand
            if inv < ROP and i != skip_idx:
                inv += reorder_qty
                reorder_alerts.append(test_dates.iloc[i])
            inventory.append(inv)
        inventory = inventory[1:]
        return inventory, reorder_alerts
    elif scenario == "demand_surge":
        # Simulate a demand surge
        surge = np.random.randint(10, 30)
        pred = pred.copy()
        pred[:5] += surge
        return simulate_inventory(pred, test_dates, ROP, reorder_qty)
    else:
        return simulate_inventory(pred, test_dates, ROP, reorder_qty)

def classify_volatility(df):
    # Classify items as stable/volatile based on demand std
    item_stats = df.groupby('item_id')['demand'].agg(['mean', 'std'])
    item_stats['volatility'] = np.where(item_stats['std'] > item_stats['mean']*0.5, 'volatile', 'stable')
    return item_stats[['volatility']]

def cluster_products(df, n_clusters=3):
    # Cluster items by mean and std of demand
    item_stats = df.groupby('item_id')['demand'].agg(['mean', 'std'])
    if item_stats.shape[0] == 0:
        item_stats['cluster'] = np.nan
        return item_stats
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    item_stats['cluster'] = kmeans.fit_predict(item_stats)
    return item_stats

def plot_inventory(test_dates, inventory, ROP, reorder_alerts):
    plt.figure(figsize=(12,5))
    plt.plot(test_dates, inventory, label='Projected Inventory', marker='o')
    plt.axhline(ROP, color='red', linestyle='--', label='ROP')
    for alert in reorder_alerts:
        plt.axvline(alert, color='purple', linestyle=':', alpha=0.5)
    plt.title('Inventory Simulation and Reorder Alerts')
    plt.xlabel('Date')
    plt.ylabel('Inventory Level')
    plt.legend()
    plt.show()

def plot_service_level(service_level, stockouts):
    import seaborn as sns
    sns.barplot(x=['Service Level', 'Stockout Days'], y=[service_level*100, stockouts])
    plt.title('Service Level and Stockout Risk')
    plt.ylabel('Value')
    plt.show()

if __name__ == "__main__":
    import sys
    df = pd.read_csv("sales_long.csv", parse_dates=['date'])
    from forecasting import sarimax_forecast
    item_id = 'item_1'
    store_id = 'store_1'
    df_target = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].sort_values('date')
    train = df_target.iloc[:-30]
    test = df_target.iloc[-30:]
    pred, _ = sarimax_forecast(train, test)
    ROP, safety_stock = calculate_rop(train)
    inventory, reorder_alerts, service_level, stockouts, tci = simulate_inventory(pred, test['date'], ROP, ROP)
    print(f"ROP: {ROP:.2f}, Service Level: {service_level*100:.1f}%, Stockout Days: {stockouts}, TCI: {tci:.2f}")
    # Scenario test
    inv_scenario, alerts_scenario = scenario_test(pred, test['date'], ROP, ROP, scenario='delay')
    # ML
    print(classify_volatility(df))
    print(cluster_products(df))
    plot_inventory(test['date'], inventory, ROP, reorder_alerts)
    plot_service_level(service_level, stockouts)
