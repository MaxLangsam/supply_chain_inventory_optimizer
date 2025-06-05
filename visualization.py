import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create plot_images directory if it doesn't exist
os.makedirs('plot_images', exist_ok=True)

def plot_forecast(train, test, pred, conf_int, item_id, store_id, model_name="Model"):
    fig = plt.figure(figsize=(12,5))
    plt.plot(train['date'], train['demand'], label='Train')
    plt.plot(test['date'], test['demand'], label='Test', color='orange')
    plt.plot(test['date'], pred, label=f'Forecast ({model_name})', color='green')
    if conf_int is not None:
        plt.fill_between(test['date'], conf_int.iloc[:,0], conf_int.iloc[:,1], color='green', alpha=0.2)
    plt.legend()
    plt.title(f'Demand Forecasting for {item_id} at {store_id}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    # Save the plot
    plt.savefig('plot_images/forecast.png', bbox_inches='tight', dpi=300)
    return fig

def plot_model_comparison(results):
    fig = plt.figure(figsize=(10,5))
    df = pd.DataFrame(results).T.rename(columns={0: 'MAE', 1: 'RMSE'})
    df.plot(kind='bar', ax=plt.gca())
    plt.title('Forecast Model Comparison')
    plt.ylabel('Error')
    # Save the plot
    plt.savefig('plot_images/model_comparison.png', bbox_inches='tight', dpi=300)
    return fig

def plot_inventory(test_dates, inventory, ROP, reorder_alerts):
    fig = plt.figure(figsize=(12,5))
    plt.plot(test_dates, inventory, label='Projected Inventory', marker='o')
    plt.axhline(ROP, color='red', linestyle='--', label='ROP')
    for alert in reorder_alerts:
        plt.axvline(alert, color='purple', linestyle=':', alpha=0.5)
    plt.title('Inventory Simulation and Reorder Alerts')
    plt.xlabel('Date')
    plt.ylabel('Inventory Level')
    plt.legend()
    # Save the plot
    plt.savefig('plot_images/inventory.png', bbox_inches='tight', dpi=300)
    return fig

def plot_costs(tci_list, labels):
    fig = plt.figure(figsize=(10,5))
    plt.bar(labels, tci_list)
    plt.title('Total Cost of Inventory (TCI) by Scenario')
    plt.ylabel('Cost')
    # Save the plot
    plt.savefig('plot_images/costs.png', bbox_inches='tight', dpi=300)
    return fig

def plot_clusters(item_stats):
    fig = plt.figure(figsize=(10,6))
    if item_stats.empty or 'cluster' not in item_stats.columns or item_stats['cluster'].isnull().all():
        plt.title('No data available for clustering')
        plt.xlabel('Mean Demand')
        plt.ylabel('Std Demand')
        # Save the plot
        plt.savefig('plot_images/clusters.png', bbox_inches='tight', dpi=300)
        return fig
    
    sns.scatterplot(x='mean', y='std', hue='cluster', data=item_stats, palette='tab10')
    plt.title('Product Clusters by Demand Mean/Std')
    plt.xlabel('Mean Demand')
    plt.ylabel('Std Demand')
    # Save the plot
    plt.savefig('plot_images/clusters.png', bbox_inches='tight', dpi=300)
    return fig

def plot_service_level(service_level, stockouts):
    fig = plt.figure(figsize=(8,5))
    sns.barplot(x=['Service Level', 'Stockout Days'], y=[service_level*100, stockouts])
    plt.title('Service Level and Stockout Risk')
    plt.ylabel('Value')
    # Save the plot
    plt.savefig('plot_images/service_level.png', bbox_inches='tight', dpi=300)
    return fig
