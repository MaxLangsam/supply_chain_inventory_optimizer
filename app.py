import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_generation import generate_synthetic_sales, melt_sales, generate_calendar, join_sales_calendar
from forecasting import compare_models, get_best_model, sarimax_forecast
from inventory_optimization import calculate_rop, simulate_inventory, scenario_test, classify_volatility, cluster_products
from visualization import plot_forecast, plot_model_comparison, plot_inventory, plot_costs, plot_clusters

st.set_page_config(page_title="Supply Chain Inventory Optimization", layout="wide")

st.title("📦 Supply Chain Inventory Optimization Dashboard")

# Data loading/generation
@st.cache_data
def load_data():
    # Generate synthetic data with smaller parameters for better performance
    sales_df, n_days, aux = generate_synthetic_sales(
        n_departments=2,
        n_categories=4,
        n_items=5,
        n_stores=2,
        n_days=365,
        seed=42
    )
    sales_long = melt_sales(sales_df, n_days)
    calendar = generate_calendar(n_days, aux=aux)
    sales_long = join_sales_calendar(sales_long, calendar)
    
    # Ensure date column is datetime
    sales_long['date'] = pd.to_datetime(sales_long['date'])
    return sales_long, calendar

# Load data
try:
    sales_long, calendar = load_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar: Select item/store
item_ids = sorted(sales_long['item_id'].unique())
store_ids = sorted(sales_long['store_id'].unique())

st.sidebar.header("Configuration")
item_id = st.sidebar.selectbox("Select Item", item_ids)
store_id = st.sidebar.selectbox("Select Store", store_ids)

# Get target data
df_target = sales_long[(sales_long['item_id'] == item_id) & 
                      (sales_long['store_id'] == store_id)].sort_values('date').copy()

if len(df_target) < 30:
    st.error(f"Not enough data for {item_id} at {store_id}. Please select another combination.")
    st.stop()

train = df_target.iloc[:-30].copy()
test = df_target.iloc[-30:].copy()

st.header(f"Forecasting for {item_id} at {store_id}")

# Forecasting model comparison
with st.spinner("Comparing forecasting models..."):
    results = compare_models(train, test)
    
    # Display results in a more readable format
    results_df = pd.DataFrame(results, index=['MAE', 'RMSE']).T
    results_df = results_df.round(2)
    st.subheader("Model Comparison (MAE, RMSE)")
    st.dataframe(results_df)
    
    best_model, (best_mae, best_rmse) = get_best_model(results)
    st.success(f"Best model: {best_model} (MAE: {best_mae:.2f}, RMSE: {best_rmse:.2f})")

    # Plot model comparison
    fig = plot_model_comparison(results)
    st.pyplot(fig)
    plt.close(fig)

# SARIMAX forecast for inventory simulation
with st.spinner("Generating SARIMAX forecast..."):
    pred, conf_int = sarimax_forecast(train, test)
    st.subheader("SARIMAX Forecast")
    fig = plot_forecast(train, test, pred, conf_int, item_id, store_id, model_name="SARIMAX")
    st.pyplot(fig)
    plt.close(fig)

# Inventory parameters
st.sidebar.header("Inventory Parameters")
lead_time = st.sidebar.slider("Lead Time (days)", 1, 21, 7)
service_level = st.sidebar.slider("Service Level", 0.8, 0.99, 0.95)
holding_cost = st.sidebar.number_input("Holding Cost per Unit", 0.1, 10.0, 1.0)
stockout_cost = st.sidebar.number_input("Stockout Cost per Unit", 1.0, 100.0, 10.0)
order_cost = st.sidebar.number_input("Order Cost", 1.0, 500.0, 50.0)

# Calculate ROP and simulate inventory
with st.spinner("Calculating inventory parameters..."):
    ROP, safety_stock = calculate_rop(train, lead_time=lead_time, service_level=service_level)
    inventory, reorder_alerts, service_level_val, stockouts, tci = simulate_inventory(
        pred, test['date'], ROP, ROP, 
        holding_cost=holding_cost, 
        stockout_cost=stockout_cost, 
        order_cost=order_cost
    )

st.header("Inventory Simulation")
st.markdown(f"""
**Key Metrics:**
- **ROP:** {ROP:.2f}
- **Service Level:** {service_level_val*100:.1f}%
- **Stockout Days:** {stockouts}
- **Total Cost of Inventory (TCI):** ${tci:.2f}
""")

fig = plot_inventory(test['date'], inventory, ROP, reorder_alerts)
st.pyplot(fig)
plt.close(fig)

# Scenario testing
st.header("Scenario Testing")
scenario = st.selectbox("Select Scenario", ["Base", "Supplier Delay", "Demand Surge"])

with st.spinner("Running scenario analysis..."):
    if scenario == "Base":
        inv, alerts, _, _, tci_scenario = simulate_inventory(
            pred, test['date'], ROP, ROP,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            order_cost=order_cost
        )
    elif scenario == "Supplier Delay":
        inv, alerts = scenario_test(pred, test['date'], ROP, ROP, scenario='delay')
        tci_scenario = None
    elif scenario == "Demand Surge":
        inv, alerts, _, _, tci_scenario = scenario_test(pred, test['date'], ROP, ROP, scenario='demand_surge')

    fig = plot_inventory(test['date'], inv, ROP, alerts)
    st.pyplot(fig)
    plt.close(fig)

    if tci_scenario is not None:
        st.markdown(f"**Scenario TCI:** ${tci_scenario:.2f}")

# ML: Volatility & Clustering
st.header("Product Segmentation (ML)")
with st.spinner("Analyzing product patterns..."):
    volatility = classify_volatility(sales_long)
    st.subheader("Product Volatility Classification")
    st.dataframe(volatility)
    
    item_stats = cluster_products(sales_long)
    st.subheader("Product Clusters")
    fig = plot_clusters(item_stats)
    st.pyplot(fig)
    plt.close(fig)

# Footer
st.markdown("---")
st.caption("Built with Python, Streamlit, and open-source ML/TS libraries. | Data is synthetic and for demonstration purposes only.")
