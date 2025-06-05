import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def train_forecast_sarimax(df, item_id, store_id, forecast_days=30):
    df_target = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].sort_values('date')
    train = df_target.iloc[:-forecast_days]
    test = df_target.iloc[-forecast_days:]
    model = SARIMAX(train['demand'], order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=forecast_days)
    pred = forecast.predicted_mean
    conf_int = forecast.conf_int()
    mae = mean_absolute_error(test['demand'], pred)
    rmse = np.sqrt(mean_squared_error(test['demand'], pred))
    return train, test, pred, conf_int, mae, rmse

def plot_forecast(train, test, pred, conf_int, item_id, store_id, model_name="Model"):
    plt.figure(figsize=(12,5))
    plt.plot(train['date'], train['demand'], label='Train')
    plt.plot(test['date'], test['demand'], label='Test', color='orange')
    plt.plot(test['date'], pred, label=f'Forecast ({model_name})', color='green')
    if conf_int is not None:
        plt.fill_between(test['date'], conf_int['lower'], conf_int['upper'], color='green', alpha=0.2)
    plt.legend()
    plt.title(f'Demand Forecasting for {item_id} at {store_id}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.show()

def naive_forecast(train, test):
    if len(train) == 0 or len(test) == 0:
        return np.full(len(test), np.nan)
    pred = np.repeat(train['demand'].iloc[-1], len(test))
    return pred

def moving_average_forecast(train, test, window=7):
    if len(train) < window or len(test) == 0:
        return np.full(len(test), np.nan)
    pred = np.repeat(train['demand'].iloc[-window:].mean(), len(test))
    return pred

def sarimax_forecast(train, test):
    # Require at least 14 points for SARIMAX (1 season)
    if len(train) < 14 or len(test) == 0:
        return np.full(len(test), np.nan), None
    try:
        # Create a proper time series index with daily frequency
        train_ts = train.set_index('date')['demand'].asfreq('D')
        test_ts = test.set_index('date')['demand'].asfreq('D')
        
        # Fit the model with the proper time series index
        model = SARIMAX(
            train_ts,
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False,
            freq='D'  # Explicitly set frequency to daily
        )
        results = model.fit(disp=False)
        
        # Generate forecast
        forecast = results.get_forecast(steps=len(test))
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Convert predictions back to numpy arrays
        pred = pred.values
        conf_int = pd.DataFrame({
            'lower': conf_int.iloc[:, 0].values,
            'upper': conf_int.iloc[:, 1].values
        })
        
        return pred, conf_int
    except Exception as e:
        print(f"SARIMAX forecast error: {str(e)}")
        return np.full(len(test), np.nan), None

def holt_winters_forecast(train, test):
    # Require at least 14 points for Holt-Winters (1 season)
    if len(train) < 14 or len(test) == 0:
        return np.full(len(test), np.nan)
    try:
        # Create a proper time series index
        train_ts = train.set_index('date')['demand'].asfreq('D')
        
        model = ExponentialSmoothing(
            train_ts,
            trend='add',
            seasonal='add',
            seasonal_periods=7,
            freq='D'
        )
        results = model.fit()
        pred = results.forecast(len(test))
        return pred.values
    except Exception as e:
        print(f"Holt-Winters forecast error: {str(e)}")
        return np.full(len(test), np.nan)

def prophet_forecast(train, test):
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet not installed.")
        return np.full(len(test), np.nan)
    
    # Check for at least 2 non-NaN rows
    if train['demand'].dropna().shape[0] < 2 or len(test) == 0:
        return np.full(len(test), np.nan)
    try:
        # Handle missing values using ffill and bfill
        df = pd.DataFrame({
            'ds': train['date'],
            'y': train['demand'].ffill().bfill()  # Updated to use ffill() and bfill()
        })
        
        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        m.fit(df)
        
        future = pd.DataFrame({'ds': test['date']})
        forecast = m.predict(future)
        return forecast['yhat'].values
    except Exception as e:
        print(f"Prophet forecast error: {str(e)}")
        return np.full(len(test), np.nan)

def xgboost_forecast(train, test):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("XGBoost not installed.")
        return np.full(len(test), np.nan)
    
    try:
        # Create copies of the dataframes to avoid SettingWithCopyWarning
        train_df = train.copy()
        test_df = test.copy()
        
        # Create features
        train_df.loc[:, 'day_of_week'] = train_df['date'].dt.dayofweek
        train_df.loc[:, 'month'] = train_df['date'].dt.month
        test_df.loc[:, 'day_of_week'] = test_df['date'].dt.dayofweek
        test_df.loc[:, 'month'] = test_df['date'].dt.month
        
        # Prepare features
        feature_cols = ['day_of_week', 'month']
        X_train = train_df[feature_cols]
        y_train = train_df['demand'].values
        X_test = test_df[feature_cols]
        
        # Train model
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        pred = model.predict(X_test)
        return pred
    except Exception as e:
        print(f"XGBoost forecast error: {str(e)}")
        return np.full(len(test), np.nan)

def evaluate_forecast(true, pred):
    if len(true) == 0 or len(pred) == 0:
        return (np.nan, np.nan)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return mae, rmse

def compare_models(train, test):
    results = {}
    
    # Create copies to avoid any potential data modification issues
    train_df = train.copy()
    test_df = test.copy()
    
    # Naive
    naive_pred = naive_forecast(train_df, test_df)
    results['Naive'] = evaluate_forecast(test_df['demand'], naive_pred)
    
    # Moving Average
    ma_pred = moving_average_forecast(train_df, test_df)
    results['MovingAverage'] = evaluate_forecast(test_df['demand'], ma_pred)
    
    # SARIMAX
    sarimax_pred, _ = sarimax_forecast(train_df, test_df)
    if not np.isnan(sarimax_pred).all():
        results['SARIMAX'] = evaluate_forecast(test_df['demand'], sarimax_pred)
    
    # Holt-Winters
    hw_pred = holt_winters_forecast(train_df, test_df)
    if not np.isnan(hw_pred).all():
        results['HoltWinters'] = evaluate_forecast(test_df['demand'], hw_pred)
    
    # Prophet
    prophet_pred = prophet_forecast(train_df, test_df)
    if not np.isnan(prophet_pred).all():
        results['Prophet'] = evaluate_forecast(test_df['demand'], prophet_pred)
    
    # XGBoost
    xgb_pred = xgboost_forecast(train_df, test_df)
    if not np.isnan(xgb_pred).all():
        results['XGBoost'] = evaluate_forecast(test_df['demand'], xgb_pred)
    
    return results

def get_best_model(results):
    return min(results.items(), key=lambda x: x[1][1])  # by RMSE

if __name__ == "__main__":
    df = pd.read_csv("sales_long.csv", parse_dates=['date'])
    item_id = 'item_1'
    store_id = 'store_1'
    df_target = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].sort_values('date')
    train = df_target.iloc[:-30]
    test = df_target.iloc[-30:]
    results = compare_models(train, test)
    for model, (mae, rmse) in results.items():
        print(f"{model}: MAE={mae:.2f}, RMSE={rmse:.2f}")
    best_model, (best_mae, best_rmse) = get_best_model(results)
    print(f"Best model: {best_model} (RMSE={best_rmse:.2f})")
