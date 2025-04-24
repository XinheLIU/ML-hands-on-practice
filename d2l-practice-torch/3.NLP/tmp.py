import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# 1. Download FTSE100 data
def download_ftse_data(start_date='2015-01-01', end_date='2023-12-31'):
    print("Downloading FTSE100 data...")
    ftse_data = yf.download('^FTSE', start=start_date, end=end_date)
    print(f"Downloaded {len(ftse_data)} days of FTSE100 data")
    return ftse_data

# 2. Prepare data for modeling
def prepare_data(data, test_size=0.2):
    # Use Adjusted Close for predictions
    df = data[['Adj Close']].copy()
    df.columns = ['price']
    
    # Create a date column for Prophet
    df_prophet = df.reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # Split into train and test
    train_size = int(len(df) * (1 - test_size))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Same for Prophet data
    train_prophet = df_prophet.iloc[:train_size]
    test_prophet = df_prophet.iloc[train_size:]
    
    return train, test, train_prophet, test_prophet, df

# 3. SARIMA model training and prediction
def run_sarima(train, test):
    print("Training SARIMA model...")
    # Simple SARIMA model (parameters can be optimized)
    model = SARIMAX(train['price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    model_fit = model.fit(disp=False)
    
    # Make predictions
    forecast_steps = len(test)
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_df = pd.DataFrame({'sarima_forecast': forecast}, index=test.index)
    
    print("SARIMA modeling completed")
    return forecast_df

# 4. Prophet model training and prediction
def run_prophet(train_prophet, test_prophet):
    print("Training Prophet model...")
    model = Prophet()
    model.fit(train_prophet)
    
    # Create dataframe for future predictions
    future = model.make_future_dataframe(periods=len(test_prophet), freq='D')
    forecast = model.predict(future)
    
    # Filter for test period and create forecast dataframe
    prophet_forecast = forecast.iloc[-len(test_prophet):][['ds', 'yhat']]
    prophet_forecast.set_index('ds', inplace=True)
    prophet_forecast.columns = ['prophet_forecast']
    
    print("Prophet modeling completed")
    return prophet_forecast

# 5. Calculate performance metrics
def calculate_metrics(actual, predictions):
    results = {}
    for model_name, preds in predictions.items():
        mse = mean_squared_error(actual, preds)
        mape = mean_absolute_percentage_error(actual, preds) * 100  # Convert to percentage
        
        results[model_name] = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAPE': mape
        }
    return results

# 6. Plot results
def plot_results(original_data, test_data, predictions):
    plt.figure(figsize=(14, 7))
    
    # Plot full historical data
    plt.plot(original_data.index, original_data['price'], label='Historical Data', color='black', alpha=0.5)
    
    # Plot test data
    plt.plot(test_data.index, test_data['price'], label='Actual (Test Set)', color='blue', linewidth=2)
    
    # Plot predictions
    colors = ['red', 'green']
    for i, (model_name, preds) in enumerate(predictions.items()):
        plt.plot(test_data.index, preds, label=f'{model_name} Forecast', color=colors[i], linestyle='--', linewidth=2)
    
    plt.title('FTSE100 Time Series Forecasting', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('FTSE100 Index Price', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ftse100_forecast.png')
    plt.show()

# Main execution function
def main():
    # Step 1: Download data
    ftse_data = download_ftse_data()
    
    # Step 2: Prepare data
    train, test, train_prophet, test_prophet, full_data = prepare_data(ftse_data)
    
    # Step 3 & 4: Run models
    sarima_predictions = run_sarima(train, test)
    prophet_predictions = run_prophet(train_prophet, test_prophet)
    
    # Combine predictions
    all_predictions = {
        'SARIMA': sarima_predictions['sarima_forecast'],
        'Prophet': prophet_predictions['prophet_forecast']
    }
    
    # Step 5: Calculate metrics
    metrics = calculate_metrics(test['price'], all_predictions)
    
    # Print results
    print("\n===== FTSE100 Forecast Backtest Results =====")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name} Model Performance:")
        print(f"MSE: {model_metrics['MSE']:.2f}")
        print(f"RMSE: {model_metrics['RMSE']:.2f}")
        print(f"MAPE: {model_metrics['MAPE']:.2f}%")
    
    # Step 6: Plot results
    plot_results(full_data, test, all_predictions)
    print("\nAnalysis complete! Check the generated plot.")

if __name__ == "__main__":
    main()
