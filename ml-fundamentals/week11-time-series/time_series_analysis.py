"""
Phase 2 — Week 11: Time Series Analysis
=========================================
Covers: ARIMA/SARIMA, Prophet, LSTM for time series, walk-forward validation
Job relevance: 68% of AI/ML roles include time series in requirements
Common use cases: demand forecasting, anomaly detection, financial modeling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


# ── Data Generation (synthetic retail demand) ─────────────────────────────────
def generate_demand_data(
    n_days: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic retail demand with trend + seasonality + noise.
    Mimics real-world forecasting problems.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

    trend = np.linspace(100, 150, n_days)
    weekly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    yearly = 30 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = rng.normal(0, 8, n_days)

    demand = trend + weekly + yearly + noise
    demand = np.maximum(demand, 0)  # no negative demand

    return pd.DataFrame({"date": dates, "demand": demand}).set_index("date")


# ── Stationarity Checks ───────────────────────────────────────────────────────
def check_stationarity(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    ARIMA requires stationary data — this determines how many differences (d) to use.
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series.dropna())
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "n_lags": result[2],
        "n_obs": result[3],
        "is_stationary": result[1] < 0.05,
        "critical_values": {k: float(v) for k, v in result[4].items()},
    }


# ── ARIMA Forecasting ─────────────────────────────────────────────────────────
def arima_forecast(
    train: pd.Series,
    test: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
) -> dict:
    """
    SARIMA: Seasonal ARIMA — handles trend, seasonality, and autocorrelation.
    p = AR order, d = differencing, q = MA order
    P,D,Q,S = seasonal equivalents
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = float(np.mean(np.abs((test.values - forecast.values) / test.values)) * 100)

    return {
        "model": f"SARIMA{order}×{seasonal_order}",
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "forecast": forecast,
        "n_params": fitted.df_model,
    }


# ── Prophet Forecasting ───────────────────────────────────────────────────────
def prophet_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
) -> dict:
    """
    Facebook/Meta Prophet: handles trend, seasonality, holidays automatically.
    Best for business time series with strong seasonality patterns.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {"error": "prophet not installed. Run: pip install prophet"}

    # Prophet expects 'ds' (date) and 'y' (target) columns
    train_df = train.reset_index()
    train_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test), freq="D")
    forecast_df = model.predict(future)
    forecast = forecast_df.tail(len(test))["yhat"].values

    mae = mean_absolute_error(test["demand"].values, forecast)
    rmse = np.sqrt(mean_squared_error(test["demand"].values, forecast))
    mape = float(np.mean(np.abs((test["demand"].values - forecast) / test["demand"].values)) * 100)

    return {
        "model": "Facebook Prophet",
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "n_changepoints": len(model.changepoints),
        "forecast_values": forecast,
    }


# ── LSTM for Time Series ───────────────────────────────────────────────────────
def create_sequences(data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def lstm_forecast(
    train: pd.Series,
    test: pd.Series,
    seq_length: int = 30,
    hidden_size: int = 64,
    n_epochs: int = 30,
    batch_size: int = 32,
) -> dict:
    """
    LSTM for time series: captures long-range temporal dependencies.
    Better than ARIMA when patterns are nonlinear or very complex.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"error": "PyTorch not installed"}

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()

    full_scaled = np.concatenate([train_scaled, test_scaled])
    X_all, y_all = create_sequences(full_scaled, seq_length)

    n_train = len(train_scaled) - seq_length
    X_train = torch.FloatTensor(X_all[:n_train]).unsqueeze(-1)
    y_train = torch.FloatTensor(y_all[:n_train])
    X_test = torch.FloatTensor(X_all[n_train:]).unsqueeze(-1)
    y_test_scaled = y_all[n_train:]

    # LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze()

    model = LSTMModel(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    final_loss = 0.0
    for epoch in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        final_loss = loss.item()

    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test).numpy()

    # Inverse transform
    predictions = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()
    actuals = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)

    return {
        "model": "LSTM",
        "seq_length": seq_length,
        "hidden_size": hidden_size,
        "n_epochs": n_epochs,
        "final_train_loss": float(final_loss),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "n_test_samples": len(predictions),
    }


# ── Walk-Forward Validation ────────────────────────────────────────────────────
def walk_forward_validation(
    series: pd.Series,
    window_size: int = 365,
    forecast_horizon: int = 30,
    n_splits: int = 5,
) -> list[dict]:
    """
    Time-series cross-validation: train on past, test on future.
    Standard practice — regular k-fold is WRONG for time series (data leakage).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    results = []
    total_len = len(series)

    for i in range(n_splits):
        train_end = window_size + i * forecast_horizon
        test_start = train_end
        test_end = min(test_start + forecast_horizon, total_len)

        if test_end >= total_len:
            break

        train = series.iloc[:train_end]
        test = series.iloc[test_start:test_end]

        try:
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=len(test))
            mae = float(mean_absolute_error(test, forecast))
            mape = float(np.mean(np.abs((test.values - forecast.values) / test.values)) * 100)
            results.append({"split": i + 1, "train_size": len(train), "test_size": len(test), "mae": mae, "mape": mape})
        except Exception as e:
            results.append({"split": i + 1, "error": str(e)})

    return results


if __name__ == "__main__":
    print("=== Generating Synthetic Demand Data ===")
    df = generate_demand_data(n_days=730)
    print(f"Shape: {df.shape}")
    print(df.head())

    train_size = 600
    train = df["demand"].iloc[:train_size]
    test = df["demand"].iloc[train_size:]

    print("\n=== Stationarity Check ===")
    stat = check_stationarity(df["demand"])
    print(f"ADF statistic: {stat['adf_statistic']:.4f}")
    print(f"p-value: {stat['p_value']:.4f}")
    print(f"Is stationary: {stat['is_stationary']}")

    print("\n=== SARIMA Forecast ===")
    sarima = arima_forecast(train, test)
    print(f"Model: {sarima['model']}")
    print(f"MAE:  {sarima['mae']:.2f}")
    print(f"RMSE: {sarima['rmse']:.2f}")
    print(f"MAPE: {sarima['mape']:.2f}%")

    print("\n=== LSTM Forecast ===")
    lstm = lstm_forecast(train, test, n_epochs=20)
    if "error" not in lstm:
        print(f"Model: {lstm['model']}")
        print(f"MAE:  {lstm['mae']:.2f}")
        print(f"RMSE: {lstm['rmse']:.2f}")
        print(f"MAPE: {lstm['mape']:.2f}%")
    else:
        print(lstm["error"])

    print("\n=== Walk-Forward Validation ===")
    wf_results = walk_forward_validation(df["demand"], n_splits=4)
    for r in wf_results:
        if "error" not in r:
            print(f"  Split {r['split']}: train={r['train_size']}, MAPE={r['mape']:.2f}%")
