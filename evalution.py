"""
evaluation.py
Walk-forward validacija i poređenje naivnog modela sa ARIMA modelom.
"""
 
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
 
 
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
 
 
def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
 
 
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
 
 
def directional_accuracy(y_true_series, y_pred_series):
    """
    Procenat pogodjenog SMERA kretanja (gore/dole) - često informativnije
    za FX nego RMSE na sirovoj ceni.
    """
    true_direction = np.sign(np.diff(y_true_series))
    pred_direction = np.sign(np.array(y_pred_series[1:]) - np.array(y_true_series[:-1]))
    return np.mean(true_direction == pred_direction) * 100
 
 
def walk_forward_comparison(series: pd.Series, order: tuple, test_size: int = 250, refit_every: int = 5):
    """
    Poredi naivni model i ARIMA na poslednjih `test_size` dana,
    koristeći rolling window sa refitovanjem svakih `refit_every` koraka
    (refit svaki dan je sporo, pa refituj periodično radi brzine).
 
    Parametri
    ---------
    series : pd.Series
        Kompletna vremenska serija (npr. df["EURUSD"])
    order : tuple
        ARIMA red (p, d, q) - npr. najbolji red iz tvog grid search-a
    test_size : int
        Broj dana koje testiraš unazad
    refit_every : int
        Na koliko koraka ponovo treniraš ARIMA (radi brzine)
    """
    naive_preds, arima_preds, actuals = [], [], []
 
    n = len(series)
    start = n - test_size
    model_fit = None
 
    for i in range(start, n):
        history = series.iloc[:i]  # sve do trenutka i (bez data leakage-a)
        actual = series.iloc[i]
 
        # Naivni model
        naive_pred = history.iloc[-1]
 
        # ARIMA - refituj periodično radi brzine
        if (i - start) % refit_every == 0 or model_fit is None:
            model_fit = ARIMA(history, order=order).fit()
        else:
            # apenduj novi podatak bez potpunog refitovanja (brže)
            model_fit = model_fit.append([history.iloc[-1]], refit=False)
 
        arima_pred = model_fit.forecast(steps=1).iloc[0]
 
        naive_preds.append(naive_pred)
        arima_preds.append(arima_pred)
        actuals.append(actual)
 
    results = pd.DataFrame({
        "actual": actuals,
        "naive_pred": naive_preds,
        "arima_pred": arima_preds,
    }, index=series.index[start:n])
 
    print("=== Poređenje modela (walk-forward, poslednjih {} dana) ===".format(test_size))
    print(f"{'Model':<12}{'RMSE':>10}{'MAE':>10}{'MAPE':>10}{'Dir. Acc %':>12}")
    for name, col in [("Naive", "naive_pred"), ("ARIMA", "arima_pred")]:
        r = rmse(results["actual"], results[col])
        m = mae(results["actual"], results[col])
        mp = mape(results["actual"], results[col])
        da = directional_accuracy(results["actual"].values, results[col].values)
        print(f"{name:<12}{r:>10.5f}{m:>10.5f}{mp:>10.3f}{da:>12.2f}")
 
    return results
 
 
if __name__ == "__main__":
    # Primer korišćenja:
    # df = pd.read_csv("data/raw/eurusd_data_clean.csv", index_col="Date", parse_dates=True)
    # results = walk_forward_comparison(df["EURUSD"], order=(1, 1, 1), test_size=250)
    # results.to_csv("results/walk_forward_comparison.csv")
    pass
 