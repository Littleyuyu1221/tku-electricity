from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "app.py").is_file():
    ROOT = SCRIPT_DIR
    ARTIFACTS = SCRIPT_DIR
else:
    ROOT = SCRIPT_DIR.parent
    ARTIFACTS = ROOT / "outputs"
APP_PATH = ARTIFACTS / "app.py"
DATA_PATH = ARTIFACTS / "宿舍用電資料範本.csv"
WORKBOOK_PATH = ARTIFACTS / "宿舍用電與冷氣資料蒐集範本.xlsx"


def ridge_fit_predict(train_x, train_y, test_x, alpha=1.0):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale
    centered_y = train_y - train_y.mean()
    coef = np.linalg.solve(train_z.T @ train_z + alpha * np.eye(train_z.shape[1]), train_z.T @ centered_y)
    return test_z @ coef + train_y.mean()


def validate_energy_model():
    frame = pd.read_csv(DATA_PATH)
    assert "年份" in frame.columns
    assert not frame.duplicated(["年份", "月份"]).any()
    frame["人日"] = frame["住宿人數"] * frame["住宿天數"]
    frame["冷房度日"] = np.maximum(0, frame["日平均溫度_C"] - 23.0) * frame["開館天數"]
    frame = frame.loc[frame["人日"] > 0].copy()
    x = frame[["人日", "冷房度日", "開館天數"]].to_numpy(dtype=float)
    y = frame["總用電_kWh"].to_numpy(dtype=float)
    predictions = []
    for index in range(len(frame)):
        train_mask = np.arange(len(frame)) != index
        predictions.append(ridge_fit_predict(x[train_mask], y[train_mask], x[index : index + 1])[0])
    predictions = np.maximum(0, np.asarray(predictions))
    residual = y - predictions
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    mape = np.mean(np.abs(residual / y)) * 100
    cv_r2 = 1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)
    baseline_predictions = (y.sum() - y) / (len(y) - 1)
    baseline_mae = np.mean(np.abs(y - baseline_predictions))
    improvement = (baseline_mae - mae) / baseline_mae * 100
    assert np.isfinite([mae, rmse, mape, cv_r2]).all()
    assert improvement > 0
    app_text = APP_PATH.read_text(encoding="utf-8")
    assert 'df[["年份", "月份"]].duplicated().any()' in app_text
    assert 'df["期間"]' in app_text
    return {
        "samples": len(y),
        "cv_r2": cv_r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "baseline_mae": baseline_mae,
        "mae_improvement": improvement,
    }


def load_ac_function():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & {"AC_MODELS", "TIME_BLOCKS"}:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "estimate_ac_operation":
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"np": np, "pd": pd}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["estimate_ac_operation"]


def validate_ac_model():
    estimate = load_ac_function()
    base_args = dict(
        model_name="變頻一級能效",
        quantity=50,
        rated_power_kw=1.45,
        average_temp=25.0,
        setpoint=26.0,
        operating_days=30,
        usage_rates=[70, 20, 35, 85],
        age_years=6.0,
        maintenance_months=8,
        previous_faults=0,
    )
    operation, risk = estimate(**base_args)
    assert len(operation) == 4
    assert (operation[["每日耗電_kWh", "每月耗電_kWh", "運轉台時_月"]] >= 0).all().all()
    assert 0 <= risk["annual_probability"] <= 0.95

    hot_operation, _ = estimate(**{**base_args, "average_temp": 32.0})
    _, high_risk = estimate(
        **{**base_args, "age_years": 14.0, "maintenance_months": 24, "previous_faults": 2}
    )
    assert hot_operation["每月耗電_kWh"].sum() > operation["每月耗電_kWh"].sum()
    assert high_risk["annual_probability"] > risk["annual_probability"]

    workbook = pd.ExcelFile(WORKBOOK_PATH)
    expected_sheets = {
        "使用說明",
        "月用電資料",
        "冷氣設備清冊",
        "保養故障紀錄",
        "資料字典",
        "品質檢核",
        "選單設定",
    }
    assert expected_sheets.issubset(workbook.sheet_names)
    monthly_sheet = pd.read_excel(WORKBOOK_PATH, sheet_name="月用電資料", header=4).dropna(how="all")
    asset_sheet = pd.read_excel(WORKBOOK_PATH, sheet_name="冷氣設備清冊", header=4).dropna(how="all")
    assert len(monthly_sheet) == 12
    assert len(asset_sheet) == 3

    batch_results = []
    rate_columns = ["使用率_00_06", "使用率_06_12", "使用率_12_18", "使用率_18_24"]
    fractional_rates = [0.7, 0.2, 0.35, 0.85]
    normalized_rates = [value * 100 if 0 <= value <= 1 else value for value in fractional_rates]
    assert normalized_rates == [70, 20, 35, 85]
    for _, row in asset_sheet.iterrows():
        operation_group, risk_group = estimate(
            model_name=str(row["冷氣機型"]),
            quantity=int(row["台數"]),
            rated_power_kw=float(row["額定功率_kW"]),
            average_temp=float(row["日平均溫度_C"]),
            setpoint=float(row["設定溫度_C"]),
            operating_days=int(row["每月運轉天數"]),
            usage_rates=[float(row[column]) * 100 for column in rate_columns],
            age_years=float(row["平均機齡_年"]),
            maintenance_months=int(row["距上次保養_月"]),
            previous_faults=int(row["近兩年故障次數"]),
        )
        batch_results.append((operation_group["每月耗電_kWh"].sum(), risk_group["annual_probability"]))
    assert len(batch_results) == 3
    assert all(kwh > 0 and 0 <= probability <= 0.95 for kwh, probability in batch_results)
    return {
        "monthly_kwh": operation["每月耗電_kWh"].sum(),
        "annual_failure_probability": risk["annual_probability"],
        "hot_monthly_kwh": hot_operation["每月耗電_kWh"].sum(),
        "high_risk_probability": high_risk["annual_probability"],
        "batch_groups": len(batch_results),
        "batch_monthly_kwh": sum(item[0] for item in batch_results),
    }


if __name__ == "__main__":
    print("ENERGY", validate_energy_model())
    print("AC", validate_ac_model())
    print("MODEL_VALIDATION_OK")
