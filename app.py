import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="宿舍用電分析與節能決策系統",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#0757A6"
ORANGE = "#D76800"
GREEN = "#18794E"
PURPLE = "#6D3FA0"
GRAY = "#65758B"
LIGHT_BLUE = "#D8E9FF"
COLORS = [BLUE, ORANGE, GREEN, PURPLE, GRAY]
CHART_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "responsive": True,
}

BASE_DIR = Path(__file__).resolve().parent
OFFICIAL_MODEL_PATH = BASE_DIR / "models" / "official_models.json"
OFFICIAL_MONTHLY_PATH = BASE_DIR / "data" / "songtao_model_monthly_base.csv"
OFFICIAL_ENERGY_EVALUATION_PATH = BASE_DIR / "data" / "energy_model_forward_validation.csv"


def load_official_models() -> dict:
    if not OFFICIAL_MODEL_PATH.exists():
        return {}
    return json.loads(OFFICIAL_MODEL_PATH.read_text(encoding="utf-8"))


OFFICIAL_MODELS = load_official_models()

AC_MODELS = {
    "窗型定頻（舊式）": {
        "rated_power_kw": 2.40,
        "partial_load_factor": 1.08,
        "base_failure_rate": 0.055,
        "description": "啟停頻繁、部分負載效率較低，適合作為老舊設備汰換比較基準。",
    },
    "分離式定頻": {
        "rated_power_kw": 1.90,
        "partial_load_factor": 0.96,
        "base_failure_rate": 0.042,
        "description": "一般定頻分離式冷氣，效率與故障基準介於窗型及變頻機種之間。",
    },
    "變頻一級能效": {
        "rated_power_kw": 1.45,
        "partial_load_factor": 0.74,
        "base_failure_rate": 0.026,
        "description": "可依負載調整輸出，部分負載時通常比定頻機種省電。",
    },
    "高效變頻機種": {
        "rated_power_kw": 1.20,
        "partial_load_factor": 0.64,
        "base_failure_rate": 0.020,
        "description": "以高效率變頻設備作為節能更新情境，實際功率仍應以銘牌為準。",
    },
}

TIME_BLOCKS = pd.DataFrame(
    {
        "時段": ["00–06", "06–12", "12–18", "18–24"],
        "時數": [6, 6, 6, 6],
        "室外溫度修正_C": [-1.5, 0.8, 3.2, 1.0],
        "預設使用率_%": [70, 20, 35, 85],
    }
)

AC_BATCH_SAMPLE = pd.DataFrame(
    [
        ["AC-A-01", "A棟寢室", "變頻一級能效", 1.45, 50, 6, 25, 26, 30, 70, 20, 35, 85, 8, 0],
        ["AC-B-01", "B棟公共區", "分離式定頻", 1.90, 12, 10, 25, 26, 30, 30, 55, 75, 65, 14, 1],
        ["AC-C-01", "C棟舊寢室", "窗型定頻（舊式）", 2.40, 20, 15, 25, 26, 30, 60, 15, 30, 80, 20, 2],
    ],
    columns=[
        "設備群組_ID",
        "位置",
        "冷氣機型",
        "額定功率_kW",
        "台數",
        "平均機齡_年",
        "日平均溫度_C",
        "設定溫度_C",
        "每月運轉天數",
        "使用率_00_06",
        "使用率_06_12",
        "使用率_12_18",
        "使用率_18_24",
        "距上次保養_月",
        "近兩年故障次數",
    ],
)
AC_BATCH_REQUIRED = list(AC_BATCH_SAMPLE.columns)


def standard_chart(fig: go.Figure, height: int = 410) -> go.Figure:
    """套用一致的圖表格式。"""
    fig.update_layout(
        height=height,
        font=dict(size=15),
        title=dict(font=dict(size=21), x=0.01),
        legend=dict(font=dict(size=14), orientation="h", y=1.12, x=0),
        hoverlabel=dict(font_size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=35, t=80, b=55),
    )
    fig.update_xaxes(
        title_font=dict(size=15),
        tickfont=dict(size=14),
        gridcolor="#D8DEE8",
        linecolor="#64748B",
        linewidth=1,
    )
    fig.update_yaxes(
        title_font=dict(size=15),
        tickfont=dict(size=14),
        gridcolor="#D8DEE8",
        linecolor="#64748B",
        linewidth=1,
    )
    return fig


def safe_mape(actual: pd.Series, predicted: np.ndarray) -> float:
    """避免實際值為 0 時產生無限大的 MAPE。"""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    valid = actual_array != 0
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs((actual_array[valid] - predicted_array[valid]) / actual_array[valid])) * 100)


def build_energy_model(model_df: pd.DataFrame):
    """建立 Ridge 模型，並以留一法交叉驗證避免只報訓練分數。"""
    feature_names = ["人日", "冷房度日", "開館天數"]
    features = model_df[feature_names]
    target = model_df["總用電_kWh"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    cv_predictions = cross_val_predict(model, features, target, cv=LeaveOneOut())
    cv_predictions = np.maximum(0, cv_predictions)
    model.fit(features, target)
    fitted_predictions = np.maximum(0, model.predict(features))
    metrics = {
        "MAE": mean_absolute_error(target, cv_predictions),
        "RMSE": np.sqrt(mean_squared_error(target, cv_predictions)),
        "MAPE": safe_mape(target, cv_predictions),
        "CV_R2": r2_score(target, cv_predictions),
        "TRAIN_R2": r2_score(target, fitted_predictions),
    }
    baseline_predictions = (target.sum() - target) / (len(target) - 1)
    metrics["BASELINE_MAE"] = mean_absolute_error(target, baseline_predictions)
    metrics["MAE_IMPROVEMENT"] = (
        (metrics["BASELINE_MAE"] - metrics["MAE"]) / metrics["BASELINE_MAE"] * 100
        if metrics["BASELINE_MAE"] > 0
        else 0.0
    )
    residuals = np.asarray(target) - cv_predictions
    uncertainty = 1.96 * np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0
    coefficients = pd.DataFrame(
        {
            "影響因子": ["住宿人日", "冷房度日", "開館天數"],
            "標準化係數": model.named_steps["ridge"].coef_,
        }
    )
    return model, cv_predictions, metrics, uncertainty, coefficients


def estimate_ac_operation(
    model_name: str,
    quantity: int,
    rated_power_kw: float,
    average_temp: float,
    setpoint: float,
    operating_days: int,
    usage_rates: list[int],
    age_years: float,
    maintenance_months: int,
    previous_faults: int,
) -> tuple[pd.DataFrame, dict]:
    """估算分時冷氣耗電與比例風險模型的年度故障機率。"""
    profile = AC_MODELS[model_name]
    operation = TIME_BLOCKS.copy()
    operation["使用率_%"] = usage_rates
    operation["估計室外溫度_C"] = average_temp + operation["室外溫度修正_C"]
    temperature_gap = np.maximum(0, operation["估計室外溫度_C"] - setpoint)
    operation["負載率"] = np.clip(0.22 + 0.105 * temperature_gap, 0.18, 1.0)
    age_efficiency_penalty = 1 + max(0.0, age_years - 5) * 0.018
    operation["每日耗電_kWh"] = (
        quantity
        * rated_power_kw
        * profile["partial_load_factor"]
        * age_efficiency_penalty
        * operation["負載率"]
        * operation["時數"]
        * operation["使用率_%"]
        / 100
    )
    operation["每月耗電_kWh"] = operation["每日耗電_kWh"] * operating_days
    operation["運轉台時_月"] = quantity * operation["時數"] * operation["使用率_%"] / 100 * operating_days

    per_unit_monthly_hours = float((operation["運轉台時_月"] / max(quantity, 1)).sum())
    weighted_load = float(
        np.average(operation["負載率"], weights=np.maximum(operation["運轉台時_月"], 0.001))
    )
    age_multiplier = np.exp(0.12 * max(0.0, age_years - 3))
    usage_multiplier = np.clip(0.55 + per_unit_monthly_hours * 12 / 2000, 0.55, 2.2)
    load_multiplier = 0.65 + 0.85 * weighted_load
    maintenance_multiplier = 1 + 0.045 * max(0, maintenance_months - 6)
    history_multiplier = 1 + 0.55 * previous_faults
    annual_hazard = (
        profile["base_failure_rate"]
        * age_multiplier
        * usage_multiplier
        * load_multiplier
        * maintenance_multiplier
        * history_multiplier
    )
    annual_probability = float(np.clip(1 - np.exp(-annual_hazard), 0, 0.95))
    monthly_probability = float(1 - (1 - annual_probability) ** (1 / 12))
    risk = {
        "annual_probability": annual_probability,
        "monthly_probability": monthly_probability,
        "expected_failures": annual_probability * quantity,
        "per_unit_monthly_hours": per_unit_monthly_hours,
        "weighted_load": weighted_load,
        "age_multiplier": age_multiplier,
        "usage_multiplier": usage_multiplier,
        "load_multiplier": load_multiplier,
        "maintenance_multiplier": maintenance_multiplier,
        "history_multiplier": history_multiplier,
    }
    return operation, risk


def classify_ac_risk(annual_failure_percent: float) -> str:
    if annual_failure_percent < 5:
        return "低"
    if annual_failure_percent < 15:
        return "中"
    if annual_failure_percent < 30:
        return "高"
    return "極高"


def official_energy_prediction(
    year: int,
    month: int,
    resident_count: int,
    resident_days: int,
    open_days: int,
    temperature_c: float,
) -> tuple[float, float, float, dict[str, float]]:
    model = OFFICIAL_MODELS["energy_model"]["model"]
    feature_values = {
        "estimated_resident_days": float(resident_count * resident_days),
        "cooling_degree_days_26c": max(float(temperature_c) - 26.0, 0.0) * float(open_days),
        "estimated_open_days": float(open_days),
        "time_index": float((year - 2021) * 12 + month),
    }
    means = np.asarray(model["feature_mean"], dtype=float)
    scales = np.asarray(model["feature_scale"], dtype=float)
    beta = np.asarray(model["residual_intercept_and_standardized_coefficients"], dtype=float)
    raw = np.asarray([feature_values[name] for name in model["feature_names"]], dtype=float)
    standardized = (raw - means) / scales
    residual_adjustment = float(beta[0] + standardized @ beta[1:])
    month_baselines = {int(key): float(value) for key, value in model["month_baseline_kwh"].items()}
    baseline = month_baselines.get(month, float(model["global_baseline_kwh"]))
    prediction = max(0.0, baseline + residual_adjustment)
    interval = model["prediction_interval"]
    lower = max(0.0, prediction + float(interval["lower_residual_quantile_5pct_kwh"]))
    upper = max(lower, prediction + float(interval["upper_residual_quantile_95pct_kwh"]))
    return prediction, lower, upper, feature_values


def official_ac_risk_prediction(
    building_code: str,
    brand_group: str,
    model_prefix: str,
    age_years: float,
    prior_fault_count: int,
    prior_service_count: int,
    years_since_last_service: float,
    fault_last_year: bool,
) -> tuple[float, str, pd.DataFrame]:
    model = OFFICIAL_MODELS["ac_failure_risk_model"]["model"]
    numeric = {
        "age_years": float(age_years),
        "prior_fault_count_log1p": float(np.log1p(prior_fault_count)),
        "prior_service_count_log1p": float(np.log1p(prior_service_count)),
        "years_since_last_service": float(years_since_last_service),
        "fault_last_year": float(bool(fault_last_year)),
    }
    means = np.asarray(model["numeric_mean"], dtype=float)
    scales = np.asarray(model["numeric_scale"], dtype=float)
    standardized = (
        np.asarray([numeric[name] for name in model["numeric_features"]], dtype=float) - means
    ) / scales
    encoded = list(standardized)
    category_values = {
        "building_code": building_code,
        "brand_group": brand_group,
        "model_prefix": model_prefix,
    }
    encoded_names = ["機齡", "累計故障", "累計維修／保養", "距上次維修／保養", "前一年故障"]
    for column in model["categorical_features"]:
        categories = model["categories"][column]
        selected = category_values[column]
        for category in categories[1:]:
            encoded.append(float(selected == category))
            encoded_names.append(f"{column}={category}")
    coefficients = np.asarray(model["coefficients"], dtype=float)
    linear = float(coefficients[0] + np.asarray(encoded) @ coefficients[1:])
    probability = float(1 / (1 + np.exp(-np.clip(linear, -35, 35))))
    contributions = np.asarray(encoded) * coefficients[1:]
    contribution_frame = pd.DataFrame(
        {"影響因子": encoded_names, "對風險分數的貢獻": contributions}
    ).sort_values("對風險分數的貢獻")
    if probability < 0.15:
        band = "相對低"
    elif probability < 0.28:
        band = "相對中"
    elif probability < 0.35:
        band = "相對高"
    else:
        band = "最高風險群"
    return probability, band, contribution_frame


def render_official_energy_model(electricity_price: float, carbon_factor: float) -> None:
    energy = OFFICIAL_MODELS["energy_model"]
    validation = energy["validation"]
    metrics = validation["model_metrics"]
    baseline = validation["seasonal_median_baseline_metrics"]
    st.header("⑤ 正式資料用電預測")
    st.info(
        "模型以三館合計月用電為目標，使用月份季節中位數，再由住宿人日、26°C 冷房度日、"
        "估算開館天數與時間趨勢修正。驗證採逐月向前切分，不讓模型看到未來資料。"
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("向前驗證 R²", f"{metrics['r2']:.3f}")
    m2.metric("MAE", f"{metrics['mae_kwh']:,.0f} 度")
    m3.metric("RMSE", f"{metrics['rmse_kwh']:,.0f} 度")
    m4.metric("sMAPE", f"{metrics['smape_percent']:.1f}%")
    m5.metric("比季節基準改善", f"{validation['mae_improvement_vs_baseline_percent']:.1f}%")
    st.caption(
        f"共 {energy['training_period']['months']} 個訓練月份；其中 {validation['test_months']} 個月份做樣本外驗證。"
        f"季節基準 MAE 為 {baseline['mae_kwh']:,.0f} 度。"
    )

    if OFFICIAL_ENERGY_EVALUATION_PATH.exists():
        evaluation = pd.read_csv(OFFICIAL_ENERGY_EVALUATION_PATH, encoding="utf-8-sig")
        comparison = go.Figure()
        comparison.add_trace(
            go.Scatter(
                x=evaluation["date"], y=evaluation["actual_kwh"], name="實際用電",
                mode="lines+markers", line=dict(color=BLUE, width=4)
            )
        )
        comparison.add_trace(
            go.Scatter(
                x=evaluation["date"], y=evaluation["predicted_kwh"], name="向前預測",
                mode="lines+markers", line=dict(color=ORANGE, width=3, dash="dash")
            )
        )
        comparison.update_layout(title="正式資料的逐月向前驗證", xaxis_title="月份", yaxis_title="用電量（度）")
        comparison.update_xaxes(tickangle=-45)
        st.plotly_chart(standard_chart(comparison), use_container_width=True, config=CHART_CONFIG)

    coefficients = OFFICIAL_MODELS["energy_model"]["model"]
    coefficient_frame = pd.DataFrame(
        {
            "影響因子": ["住宿人日", "26°C 冷房度日", "估算開館天數", "時間趨勢"],
            "標準化殘差修正係數": coefficients["residual_intercept_and_standardized_coefficients"][1:],
        }
    )
    coefficient_chart = px.bar(
        coefficient_frame.sort_values("標準化殘差修正係數"),
        x="標準化殘差修正係數", y="影響因子", orientation="h",
        color="標準化殘差修正係數",
        color_continuous_scale=[[0, ORANGE], [0.5, "#F3F4F6"], [1, BLUE]],
        color_continuous_midpoint=0,
        title="月份季節基準之外，各因子的修正方向",
    )
    coefficient_chart.update_coloraxes(showscale=False)
    st.plotly_chart(standard_chart(coefficient_chart, 330), use_container_width=True, config=CHART_CONFIG)

    st.subheader("設定要預測的月份")
    row1, row2, row3 = st.columns(3)
    forecast_year = row1.number_input("年份", min_value=2021, max_value=2030, value=2026, step=1)
    forecast_month = row2.selectbox("月份", [1, 2, 3, 4, 5, 6, 9, 10, 11, 12], index=6)
    forecast_temp = row3.number_input("淡水月平均溫度（°C）", min_value=5.0, max_value=40.0, value=28.5, step=0.5)
    row4, row5, row6 = st.columns(3)
    forecast_people = row4.number_input("住宿人數（人）", min_value=0, max_value=3000, value=2000, step=10)
    forecast_stay_days = row5.number_input("住宿天數（天）", min_value=0, max_value=31, value=25)
    forecast_open_days = row6.number_input("開館天數（天）", min_value=1, max_value=31, value=25)
    if forecast_stay_days > forecast_open_days:
        st.error("住宿天數不可大於開館天數。")
        return
    forecast, lower, upper, feature_values = official_energy_prediction(
        int(forecast_year), int(forecast_month), int(forecast_people),
        int(forecast_stay_days), int(forecast_open_days), float(forecast_temp)
    )
    f1, f2, f3 = st.columns(3)
    f1.metric("預測三館合計用電", f"{forecast:,.0f} 度")
    f2.metric("估計電費", f"{forecast * electricity_price:,.0f} 元")
    f3.metric("估計碳排", f"{forecast * carbon_factor / 1000:,.2f} 公噸")
    st.write(f"依向前驗證殘差形成的經驗 90% 範圍：約 **{lower:,.0f}–{upper:,.0f} 度**。")
    official_monthly = pd.read_csv(OFFICIAL_MONTHLY_PATH, encoding="utf-8-sig")
    ready = official_monthly[official_monthly["model_ready"] == True]  # noqa: E712
    ranges = {
        "estimated_resident_days": "住宿人日",
        "cooling_degree_days_26c": "冷房度日",
        "estimated_open_days": "開館天數",
    }
    outside = [
        label for column, label in ranges.items()
        if feature_values[column] < ready[column].min() or feature_values[column] > ready[column].max()
    ]
    if outside:
        st.warning("這個情境超出正式資料範圍（" + "、".join(outside) + "），屬外插預測，請降低解讀強度。")
    with st.expander("資料範圍與限制"):
        st.write("訓練期間為 2021-09 至 2026-06；114 年 9 月二館電表明示異常，另有統計低離群月份，均未納入嚴格模型。")
        st.write("住宿人數只有三館合計；開館天數缺少閉館日，因此住宿天數與開館天數為透明估算，不是刷卡或門禁實住紀錄。")
        st.write("熱水沒有獨立電表，模型預測的是總用電，不能拆解為冷氣、熱水與照明的實測比例。")


st.title("⚡ 淡江松濤館用電與冷氣風險系統")
st.write("整合校方近五年用電、住宿人數、開館日期、530 台現有冷氣與 2,442 筆修繕紀錄。")

DEMO_SAMPLE = pd.DataFrame(
    {
        "年份": [2024] * 12,
        "月份": range(1, 13),
        "總用電_kWh": [19521, 17634, 45606, 42192, 50989, 41913, 4682, 5681, 48200, 46000, 44000, 49000],
        "熱水用電_kWh": [6192, 5885, 15274, 9843, 11635, 7142, 1123, 1084, 12650, 11500, 11000, 12000],
        "住宿天數": [16, 12, 31, 30, 31, 23, 0, 0, 24, 31, 30, 31],
        "住宿人數": [653, 653, 653, 653, 653, 653, 0, 0, 653, 653, 653, 653],
        "開館天數": [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "日平均溫度_C": [15.2, 15.6, 17.5, 21.2, 24.1, 27.2, 28.9, 28.7, 27.1, 24.1, 20.6, 16.9],
    }
)
if OFFICIAL_MONTHLY_PATH.exists():
    official_monthly_source = pd.read_csv(OFFICIAL_MONTHLY_PATH, encoding="utf-8-sig")
    official_monthly_source = official_monthly_source[
        official_monthly_source["model_ready"] == True  # noqa: E712
    ].copy()
    SAMPLE = pd.DataFrame(
        {
            "年份": official_monthly_source["gregorian_year"],
            "月份": official_monthly_source["month"],
            "總用電_kWh": official_monthly_source["songtao_total_kwh"],
            # 校方未提供熱水獨立電表。保留欄位只為相容既有上傳格式，畫面會標示不可拆分。
            "熱水用電_kWh": 0.0,
            "住宿天數": official_monthly_source["estimated_open_days"],
            "住宿人數": official_monthly_source["songtao_resident_count"],
            "開館天數": official_monthly_source["estimated_open_days"],
            "日平均溫度_C": official_monthly_source["daily_mean_temperature_c"],
        }
    ).reset_index(drop=True)
    OFFICIAL_DEFAULT_AVAILABLE = True
else:
    SAMPLE = DEMO_SAMPLE.copy()
    OFFICIAL_DEFAULT_AVAILABLE = False
REQUIRED = list(SAMPLE.columns)


@st.cache_data
def read_upload(raw: bytes, name: str) -> pd.DataFrame:
    stream = io.BytesIO(raw)
    if not name.lower().endswith(".xlsx"):
        return pd.read_csv(stream)
    excel = pd.ExcelFile(stream)
    if "月用電資料" in excel.sheet_names:
        return pd.read_excel(excel, sheet_name="月用電資料", header=4)
    return pd.read_excel(excel)


@st.cache_data
def read_ac_upload(raw: bytes, name: str) -> pd.DataFrame:
    stream = io.BytesIO(raw)
    if not name.lower().endswith(".xlsx"):
        return pd.read_csv(stream)
    excel = pd.ExcelFile(stream)
    if "冷氣設備清冊" in excel.sheet_names:
        return pd.read_excel(excel, sheet_name="冷氣設備清冊", header=4)
    return pd.read_excel(excel)


with st.sidebar:
    st.header("⚙️ 簡單設定")
    uploaded = st.file_uploader("選擇用電資料", type=["csv", "xlsx"])
    st.caption("沒有上傳檔案時，會使用已匿名彙整的松濤館正式資料。" if OFFICIAL_DEFAULT_AVAILABLE else "沒有選擇檔案時，會使用示範資料。")
    st.download_button(
        "下載資料填寫範本",
        DEMO_SAMPLE.to_csv(index=False).encode("utf-8-sig"),
        "宿舍用電資料範本.csv",
        "text/csv",
        use_container_width=True,
    )
    collection_workbook_path = Path(__file__).with_name("宿舍用電與冷氣資料蒐集範本.xlsx")
    if collection_workbook_path.exists():
        st.download_button(
            "下載完整 Excel 蒐集工作簿",
            collection_workbook_path.read_bytes(),
            collection_workbook_path.name,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    default_capacity = max(700, int(SAMPLE["住宿人數"].max()) + 100)
    bed_capacity = st.number_input("宿舍總床位數", min_value=1, value=default_capacity, step=10)

    with st.expander("進階設定（一般不用更改）"):
        ac_base_temp = st.slider("冷氣啟用溫度（°C）", 20.0, 28.0, 23.0, 0.5)
        lighting_rate = st.slider("每人每日照明用電（kWh）", 0.1, 2.0, 1.15, 0.05)
        base_load = st.number_input("每月基礎用電（kWh）", min_value=0.0, value=4164.0, step=100.0)
        electricity_price = st.number_input("每度平均電價（元）", min_value=0.0, value=4.0, step=0.1)
        carbon_factor = st.number_input(
            "每度電碳排（kg CO₂e）",
            min_value=0.0,
            value=0.466,
            step=0.001,
            help="預設採經濟部能源署公布的 2025 年產業用電排碳係數；正式盤查請依資料年度與學校電價類別更新。",
        )

try:
    source = read_upload(uploaded.getvalue(), uploaded.name) if uploaded else SAMPLE.copy()
except Exception as exc:
    st.error(f"檔案無法讀取：{exc}")
    st.stop()

using_official_default = uploaded is None and OFFICIAL_DEFAULT_AVAILABLE
hotwater_submeter_available = not using_official_default
if using_official_default:
    st.success("目前顯示：松濤一至三館正式彙整資料。房號、財產編號與人員欄位未放入公開網站。")
    st.caption("模型資料期間：2021-09 至 2026-06；官方氣溫來源為中央氣象署淡水站（466900）。")

if "平均溫度_C" in source.columns and "日平均溫度_C" not in source.columns:
    source = source.rename(columns={"平均溫度_C": "日平均溫度_C"})
if "年份" not in source.columns:
    source.insert(0, "年份", 2024)
    st.info("上傳資料沒有「年份」欄，系統暫以 2024 年處理；跨年度分析請使用新版範本。")

missing = [column for column in REQUIRED if column not in source.columns]
if missing:
    st.error("資料缺少這些欄位：" + "、".join(missing))
    st.info("請下載左側的資料範本，依照欄位填寫後再上傳。")
    st.stop()

df = source[REQUIRED].copy()
for column in REQUIRED:
    df[column] = pd.to_numeric(df[column], errors="coerce")

if df.empty:
    st.error("資料檔案沒有內容。")
    st.stop()
if df.isna().any().any():
    st.error("資料中有空白或非數字內容，請修正後重新上傳。")
    st.stop()
if not df["年份"].between(2000, 2100).all():
    st.error("年份必須是 2000 到 2100 之間的西元年。")
    st.stop()
if not np.allclose(df[["年份", "月份"]], np.round(df[["年份", "月份"]])):
    st.error("年份與月份必須是整數。")
    st.stop()
if not df["月份"].between(1, 12).all() or df[["年份", "月份"]].duplicated().any():
    st.error("月份必須介於 1 到 12，而且同一個年份與月份不可重複。")
    st.stop()
if (df.drop(columns=["日平均溫度_C"]) < 0).any().any():
    st.error("除溫度外，其餘欄位不可小於 0。")
    st.stop()
if (df["住宿天數"] > df["開館天數"]).any():
    st.error("住宿天數不可大於開館天數。")
    st.stop()
if df["開館天數"].sum() <= 0:
    st.error("資料中至少要有 1 天開館日。")
    st.stop()
if (df["熱水用電_kWh"] > df["總用電_kWh"]).any():
    st.error("熱水用電不可大於同月的總用電。")
    st.stop()
if (df["住宿人數"] > bed_capacity).any():
    st.warning("有月份的住宿人數大於總床位數，請確認資料或左側床位設定。")

df = df.sort_values(["年份", "月份"]).reset_index(drop=True)
df["期間"] = df["年份"].astype(int).astype(str) + "-" + df["月份"].astype(int).astype(str).str.zfill(2)
df["人日"] = df["住宿天數"] * df["住宿人數"]
df["住房率_%"] = np.where(df["住宿天數"] > 0, df["住宿人數"] / bed_capacity * 100, 0)
df["開館使用率_%"] = np.where(df["開館天數"] > 0, df["住宿天數"] / df["開館天數"] * 100, 0)
df["冷房度日"] = np.maximum(0, df["日平均溫度_C"] - ac_base_temp) * df["開館天數"]
df["照明估算原始值"] = df["人日"] * lighting_rate
df["冷氣估算原始值"] = np.maximum(0, df["日平均溫度_C"] - ac_base_temp) * df["人日"] * 0.29

# 確保所有分項加總等於實際總用電。熱水視為量測值，照明與冷氣是估算值。
df["基礎用電"] = np.minimum(base_load, (df["總用電_kWh"] - df["熱水用電_kWh"]).clip(lower=0))
available_for_estimates = (df["總用電_kWh"] - df["熱水用電_kWh"] - df["基礎用電"]).clip(lower=0)
raw_estimate_total = df["照明估算原始值"] + df["冷氣估算原始值"]
estimate_scale = pd.Series(0.0, index=df.index)
has_estimate = raw_estimate_total > 0
estimate_scale.loc[has_estimate] = np.minimum(
    1,
    available_for_estimates.loc[has_estimate] / raw_estimate_total.loc[has_estimate],
)
df["照明用電"] = df["照明估算原始值"] * estimate_scale
df["冷氣用電"] = df["冷氣估算原始值"] * estimate_scale
df["其他用電"] = (
    available_for_estimates - df["照明用電"] - df["冷氣用電"]
).clip(lower=0)
df["每人日用電"] = np.where(df["人日"] > 0, df["總用電_kWh"] / df["人日"], np.nan)
df["每開館日用電"] = np.where(df["開館天數"] > 0, df["總用電_kWh"] / df["開館天數"], np.nan)

total = df["總用電_kWh"].sum()
if total <= 0:
    st.error("總用電必須大於 0，才能進行分析。")
    st.stop()

peak = df.loc[df["總用電_kWh"].idxmax()]
lowest = df.loc[df["總用電_kWh"].idxmin()]
occupied = df.loc[df["人日"] > 0].copy()
unit_kwh = occupied["總用電_kWh"].sum() / occupied["人日"].sum() if not occupied.empty else 0
total_stay_days = int(df["住宿天數"].sum())
total_open_days = int(df["開館天數"].sum())
weighted_temp = np.average(df["日平均溫度_C"], weights=df["開館天數"])
average_monthly = total / len(df)
occupied_days = occupied["住宿天數"].sum()
average_occupancy = occupied["人日"].sum() / (bed_capacity * occupied_days) * 100 if occupied_days else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("資料期間總用電", f"{total:,.0f} 度")
c2.metric("估計總電費", f"{total * electricity_price:,.0f} 元")
c3.metric("最高用電期間", peak["期間"])
c4.metric("每人每天平均用電", f"{unit_kwh:.2f} 度")

page = st.selectbox(
    "👇 請選擇想看的內容",
    [
        "① 快速看懂用電",
        "② 住宿人數與開館",
        "③ 電都用到哪裡",
        "④ 可以省多少電",
        "⑤ 下個月用電預測",
        "⑥ 冷氣機型與故障風險",
    ],
)

if page == "① 快速看懂用電":
    st.header("① 快速看懂用電")
    st.info(
        f"重點：{peak['期間']} 用電最多（{peak['總用電_kWh']:,.0f} 度）；"
        f"{lowest['期間']} 最少（{lowest['總用電_kWh']:,.0f} 度）。"
    )

    trend = go.Figure()
    trend.add_trace(
        go.Scatter(
            x=df["期間"],
            y=df["總用電_kWh"],
            name="每月用電",
            mode="lines+markers+text",
            text=[f"{value / 1000:.0f}千" for value in df["總用電_kWh"]],
            textposition="top center",
            line=dict(color=BLUE, width=5),
            marker=dict(size=12),
            hovertemplate="%{x}：%{y:,.0f} 度<extra></extra>",
        )
    )
    trend.add_hline(
        y=average_monthly,
        line_dash="dash",
        line_color=ORANGE,
        line_width=3,
        annotation_text=f"月平均 {average_monthly:,.0f} 度",
        annotation_font_size=17,
    )
    trend.update_layout(title="每個期間用了多少電？", xaxis_title="年月", yaxis_title="用電量（度）", showlegend=False)
    trend.update_xaxes(tickangle=-45)
    st.plotly_chart(standard_chart(trend), use_container_width=True, config=CHART_CONFIG)

    ranking = df.sort_values("總用電_kWh", ascending=True)
    rank_fig = px.bar(
        ranking,
        x="總用電_kWh",
        y="期間",
        orientation="h",
        text="總用電_kWh",
        color="總用電_kWh",
        color_continuous_scale=[[0, LIGHT_BLUE], [1, BLUE]],
        title="年月用電量排名",
        labels={"總用電_kWh": "用電量（度）", "期間": "年月"},
    )
    rank_fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    rank_fig.update_coloraxes(showscale=False)
    st.plotly_chart(standard_chart(rank_fig, 500), use_container_width=True, config=CHART_CONFIG)

elif page == "② 住宿人數與開館":
    st.header("② 住宿人數與開館")
    st.info(
        f"資料期間共住宿 {total_stay_days} 天、開館 {total_open_days} 天；"
        f"有住宿月份的平均住房率約 {average_occupancy:.1f}%。"
    )

    operation_fig = go.Figure()
    operation_fig.add_trace(go.Bar(x=df["期間"], y=df["住宿天數"], name="住宿天數", marker_color=BLUE))
    operation_fig.add_trace(go.Bar(x=df["期間"], y=df["開館天數"], name="開館天數", marker_color=ORANGE))
    operation_fig.add_trace(
        go.Scatter(
            x=df["期間"],
            y=df["日平均溫度_C"],
            name="日平均溫度",
            mode="lines+markers",
            line=dict(color=GREEN, width=5),
            marker=dict(size=10),
            yaxis="y2",
        )
    )
    operation_fig.update_layout(
        title="住宿天數、開館天數與溫度",
        barmode="group",
        xaxis=dict(title="年月", tickangle=-45),
        yaxis=dict(title="天數"),
        yaxis2=dict(title="溫度（°C）", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(standard_chart(operation_fig), use_container_width=True, config=CHART_CONFIG)

    left, right = st.columns([3, 2])
    with left:
        bubble = px.scatter(
            df,
            x="住宿人數",
            y="總用電_kWh",
            size="開館天數",
            color="日平均溫度_C",
            text=df["期間"],
            color_continuous_scale=[[0, BLUE], [0.55, ORANGE], [1, "#B42318"]],
            title="人數、溫度與用電的關係",
            labels={"住宿人數": "住宿人數（人）", "總用電_kWh": "用電量（度）", "日平均溫度_C": "溫度（°C）"},
        )
        bubble.update_traces(marker=dict(line=dict(width=2, color="#FFFFFF")))
        st.plotly_chart(standard_chart(bubble), use_container_width=True, config=CHART_CONFIG)
    with right:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=average_occupancy,
                number={"suffix": "%", "font": {"size": 42, "color": "#0B356B"}},
                title={"text": "平均住房率", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 16}},
                    "bar": {"color": BLUE, "thickness": 0.35},
                    "steps": [
                        {"range": [0, 60], "color": "#E5E7EB"},
                        {"range": [60, 85], "color": "#FDE7C7"},
                        {"range": [85, 100], "color": "#D7F0E2"},
                    ],
                },
            )
        )
        st.plotly_chart(standard_chart(gauge), use_container_width=True, config=CHART_CONFIG)
        st.caption(f"開館天數加權的平均溫度：{weighted_temp:.1f} °C")

elif page == "③ 電都用到哪裡":
    st.header("③ 電都用到哪裡")
    if not hotwater_submeter_available:
        st.warning(
            "校方目前提供的是總用電，沒有熱水、冷氣與照明的獨立電表；本頁除總用電外皆為工程估算，"
            "『其他』也包含尚未分表的熱水負載，不能當成實測占比。"
        )
    long_df = df.melt(
        id_vars=["年份", "月份", "期間"],
        value_vars=["基礎用電", "照明用電", "熱水用電_kWh", "冷氣用電", "其他用電"],
        var_name="用電類別",
        value_name="用電量_kWh",
    )
    category_names = {
        "基礎用電": "基礎設備",
        "照明用電": "照明",
        "熱水用電_kWh": "熱水",
        "冷氣用電": "冷氣",
        "其他用電": "其他",
    }
    long_df["用電類別"] = long_df["用電類別"].replace(category_names)
    sums = long_df.groupby("用電類別", as_index=False)["用電量_kWh"].sum().sort_values("用電量_kWh", ascending=False)
    main_category = sums.iloc[0]
    st.info(f"重點：目前估算占比最高的是「{main_category['用電類別']}」，資料期間約 {main_category['用電量_kWh']:,.0f} 度。")

    area = px.area(
        long_df,
        x="期間",
        y="用電量_kWh",
        color="用電類別",
        color_discrete_sequence=COLORS,
        title="每月各類用電的變化",
        labels={"用電量_kWh": "用電量（度）"},
    )
    area.update_xaxes(tickangle=-45)
    st.plotly_chart(standard_chart(area), use_container_width=True, config=CHART_CONFIG)

    left, right = st.columns(2)
    with left:
        pie = px.pie(
            sums,
            values="用電量_kWh",
            names="用電類別",
            hole=0.42,
            color_discrete_sequence=COLORS,
            title="資料期間用電占比",
        )
        pie.update_traces(textposition="inside", textinfo="label+percent", textfont_size=17)
        st.plotly_chart(standard_chart(pie), use_container_width=True, config=CHART_CONFIG)
    with right:
        pivot = long_df.pivot(index="用電類別", columns="期間", values="用電量_kWh").reindex(sums["用電類別"])
        heatmap = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=pivot.index,
                colorscale=[[0, "#F2F7FF"], [1, BLUE]],
                colorbar=dict(title="度", tickfont=dict(size=15)),
                hovertemplate="%{y}<br>%{x}<br>%{z:,.0f} 度<extra></extra>",
            )
        )
        heatmap.update_layout(title="深色表示用電較多")
        st.plotly_chart(standard_chart(heatmap), use_container_width=True, config=CHART_CONFIG)

elif page == "④ 可以省多少電":
    st.header("④ 可以省多少電")
    st.write("請拖動下面三個設定，圖表和金額會立即更新。")
    temp_raise = st.slider("冷氣溫度提高幾度？", 0.0, 3.0, 1.0, 0.5)
    lighting_saving = st.slider("照明預計節省百分比", 0, 50, 20, 5) / 100
    if hotwater_submeter_available:
        hotwater_saving = st.slider("熱水預計節省百分比", 0, 40, 10, 5) / 100
    else:
        hotwater_saving = 0.0
        st.info("未提供熱水獨立電表，因此正式資料模式不估算熱水節電量。")

    new_ac_raw = np.maximum(0, df["日平均溫度_C"] - (ac_base_temp + temp_raise)) * df["人日"] * 0.29
    ac_saving_ratio = pd.Series(0.0, index=df.index)
    has_ac = df["冷氣估算原始值"] > 0
    ac_saving_ratio.loc[has_ac] = 1 - np.minimum(
        1,
        new_ac_raw.loc[has_ac] / df.loc[has_ac, "冷氣估算原始值"],
    )
    ac_saving = (df["冷氣用電"] * ac_saving_ratio).sum()
    lighting_saved = df["照明用電"].sum() * lighting_saving
    hotwater_saved = df["熱水用電_kWh"].sum() * hotwater_saving
    saving = ac_saving + lighting_saved + hotwater_saved
    saving_rate = saving / total * 100

    s1, s2, s3 = st.columns(3)
    s1.metric("資料期間可以省電", f"{saving:,.0f} 度", f"約 {saving_rate:.1f}%")
    s2.metric("資料期間可以省錢", f"{saving * electricity_price:,.0f} 元")
    s3.metric("資料期間可以減碳", f"{saving * carbon_factor / 1000:,.2f} 公噸")

    before_after = pd.DataFrame({"情況": ["改善前", "改善後"], "用電量": [total, total - saving]})
    compare = px.bar(
        before_after,
        x="情況",
        y="用電量",
        color="情況",
        text="用電量",
        color_discrete_map={"改善前": GRAY, "改善後": GREEN},
        title="改善前後用電比較",
        labels={"用電量": "資料期間用電（度）"},
    )
    compare.update_traces(texttemplate="%{text:,.0f} 度", textposition="outside")

    contributions = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["原本用電", "冷氣節省", "照明節省", "熱水節省", "改善後"],
            y=[total, -ac_saving, -lighting_saved, -hotwater_saved, total - saving],
            text=[
                f"{total:,.0f}",
                f"省 {ac_saving:,.0f}",
                f"省 {lighting_saved:,.0f}",
                f"省 {hotwater_saved:,.0f}",
                f"{total-saving:,.0f}",
            ],
            textposition="outside",
            connector={"line": {"color": GRAY, "width": 2}},
            decreasing={"marker": {"color": GREEN}},
            increasing={"marker": {"color": ORANGE}},
            totals={"marker": {"color": BLUE}},
        )
    )
    contributions.update_layout(title="每個節能方法各省多少電？", yaxis_title="用電量（度）", showlegend=False)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(standard_chart(compare), use_container_width=True, config=CHART_CONFIG)
    with right:
        st.plotly_chart(standard_chart(contributions), use_container_width=True, config=CHART_CONFIG)

elif page == "⑤ 下個月用電預測":
    if using_official_default and OFFICIAL_MODELS.get("energy_model"):
        render_official_energy_model(electricity_price, carbon_factor)
        st.stop()
    st.header("⑤ 下個月用電預測")
    st.info("以住宿人日、冷房度日及開館天數建立 Ridge 迴歸，並用留一法交叉驗證模型。")

    if len(occupied) < 6:
        st.warning("有住宿的月份少於 6 筆，樣本不足以進行交叉驗證。建議補充至少 12–24 個月資料。")
    else:
        model_df = occupied.copy()
        model, cv_predictions, metrics, uncertainty, coefficients = build_energy_model(model_df)
        model_df["交叉驗證預測_kWh"] = cv_predictions

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("交叉驗證 R²", f"{metrics['CV_R2']:.3f}")
        m2.metric("平均絕對誤差 MAE", f"{metrics['MAE']:,.0f} 度")
        m3.metric("均方根誤差 RMSE", f"{metrics['RMSE']:,.0f} 度")
        m4.metric("平均百分比誤差", f"{metrics['MAPE']:.1f}%")
        m5.metric("比月平均基準改善", f"{metrics['MAE_IMPROVEMENT']:.1f}%")
        st.caption(
            "以上為留一法（每次保留 1 筆作測試）的樣本外評估；最後一項比較 Ridge 與只用其餘月份平均值的基準模型。"
        )
        if metrics["CV_R2"] < 0:
            st.warning("目前交叉驗證 R² 小於 0，代表示範資料不足以形成穩定預測；此結果本身應列入專題限制。")
        elif metrics["CV_R2"] < 0.5:
            st.warning("模型目前只有有限的樣本外解釋力，建議增加跨年度資料及設備運轉紀錄。")
        else:
            st.success("模型在現有資料上的樣本外解釋力尚可，但仍需用新月份資料持續驗證。")

        predicted_chart = go.Figure()
        predicted_chart.add_trace(
            go.Scatter(
                x=model_df["期間"],
                y=model_df["總用電_kWh"],
                name="實際用電",
                mode="lines+markers",
                line=dict(color=BLUE, width=5),
                marker=dict(size=11),
            )
        )
        predicted_chart.add_trace(
            go.Scatter(
                x=model_df["期間"],
                y=model_df["交叉驗證預測_kWh"],
                name="留一法預測",
                mode="lines+markers",
                line=dict(color=ORANGE, width=4, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
            )
        )
        predicted_chart.update_layout(title="過去實際用電和模型估計", xaxis_title="年月", yaxis_title="用電量（度）")
        predicted_chart.update_xaxes(tickangle=-45)
        st.plotly_chart(standard_chart(predicted_chart), use_container_width=True, config=CHART_CONFIG)

        coefficient_fig = px.bar(
            coefficients.sort_values("標準化係數"),
            x="標準化係數",
            y="影響因子",
            orientation="h",
            color="標準化係數",
            color_continuous_scale=[[0, ORANGE], [0.5, "#F3F4F6"], [1, BLUE]],
            color_continuous_midpoint=0,
            title="模型如何判斷：正值提高用電，負值降低用電",
        )
        coefficient_fig.update_coloraxes(showscale=False)
        st.plotly_chart(standard_chart(coefficient_fig, 330), use_container_width=True, config=CHART_CONFIG)

        st.subheader("填寫下個月情況")
        row1, row2 = st.columns(2)
        forecast_temp = row1.number_input("日平均溫度（°C）", value=24.0, step=0.5)
        forecast_people = row2.number_input("住宿人數（人）", min_value=0, value=min(653, bed_capacity), step=10)
        row3, row4 = st.columns(2)
        forecast_stay_days = row3.number_input("住宿天數（天）", min_value=0, max_value=31, value=30)
        forecast_open_days = row4.number_input("開館天數（天）", min_value=1, max_value=31, value=30)

        if forecast_stay_days > forecast_open_days:
            st.error("住宿天數不可大於開館天數。")
        else:
            forecast_person_days = forecast_people * forecast_stay_days
            forecast_cooling_degree_days = max(0, forecast_temp - ac_base_temp) * forecast_open_days
            forecast_input = pd.DataFrame(
                [[forecast_person_days, forecast_cooling_degree_days, forecast_open_days]],
                columns=["人日", "冷房度日", "開館天數"],
            )
            forecast = max(0, model.predict(forecast_input)[0])
            lower_bound = max(0, forecast - uncertainty)
            upper_bound = forecast + uncertainty
            f1, f2, f3 = st.columns(3)
            f1.metric("下個月可能用電", f"{forecast:,.0f} 度")
            f2.metric("下個月可能電費", f"{forecast * electricity_price:,.0f} 元")
            f3.metric("下個月可能碳排", f"{forecast * carbon_factor / 1000:,.2f} 公噸")
            st.write(f"依交叉驗證殘差估計，約 95% 參考範圍為 **{lower_bound:,.0f}–{upper_bound:,.0f} 度**。")

            outside_ranges = []
            for feature in forecast_input.columns:
                value = forecast_input.iloc[0][feature]
                if value < model_df[feature].min() or value > model_df[feature].max():
                    outside_ranges.append(feature)
            if outside_ranges:
                st.warning("輸入情境超出既有資料範圍（" + "、".join(outside_ranges) + "），屬於外插預測，可信度較低。")

        with st.expander("模型說明（需要時再看）"):
            st.write(
                "模型輸入為住宿人日、冷房度日與開館天數；Ridge 迴歸可降低少量資料中因變數高度相關造成的係數不穩定。"
                f"訓練資料 R² 為 {metrics['TRAIN_R2']:.3f}，留一法交叉驗證 R² 為 {metrics['CV_R2']:.3f}。"
                "正式報告應以交叉驗證結果為主；資料少於 24 個月時，結果適合做情境比較，不應宣稱為精準電費預報。"
            )

    with st.expander("查看完整資料表"):
        shown_columns = REQUIRED + [
            "人日",
            "住房率_%",
            "開館使用率_%",
            "每人日用電",
            "每開館日用電",
        ]
        shown = df[shown_columns].round(2)
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button(
            "下載分析結果",
            shown.to_csv(index=False).encode("utf-8-sig"),
            "宿舍用電分析結果.csv",
            "text/csv",
            use_container_width=True,
        )

elif page == "⑥ 冷氣機型與故障風險":
    st.header("⑥ 冷氣機型、分時耗電與故障風險")
    official_risk_active = bool(OFFICIAL_MODELS.get("ac_failure_risk_model"))
    st.info(
        "分時耗電採額定功率與負載率工程估算；年度故障風險則使用校方現有冷氣與修繕紀錄訓練的模型。"
        if official_risk_active
        else "選擇冷氣機型與使用條件，系統會估算四個時段的耗電，並以比例風險模型評估年度故障機率。"
    )

    left, right = st.columns(2)
    with left:
        ac_model_name = st.selectbox("冷氣機型／效率類別", list(AC_MODELS))
        ac_profile = AC_MODELS[ac_model_name]
        st.caption(ac_profile["description"])
        ac_quantity = st.number_input("冷氣台數", min_value=1, max_value=2000, value=50, step=1)
        rated_power = st.number_input(
            "單台額定消耗功率（kW）",
            min_value=0.2,
            max_value=10.0,
            value=float(ac_profile["rated_power_kw"]),
            step=0.05,
            key=f"rated_power_{ac_model_name}",
        )
        ac_age = st.number_input("平均機齡（年）", min_value=0.0, max_value=30.0, value=6.0, step=0.5)
    with right:
        scenario_temp = st.number_input("情境日平均溫度（°C）", min_value=5.0, max_value=40.0, value=float(round(weighted_temp, 1)), step=0.5)
        ac_setpoint = st.number_input("冷氣設定溫度（°C）", min_value=18.0, max_value=30.0, value=26.0, step=0.5)
        ac_operating_days = st.number_input("每月運轉天數", min_value=1, max_value=31, value=30, step=1)
        maintenance_months = st.number_input("距上次保養（月）", min_value=0, max_value=60, value=8, step=1)
        previous_faults = st.number_input(
            "累計故障次數" if official_risk_active else "近兩年單台平均故障次數",
            min_value=0, max_value=30, value=0, step=1,
        )

    if official_risk_active:
        st.subheader("正式修繕紀錄風險條件")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        building_name = risk_col1.selectbox("館別", ["松濤一館", "松濤二館", "松濤三館"])
        building_code = {"松濤一館": "ZA", "松濤二館": "ZB", "松濤三館": "ZC"}[building_name]
        brand_group = risk_col2.selectbox("設備廠牌", ["大金", "日立", "萬士益"])
        model_prefix = risk_col3.selectbox("清冊型號族群", ["FTX", "RAS", "RA/RAC", "其他"])
        risk_col4, risk_col5 = st.columns(2)
        prior_service_count = risk_col4.number_input(
            "累計維修／保養事件數", min_value=int(previous_faults), max_value=100,
            value=max(int(previous_faults), 2), step=1,
        )
        fault_last_year = risk_col5.checkbox("前一年曾有故障", value=bool(previous_faults))
    else:
        building_code, brand_group, model_prefix = "ZA", "大金", "FTX"
        prior_service_count = int(previous_faults)
        fault_last_year = bool(previous_faults)

    st.subheader("各時段預估使用率")
    usage_columns = st.columns(4)
    usage_rates = []
    for index, row in TIME_BLOCKS.iterrows():
        usage_rates.append(
            usage_columns[index].slider(
                str(row["時段"]),
                min_value=0,
                max_value=100,
                value=int(row["預設使用率_%"]),
                step=5,
                key=f"usage_{index}",
                help="此時段平均有多少比例的冷氣處於運轉狀態。",
            )
        )

    operation, risk = estimate_ac_operation(
        model_name=ac_model_name,
        quantity=int(ac_quantity),
        rated_power_kw=float(rated_power),
        average_temp=float(scenario_temp),
        setpoint=float(ac_setpoint),
        operating_days=int(ac_operating_days),
        usage_rates=usage_rates,
        age_years=float(ac_age),
        maintenance_months=int(maintenance_months),
        previous_faults=int(previous_faults),
    )

    official_contributions = None
    if official_risk_active:
        official_probability, official_band, official_contributions = official_ac_risk_prediction(
            building_code=building_code,
            brand_group=brand_group,
            model_prefix=model_prefix,
            age_years=float(ac_age),
            prior_fault_count=int(previous_faults),
            prior_service_count=int(prior_service_count),
            years_since_last_service=float(maintenance_months) / 12.0,
            fault_last_year=bool(fault_last_year),
        )
        risk["annual_probability"] = official_probability
        risk["monthly_probability"] = float(1 - (1 - official_probability) ** (1 / 12))
        risk["expected_failures"] = official_probability * int(ac_quantity)

    monthly_ac_kwh = operation["每月耗電_kWh"].sum()
    annual_failure_percent = risk["annual_probability"] * 100
    risk_level = official_band if official_risk_active else classify_ac_risk(annual_failure_percent)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("冷氣每月估計耗電", f"{monthly_ac_kwh:,.0f} 度")
    a2.metric("每月估計電費", f"{monthly_ac_kwh * electricity_price:,.0f} 元")
    a3.metric("單台年度故障風險", f"{annual_failure_percent:.1f}%", risk_level)
    a4.metric("全年預期故障台數", f"{risk['expected_failures']:.1f} 台")
    st.caption(
        f"本情境冷氣耗電約為目前月平均總用電的 {monthly_ac_kwh / average_monthly * 100:.1f}%；"
        f"單台每月估計運轉 {risk['per_unit_monthly_hours']:.0f} 小時。"
    )

    chart_left, chart_right = st.columns([3, 2])
    with chart_left:
        time_chart = px.bar(
            operation,
            x="時段",
            y="每月耗電_kWh",
            color="負載率",
            text="每月耗電_kWh",
            color_continuous_scale=[[0, LIGHT_BLUE], [1, ORANGE]],
            title="不同時段的冷氣耗電量",
            labels={"每月耗電_kWh": "每月耗電（度）", "負載率": "負載率"},
        )
        time_chart.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        st.plotly_chart(standard_chart(time_chart), use_container_width=True, config=CHART_CONFIG)
    with chart_right:
        if official_risk_active and official_contributions is not None:
            factor_chart = px.bar(
                official_contributions,
                x="對風險分數的貢獻",
                y="影響因子",
                orientation="h",
                color="對風險分數的貢獻",
                color_continuous_scale=[[0, BLUE], [0.5, "#F3F4F6"], [1, "#B42318"]],
                color_continuous_midpoint=0,
                title="正式 Logistic 模型的風險貢獻",
            )
        else:
            risk_factors = pd.DataFrame(
                {
                    "風險因子": ["機齡", "使用時數", "運轉負載", "保養間隔", "故障紀錄"],
                    "風險倍數": [
                        risk["age_multiplier"],
                        risk["usage_multiplier"],
                        risk["load_multiplier"],
                        risk["maintenance_multiplier"],
                        risk["history_multiplier"],
                    ],
                }
            )
            factor_chart = px.bar(
                risk_factors,
                x="風險因子",
                y="風險倍數",
                color="風險倍數",
                color_continuous_scale=[[0, LIGHT_BLUE], [1, "#B42318"]],
                title="故障風險的主要來源",
            )
            factor_chart.add_hline(y=1, line_dash="dash", line_color=GRAY)
        factor_chart.update_coloraxes(showscale=False)
        st.plotly_chart(standard_chart(factor_chart), use_container_width=True, config=CHART_CONFIG)

    if maintenance_months >= 12 or annual_failure_percent >= 15:
        st.warning("建議優先安排濾網、冷媒壓力、壓縮機電流與室外機散熱檢查，並建立逐台維修紀錄。")
    else:
        st.success("目前情境屬可接受範圍；仍建議每 6–12 個月保養並持續記錄耗電與故障。")

    with st.expander("模型公式、假設與限制"):
        st.markdown(
            "**分時耗電模型**：台數 × 額定功率 × 機型部分負載係數 × 機齡效率修正 × "
            "溫差負載率 × 時數 × 使用率 × 天數。額定功率仍應以銘牌或實測為準。"
        )
        if official_risk_active:
            official_risk = OFFICIAL_MODELS["ac_failure_risk_model"]
            official_metrics = official_risk["validation"]["overall_metrics"]
            st.markdown(
                "**故障風險模型**：以 3,723 筆設備－年度資料建立 L2 Logistic 迴歸，特徵包含機齡、"
                "既往故障、維修／保養次數、距上次維修時間、前一年故障、館別、廠牌與型號族群。"
            )
            st.write(
                f"年份向前驗證 ROC-AUC 為 {official_metrics['roc_auc']:.3f}、"
                f"Brier score 為 {official_metrics['brier_score']:.3f}。此結果適合做相對優先順序，"
                "不應視為保固或安全認證的精準故障率。"
            )
            st.warning(
                "現有清冊只含仍在場設備，已汰換設備缺席，存在存活者偏差；另有 287 筆歷史事件無法連回現有清冊。"
            )
        else:
            st.write("目前使用工程比例風險假設，尚未連接正式修繕標籤模型。")

    download_operation = operation[["時段", "估計室外溫度_C", "使用率_%", "負載率", "每日耗電_kWh", "每月耗電_kWh"]].copy()
    download_operation["冷氣機型"] = ac_model_name
    download_operation["單台年度故障機率_%"] = annual_failure_percent
    download_operation["全年預期故障台數"] = risk["expected_failures"]
    st.download_button(
        "下載冷氣耗電與故障風險分析",
        download_operation.round(3).to_csv(index=False).encode("utf-8-sig"),
        "冷氣耗電與故障風險分析.csv",
        "text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("批次設備清冊分析")
    st.write("可上傳完整 Excel 蒐集工作簿，或只含冷氣設備欄位的 CSV／Excel；系統會依風險由高到低排序。")
    batch_left, batch_right = st.columns(2)
    with batch_left:
        ac_batch_upload = st.file_uploader(
            "上傳冷氣設備清冊",
            type=["csv", "xlsx"],
            key="ac_batch_upload",
        )
    with batch_right:
        st.download_button(
            "下載冷氣清冊 CSV 範本",
            AC_BATCH_SAMPLE.to_csv(index=False).encode("utf-8-sig"),
            "冷氣設備清冊範本.csv",
            "text/csv",
            use_container_width=True,
        )

    if ac_batch_upload:
        try:
            ac_batch_source = read_ac_upload(ac_batch_upload.getvalue(), ac_batch_upload.name)
        except Exception as exc:
            st.error(f"冷氣設備清冊無法讀取：{exc}")
        else:
            ac_batch_source = ac_batch_source.dropna(how="all")
            batch_missing = [column for column in AC_BATCH_REQUIRED if column not in ac_batch_source.columns]
            if batch_missing:
                st.error("冷氣設備清冊缺少欄位：" + "、".join(batch_missing))
            elif ac_batch_source.empty:
                st.error("冷氣設備清冊沒有可分析的資料列。")
            else:
                ac_batch_df = ac_batch_source[AC_BATCH_REQUIRED].copy()
                text_columns = ["設備群組_ID", "位置", "冷氣機型"]
                numeric_columns = [column for column in AC_BATCH_REQUIRED if column not in text_columns]
                for column in numeric_columns:
                    ac_batch_df[column] = pd.to_numeric(ac_batch_df[column], errors="coerce")

                batch_errors = []
                blank_text = ac_batch_df[text_columns].fillna("").astype(str).apply(lambda values: values.str.strip().eq(""))
                if blank_text.any().any() or ac_batch_df[numeric_columns].isna().any().any():
                    batch_errors.append("有空白或非數字內容")
                if ac_batch_df["設備群組_ID"].duplicated().any():
                    batch_errors.append("設備群組 ID 重複")
                invalid_models = sorted(set(ac_batch_df["冷氣機型"].dropna()) - set(AC_MODELS))
                if invalid_models:
                    batch_errors.append("未知冷氣機型：" + "、".join(map(str, invalid_models)))

                rate_columns = ["使用率_00_06", "使用率_06_12", "使用率_12_18", "使用率_18_24"]
                for column in rate_columns:
                    ac_batch_df[column] = np.where(
                        ac_batch_df[column].between(0, 1),
                        ac_batch_df[column] * 100,
                        ac_batch_df[column],
                    )

                range_checks = {
                    "額定功率_kW": (0.2, 10),
                    "台數": (1, 2000),
                    "平均機齡_年": (0, 30),
                    "日平均溫度_C": (-10, 45),
                    "設定溫度_C": (18, 30),
                    "每月運轉天數": (1, 31),
                    "距上次保養_月": (0, 60),
                    "近兩年故障次數": (0, 10),
                }
                range_checks.update({column: (0, 100) for column in rate_columns})
                for column, (lower, upper) in range_checks.items():
                    if not ac_batch_df[column].between(lower, upper).all():
                        batch_errors.append(f"{column} 必須介於 {lower} 和 {upper}")
                integer_columns = ["台數", "每月運轉天數", "距上次保養_月", "近兩年故障次數"]
                for column in integer_columns:
                    if not np.allclose(ac_batch_df[column].dropna(), np.round(ac_batch_df[column].dropna())):
                        batch_errors.append(f"{column} 必須是整數")

                if batch_errors:
                    st.error("；".join(dict.fromkeys(batch_errors)))
                else:
                    batch_results = []
                    for _, ac_row in ac_batch_df.iterrows():
                        group_operation, group_risk = estimate_ac_operation(
                            model_name=str(ac_row["冷氣機型"]),
                            quantity=int(ac_row["台數"]),
                            rated_power_kw=float(ac_row["額定功率_kW"]),
                            average_temp=float(ac_row["日平均溫度_C"]),
                            setpoint=float(ac_row["設定溫度_C"]),
                            operating_days=int(ac_row["每月運轉天數"]),
                            usage_rates=[float(ac_row[column]) for column in rate_columns],
                            age_years=float(ac_row["平均機齡_年"]),
                            maintenance_months=int(ac_row["距上次保養_月"]),
                            previous_faults=int(ac_row["近兩年故障次數"]),
                        )
                        group_probability = group_risk["annual_probability"] * 100
                        group_level = classify_ac_risk(group_probability)
                        recommendation = (
                            "優先檢查／評估汰換"
                            if group_probability >= 15 or ac_row["距上次保養_月"] >= 12
                            else "依週期保養"
                        )
                        batch_results.append(
                            {
                                "設備群組_ID": ac_row["設備群組_ID"],
                                "位置": ac_row["位置"],
                                "冷氣機型": ac_row["冷氣機型"],
                                "台數": int(ac_row["台數"]),
                                "每月耗電_kWh": group_operation["每月耗電_kWh"].sum(),
                                "單台年度故障機率_%": group_probability,
                                "全年預期故障台數": group_risk["expected_failures"],
                                "風險等級": group_level,
                                "建議": recommendation,
                            }
                        )

                    batch_result_df = pd.DataFrame(batch_results).sort_values(
                        "單台年度故障機率_%", ascending=False
                    )
                    b1, b2, b3 = st.columns(3)
                    b1.metric("設備群組", f"{len(batch_result_df)} 組")
                    b2.metric("估計月耗電合計", f"{batch_result_df['每月耗電_kWh'].sum():,.0f} 度")
                    b3.metric(
                        "高／極高風險群組",
                        f"{batch_result_df['風險等級'].isin(['高', '極高']).sum()} 組",
                    )

                    batch_chart = px.bar(
                        batch_result_df,
                        x="設備群組_ID",
                        y="單台年度故障機率_%",
                        color="風險等級",
                        text="單台年度故障機率_%",
                        category_orders={"風險等級": ["低", "中", "高", "極高"]},
                        color_discrete_map={"低": GREEN, "中": ORANGE, "高": "#B54708", "極高": "#B42318"},
                        title="冷氣設備群組故障風險排名",
                        labels={"單台年度故障機率_%": "年度故障機率（%）"},
                    )
                    batch_chart.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    st.plotly_chart(standard_chart(batch_chart), use_container_width=True, config=CHART_CONFIG)
                    st.dataframe(batch_result_df.round(2), use_container_width=True, hide_index=True)
                    st.download_button(
                        "下載批次冷氣風險排名",
                        batch_result_df.round(3).to_csv(index=False).encode("utf-8-sig"),
                        "批次冷氣耗電與故障風險排名.csv",
                        "text/csv",
                        use_container_width=True,
                    )

st.divider()
st.caption("提示：圖表中的『度』就是 kWh；把滑鼠移到圖上，可以看到詳細數字。")
