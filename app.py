import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="宿舍用電分析與節能決策系統", page_icon="⚡", layout="wide")
st.title("⚡ 校園宿舍用電分析與節能決策系統")
st.caption("整合月用電、住宿人數、住宿與開館天數、日平均溫度，分析能源效率並模擬節能效益。")

SAMPLE = pd.DataFrame(
    {
        "月份": range(1, 13),
        "總用電_kWh": [19521, 17634, 45606, 42192, 50989, 41913, 4682, 5681, 48200, 46000, 44000, 49000],
        "熱水用電_kWh": [6192, 5885, 15274, 9843, 11635, 7142, 1123, 1084, 12650, 11500, 11000, 12000],
        "住宿天數": [16, 12, 31, 30, 31, 23, 0, 0, 24, 31, 30, 31],
        "住宿人數": [653, 653, 653, 653, 653, 653, 0, 0, 653, 653, 653, 653],
        "開館天數": [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "日平均溫度_C": [15.2, 15.6, 17.5, 21.2, 24.1, 27.2, 28.9, 28.7, 27.1, 24.1, 20.6, 16.9],
    }
)
REQUIRED = list(SAMPLE.columns)


@st.cache_data
def read_upload(raw: bytes, name: str) -> pd.DataFrame:
    stream = io.BytesIO(raw)
    return pd.read_excel(stream) if name.lower().endswith(".xlsx") else pd.read_csv(stream)


with st.sidebar:
    st.header("資料與參數")
    uploaded = st.file_uploader("上傳 CSV 或 Excel", type=["csv", "xlsx"])
    st.caption("未上傳時使用 2024 年示範資料。")
    st.download_button(
        "下載資料填寫範本",
        SAMPLE.to_csv(index=False).encode("utf-8-sig"),
        "宿舍用電資料範本.csv",
        "text/csv",
    )
    bed_capacity = st.number_input("宿舍核定床位數", min_value=1, value=700, step=10)
    ac_base_temp = st.slider("冷氣啟用基準溫度（°C）", 20.0, 28.0, 23.0, 0.5)
    lighting_rate = st.slider("每人每日照明用電（kWh）", 0.1, 2.0, 1.15, 0.05)
    base_load = st.number_input("每月基礎負載（kWh）", min_value=0.0, value=4164.0, step=100.0)
    electricity_price = st.number_input("平均電價（元/kWh）", min_value=0.0, value=4.0, step=0.1)
    carbon_factor = st.number_input("碳排係數（kg CO₂e/kWh）", min_value=0.0, value=0.494, step=0.001)

try:
    source = read_upload(uploaded.getvalue(), uploaded.name) if uploaded else SAMPLE.copy()
except Exception as exc:
    st.error(f"無法讀取檔案：{exc}")
    st.stop()

# 相容舊版欄位名稱，避免既有檔案無法使用。
if "平均溫度_C" in source.columns and "日平均溫度_C" not in source.columns:
    source = source.rename(columns={"平均溫度_C": "日平均溫度_C"})

missing = [column for column in REQUIRED if column not in source.columns]
if missing:
    st.error("缺少必要欄位：" + "、".join(missing))
    st.info("必要欄位為：" + "、".join(REQUIRED))
    st.stop()

df = source[REQUIRED].copy()
for column in REQUIRED:
    df[column] = pd.to_numeric(df[column], errors="coerce")
if df.isna().any().any():
    st.error("資料含有空值或非數字內容，請修正後重新上傳。")
    st.stop()

if not df["月份"].between(1, 12).all():
    st.error("月份必須介於 1 到 12。")
    st.stop()
if (df.drop(columns=["日平均溫度_C"]) < 0).any().any():
    st.error("除溫度外，其餘欄位不可為負數。")
    st.stop()
if (df["住宿天數"] > df["開館天數"]).any():
    st.error("住宿天數不可大於開館天數，請檢查上傳資料。")
    st.stop()
if (df["住宿人數"] > bed_capacity).any():
    st.warning("部分月份的住宿人數高於核定床位數，請確認資料或調整側欄床位數。")

df["人日"] = df["住宿天數"] * df["住宿人數"]
df["住房率_%"] = np.where(df["住宿天數"] > 0, df["住宿人數"] / bed_capacity * 100, 0)
df["開館使用率_%"] = np.where(df["開館天數"] > 0, df["住宿天數"] / df["開館天數"] * 100, 0)
df["照明用電"] = df["人日"] * lighting_rate
df["冷房度日"] = np.maximum(0, df["日平均溫度_C"] - ac_base_temp) * df["開館天數"]
df["冷氣用電"] = np.maximum(0, df["日平均溫度_C"] - ac_base_temp) * df["人日"] * 0.29
df["基礎負載"] = base_load
df["其他用電"] = (
    df["總用電_kWh"] - df["熱水用電_kWh"] - df["照明用電"] - df["冷氣用電"] - df["基礎負載"]
).clip(lower=0)
df["每人日用電"] = np.where(df["人日"] > 0, df["總用電_kWh"] / df["人日"], np.nan)
df["每開館日用電"] = np.where(df["開館天數"] > 0, df["總用電_kWh"] / df["開館天數"], np.nan)

total = df["總用電_kWh"].sum()
peak = df.loc[df["總用電_kWh"].idxmax()]
occupied = df.loc[df["人日"] > 0]
unit_kwh = occupied["總用電_kWh"].sum() / occupied["人日"].sum()
total_stay_days = int(df["住宿天數"].sum())
total_open_days = int(df["開館天數"].sum())
weighted_temp = np.average(df["日平均溫度_C"], weights=df["開館天數"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("年度總用電", f"{total:,.0f} kWh")
c2.metric("估計電費", f"NT$ {total * electricity_price:,.0f}")
c3.metric("最高用電月份", f"{int(peak['月份'])} 月", f"{peak['總用電_kWh']:,.0f} kWh")
c4.metric("每人日平均用電", f"{unit_kwh:.2f} kWh")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["總覽", "住宿與營運", "能源拆解", "節能模擬", "資料與預測"])

with tab1:
    left, right = st.columns([2, 1])
    with left:
        fig = px.line(df, x="月份", y="總用電_kWh", markers=True, title="每月總用電趨勢")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(df, x="日平均溫度_C", y="總用電_kWh", size="開館天數", color="住宿人數", title="溫度、開館與用電關係")
        st.plotly_chart(fig, use_container_width=True)
    st.info(f"本年度最高用電出現在 {int(peak['月份'])} 月。全年估計碳排為 {total * carbon_factor / 1000:,.1f} 公噸 CO₂e。")

with tab2:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("全年住宿天數", f"{total_stay_days} 天")
    m2.metric("全年開館天數", f"{total_open_days} 天")
    m3.metric("開館加權平均溫度", f"{weighted_temp:.1f} °C")
    m4.metric("最高住宿人數", f"{df['住宿人數'].max():,.0f} 人")

    operation_fig = go.Figure()
    operation_fig.add_trace(go.Bar(x=df["月份"], y=df["住宿天數"], name="住宿天數"))
    operation_fig.add_trace(go.Bar(x=df["月份"], y=df["開館天數"], name="開館天數"))
    operation_fig.add_trace(
        go.Scatter(
            x=df["月份"],
            y=df["日平均溫度_C"],
            name="日平均溫度",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    operation_fig.update_layout(
        title="住宿、開館天數與溫度",
        barmode="group",
        xaxis=dict(title="月份", dtick=1),
        yaxis=dict(title="天數"),
        yaxis2=dict(title="溫度（°C）", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(operation_fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        people_fig = px.bar(
            df,
            x="月份",
            y="住宿人數",
            color="住房率_%",
            text_auto=".0f",
            title="每月住宿人數與住房率",
            labels={"住房率_%": "住房率（%）"},
        )
        people_fig.update_xaxes(dtick=1)
        st.plotly_chart(people_fig, use_container_width=True)
    with right:
        efficiency_fig = px.line(
            df,
            x="月份",
            y=["每開館日用電", "每人日用電"],
            markers=True,
            title="單位用電效率",
            labels={"value": "用電量（kWh）", "variable": "指標"},
        )
        efficiency_fig.update_xaxes(dtick=1)
        st.plotly_chart(efficiency_fig, use_container_width=True)

with tab3:
    long_df = df.melt(
        id_vars="月份",
        value_vars=["基礎負載", "照明用電", "熱水用電_kWh", "冷氣用電", "其他用電"],
        var_name="用電類別",
        value_name="用電量_kWh",
    )
    fig = px.bar(long_df, x="月份", y="用電量_kWh", color="用電類別", barmode="stack", title="每月能源結構")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    sums = long_df.groupby("用電類別", as_index=False)["用電量_kWh"].sum()
    st.plotly_chart(px.pie(sums, values="用電量_kWh", names="用電類別", hole=0.45, title="全年用電占比"), use_container_width=True)

with tab4:
    st.subheader("節能情境設定")
    a, b, c = st.columns(3)
    temp_raise = a.slider("冷氣設定提高（°C）", 0.0, 3.0, 1.0, 0.5)
    lighting_saving = b.slider("照明節電率", 0, 50, 20, 5) / 100
    hotwater_saving = c.slider("熱水節電率", 0, 40, 10, 5) / 100
    new_ac = (np.maximum(0, df["日平均溫度_C"] - (ac_base_temp + temp_raise)) * df["人日"] * 0.29).sum()
    saving = (df["冷氣用電"].sum() - new_ac) + df["照明用電"].sum() * lighting_saving + df["熱水用電_kWh"].sum() * hotwater_saving
    s1, s2, s3 = st.columns(3)
    s1.metric("預估節電量", f"{saving:,.0f} kWh", f"{saving / total * 100:.1f}%")
    s2.metric("預估節省電費", f"NT$ {saving * electricity_price:,.0f}")
    s3.metric("預估減碳", f"{saving * carbon_factor / 1000:,.2f} 公噸 CO₂e")
    before_after = pd.DataFrame({"情境": ["目前", "改善後"], "用電量_kWh": [total, total - saving]})
    st.plotly_chart(px.bar(before_after, x="情境", y="用電量_kWh", color="情境", title="節能前後比較"), use_container_width=True)

with tab5:
    model_df = occupied.copy()
    if len(model_df) >= 3:
        features = model_df[["日平均溫度_C", "人日", "開館天數"]]
        target = model_df["總用電_kWh"]
        model = LinearRegression().fit(features, target)
        model_df["預測用電_kWh"] = model.predict(features)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=model_df["月份"], y=model_df["總用電_kWh"], name="實際", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=model_df["月份"], y=model_df["預測用電_kWh"], name="模型預測", mode="lines+markers"))
        fig.update_layout(title=f"線性迴歸模型（R² = {model.score(features, target):.3f}）", xaxis_title="月份", yaxis_title="kWh")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("模型以日平均溫度、住宿人日與開館天數解釋總用電；資料量少時只適合展示方法，不宜視為精準預報。")

        st.subheader("下個月用電情境預測")
        p1, p2, p3, p4 = st.columns(4)
        forecast_temp = p1.number_input("預估日平均溫度（°C）", value=24.0, step=0.5)
        forecast_people = p2.number_input("預估住宿人數", min_value=0, value=min(653, bed_capacity), step=10)
        forecast_stay_days = p3.number_input("預估住宿天數", min_value=0, max_value=31, value=30)
        forecast_open_days = p4.number_input("預估開館天數", min_value=1, max_value=31, value=30)
        if forecast_stay_days > forecast_open_days:
            st.warning("預估住宿天數不可大於開館天數。")
        else:
            forecast_person_days = forecast_people * forecast_stay_days
            forecast = model.predict(
                pd.DataFrame(
                    [[forecast_temp, forecast_person_days, forecast_open_days]],
                    columns=["日平均溫度_C", "人日", "開館天數"],
                )
            )[0]
            forecast = max(0, forecast)
            f1, f2, f3 = st.columns(3)
            f1.metric("預測月用電", f"{forecast:,.0f} kWh")
            f2.metric("預測月電費", f"NT$ {forecast * electricity_price:,.0f}")
            f3.metric("預測月碳排", f"{forecast * carbon_factor / 1000:,.2f} 公噸 CO₂e")
    shown = df.round(2)
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button("下載分析結果 CSV", shown.to_csv(index=False).encode("utf-8-sig"), "宿舍用電分析結果.csv", "text/csv")
