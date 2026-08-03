import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="宿舍用電分析與節能決策系統", page_icon="⚡", layout="wide")
st.title("⚡ 校園宿舍用電分析與節能決策系統")
st.caption("以月用電、住宿人數、溫度與熱水用電，分析能源結構並模擬節能效益。")

SAMPLE = pd.DataFrame(
    {
        "月份": range(1, 13),
        "總用電_kWh": [19521, 17634, 45606, 42192, 50989, 41913, 4682, 5681, 48200, 46000, 44000, 49000],
        "熱水用電_kWh": [6192, 5885, 15274, 9843, 11635, 7142, 1123, 1084, 12650, 11500, 11000, 12000],
        "住宿天數": [16, 12, 31, 30, 31, 23, 0, 0, 24, 31, 30, 31],
        "住宿人數": [653] * 12,
        "平均溫度_C": [15.2, 15.6, 17.5, 21.2, 24.1, 27.2, 28.9, 28.7, 27.1, 24.1, 20.6, 16.9],
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

df["人日"] = df["住宿天數"] * df["住宿人數"]
df["照明用電"] = df["人日"] * lighting_rate
df["冷氣用電"] = np.maximum(0, df["平均溫度_C"] - ac_base_temp) * df["人日"] * 0.29
df["基礎負載"] = base_load
df["其他用電"] = (
    df["總用電_kWh"] - df["熱水用電_kWh"] - df["照明用電"] - df["冷氣用電"] - df["基礎負載"]
).clip(lower=0)
df["每人日用電"] = np.where(df["人日"] > 0, df["總用電_kWh"] / df["人日"], np.nan)

total = df["總用電_kWh"].sum()
peak = df.loc[df["總用電_kWh"].idxmax()]
occupied = df.loc[df["人日"] > 0]
unit_kwh = occupied["總用電_kWh"].sum() / occupied["人日"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("年度總用電", f"{total:,.0f} kWh")
c2.metric("估計電費", f"NT$ {total * electricity_price:,.0f}")
c3.metric("最高用電月份", f"{int(peak['月份'])} 月", f"{peak['總用電_kWh']:,.0f} kWh")
c4.metric("每人日平均用電", f"{unit_kwh:.2f} kWh")

tab1, tab2, tab3, tab4 = st.tabs(["總覽", "能源拆解", "節能模擬", "資料與預測"])

with tab1:
    left, right = st.columns([2, 1])
    with left:
        fig = px.line(df, x="月份", y="總用電_kWh", markers=True, title="每月總用電趨勢")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(df, x="平均溫度_C", y="總用電_kWh", size="人日", color="月份", title="溫度與用電關係")
        st.plotly_chart(fig, use_container_width=True)
    st.info(f"本年度最高用電出現在 {int(peak['月份'])} 月。全年估計碳排為 {total * carbon_factor / 1000:,.1f} 公噸 CO₂e。")

with tab2:
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

with tab3:
    st.subheader("節能情境設定")
    a, b, c = st.columns(3)
    temp_raise = a.slider("冷氣設定提高（°C）", 0.0, 3.0, 1.0, 0.5)
    lighting_saving = b.slider("照明節電率", 0, 50, 20, 5) / 100
    hotwater_saving = c.slider("熱水節電率", 0, 40, 10, 5) / 100
    new_ac = (np.maximum(0, df["平均溫度_C"] - (ac_base_temp + temp_raise)) * df["人日"] * 0.29).sum()
    saving = (df["冷氣用電"].sum() - new_ac) + df["照明用電"].sum() * lighting_saving + df["熱水用電_kWh"].sum() * hotwater_saving
    s1, s2, s3 = st.columns(3)
    s1.metric("預估節電量", f"{saving:,.0f} kWh", f"{saving / total * 100:.1f}%")
    s2.metric("預估節省電費", f"NT$ {saving * electricity_price:,.0f}")
    s3.metric("預估減碳", f"{saving * carbon_factor / 1000:,.2f} 公噸 CO₂e")
    before_after = pd.DataFrame({"情境": ["目前", "改善後"], "用電量_kWh": [total, total - saving]})
    st.plotly_chart(px.bar(before_after, x="情境", y="用電量_kWh", color="情境", title="節能前後比較"), use_container_width=True)

with tab4:
    model_df = occupied.copy()
    if len(model_df) >= 3:
        features = model_df[["平均溫度_C", "人日"]]
        target = model_df["總用電_kWh"]
        model = LinearRegression().fit(features, target)
        model_df["預測用電_kWh"] = model.predict(features)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=model_df["月份"], y=model_df["總用電_kWh"], name="實際", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=model_df["月份"], y=model_df["預測用電_kWh"], name="模型預測", mode="lines+markers"))
        fig.update_layout(title=f"線性迴歸模型（R² = {model.score(features, target):.3f}）", xaxis_title="月份", yaxis_title="kWh")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("模型以平均溫度與住宿人日解釋總用電；資料量少時只適合展示方法，不宜視為精準預報。")
    shown = df.round(2)
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button("下載分析結果 CSV", shown.to_csv(index=False).encode("utf-8-sig"), "宿舍用電分析結果.csv", "text/csv")

