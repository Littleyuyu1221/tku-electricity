import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="宿舍用電分析與節能決策系統",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 高齡友善：大字、高對比、大型操作區，並減少畫面中的次要元素。
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #111827; }
    html, body, [class*="css"] { font-size: 19px; }
    h1 { font-size: 2.25rem !important; line-height: 1.25 !important; color: #0b356b !important; }
    h2, h3 { color: #0b356b !important; }
    [data-testid="stMetric"] {
        background: #f4f8ff;
        border: 2px solid #9ab8df;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem !important; color: #24364b !important; }
    [data-testid="stMetricValue"] { font-size: 1.75rem !important; color: #082f63 !important; }
    [data-testid="stSidebar"] { background: #eef5ff; }
    [data-testid="stSidebar"] * { font-size: 1rem; }
    .stButton button, .stDownloadButton button {
        min-height: 48px;
        font-size: 1.05rem;
        border: 2px solid #315f97;
    }
    div[data-baseweb="select"] > div { min-height: 52px; font-size: 1.1rem; }
    .stAlert { font-size: 1.05rem; border-width: 2px; }
    .stDataFrame { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
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


def senior_chart(fig: go.Figure, height: int = 430) -> go.Figure:
    """套用一致的大字、高對比圖表格式。"""
    fig.update_layout(
        height=height,
        font=dict(size=18, color="#111827"),
        title=dict(font=dict(size=24, color="#0B356B"), x=0.01),
        legend=dict(font=dict(size=17), orientation="h", y=1.12, x=0),
        hoverlabel=dict(font_size=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=65, r=45, t=90, b=65),
    )
    fig.update_xaxes(
        title_font=dict(size=18),
        tickfont=dict(size=17),
        gridcolor="#D8DEE8",
        linecolor="#64748B",
        linewidth=2,
    )
    fig.update_yaxes(
        title_font=dict(size=18),
        tickfont=dict(size=17),
        gridcolor="#D8DEE8",
        linecolor="#64748B",
        linewidth=2,
    )
    return fig


st.title("⚡ 宿舍用電分析與節能系統")
st.write("用簡單的大字與圖表，看懂住宿人數、開館天數、溫度和用電的關係。")

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
    st.header("⚙️ 簡單設定")
    uploaded = st.file_uploader("選擇用電資料", type=["csv", "xlsx"])
    st.caption("沒有選擇檔案時，會使用示範資料。")
    st.download_button(
        "下載資料填寫範本",
        SAMPLE.to_csv(index=False).encode("utf-8-sig"),
        "宿舍用電資料範本.csv",
        "text/csv",
        use_container_width=True,
    )
    bed_capacity = st.number_input("宿舍總床位數", min_value=1, value=700, step=10)

    with st.expander("進階設定（一般不用更改）"):
        ac_base_temp = st.slider("冷氣啟用溫度（°C）", 20.0, 28.0, 23.0, 0.5)
        lighting_rate = st.slider("每人每日照明用電（kWh）", 0.1, 2.0, 1.15, 0.05)
        base_load = st.number_input("每月基礎用電（kWh）", min_value=0.0, value=4164.0, step=100.0)
        electricity_price = st.number_input("每度平均電價（元）", min_value=0.0, value=4.0, step=0.1)
        carbon_factor = st.number_input("每度電碳排（kg CO₂e）", min_value=0.0, value=0.494, step=0.001)

try:
    source = read_upload(uploaded.getvalue(), uploaded.name) if uploaded else SAMPLE.copy()
except Exception as exc:
    st.error(f"檔案無法讀取：{exc}")
    st.stop()

if "平均溫度_C" in source.columns and "日平均溫度_C" not in source.columns:
    source = source.rename(columns={"平均溫度_C": "日平均溫度_C"})

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
if not df["月份"].between(1, 12).all() or df["月份"].duplicated().any():
    st.error("月份必須介於 1 到 12，而且不可重複。")
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

df = df.sort_values("月份").reset_index(drop=True)
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
c1.metric("全年用了多少電", f"{total:,.0f} 度")
c2.metric("大約需要多少電費", f"{total * electricity_price:,.0f} 元")
c3.metric("用電最多的月份", f"{int(peak['月份'])} 月")
c4.metric("每人每天平均用電", f"{unit_kwh:.2f} 度")

page = st.selectbox(
    "👇 請選擇想看的內容",
    [
        "① 快速看懂用電",
        "② 住宿人數與開館",
        "③ 電都用到哪裡",
        "④ 可以省多少電",
        "⑤ 下個月用電預測",
    ],
)

if page == "① 快速看懂用電":
    st.header("① 快速看懂用電")
    st.info(
        f"重點：{int(peak['月份'])} 月用電最多（{peak['總用電_kWh']:,.0f} 度）；"
        f"{int(lowest['月份'])} 月最少（{lowest['總用電_kWh']:,.0f} 度）。"
    )

    trend = go.Figure()
    trend.add_trace(
        go.Scatter(
            x=df["月份"],
            y=df["總用電_kWh"],
            name="每月用電",
            mode="lines+markers+text",
            text=[f"{value / 1000:.0f}千" for value in df["總用電_kWh"]],
            textposition="top center",
            line=dict(color=BLUE, width=5),
            marker=dict(size=12),
            hovertemplate="%{x} 月：%{y:,.0f} 度<extra></extra>",
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
    trend.update_layout(title="每個月用了多少電？", xaxis_title="月份", yaxis_title="用電量（度）", showlegend=False)
    trend.update_xaxes(dtick=1)
    st.plotly_chart(senior_chart(trend), use_container_width=True, config=CHART_CONFIG)

    ranking = df.sort_values("總用電_kWh", ascending=True)
    rank_fig = px.bar(
        ranking,
        x="總用電_kWh",
        y=ranking["月份"].astype(int).astype(str) + " 月",
        orientation="h",
        text="總用電_kWh",
        color="總用電_kWh",
        color_continuous_scale=[[0, LIGHT_BLUE], [1, BLUE]],
        title="月份用電量排名",
        labels={"總用電_kWh": "用電量（度）", "y": "月份"},
    )
    rank_fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    rank_fig.update_coloraxes(showscale=False)
    st.plotly_chart(senior_chart(rank_fig, 500), use_container_width=True, config=CHART_CONFIG)

elif page == "② 住宿人數與開館":
    st.header("② 住宿人數與開館")
    st.info(
        f"全年共住宿 {total_stay_days} 天、開館 {total_open_days} 天；"
        f"有住宿月份的平均住房率約 {average_occupancy:.1f}%。"
    )

    operation_fig = go.Figure()
    operation_fig.add_trace(go.Bar(x=df["月份"], y=df["住宿天數"], name="住宿天數", marker_color=BLUE))
    operation_fig.add_trace(go.Bar(x=df["月份"], y=df["開館天數"], name="開館天數", marker_color=ORANGE))
    operation_fig.add_trace(
        go.Scatter(
            x=df["月份"],
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
        xaxis=dict(title="月份", dtick=1),
        yaxis=dict(title="天數"),
        yaxis2=dict(title="溫度（°C）", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(senior_chart(operation_fig), use_container_width=True, config=CHART_CONFIG)

    left, right = st.columns([3, 2])
    with left:
        bubble = px.scatter(
            df,
            x="住宿人數",
            y="總用電_kWh",
            size="開館天數",
            color="日平均溫度_C",
            text=df["月份"].astype(int).astype(str) + "月",
            color_continuous_scale=[[0, BLUE], [0.55, ORANGE], [1, "#B42318"]],
            title="人數、溫度與用電的關係",
            labels={"住宿人數": "住宿人數（人）", "總用電_kWh": "用電量（度）", "日平均溫度_C": "溫度（°C）"},
        )
        bubble.update_traces(marker=dict(line=dict(width=2, color="#FFFFFF")))
        st.plotly_chart(senior_chart(bubble), use_container_width=True, config=CHART_CONFIG)
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
        st.plotly_chart(senior_chart(gauge), use_container_width=True, config=CHART_CONFIG)
        st.caption(f"開館天數加權的平均溫度：{weighted_temp:.1f} °C")

elif page == "③ 電都用到哪裡":
    st.header("③ 電都用到哪裡")
    long_df = df.melt(
        id_vars="月份",
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
    st.info(f"重點：目前估算占比最高的是「{main_category['用電類別']}」，全年約 {main_category['用電量_kWh']:,.0f} 度。")

    area = px.area(
        long_df,
        x="月份",
        y="用電量_kWh",
        color="用電類別",
        color_discrete_sequence=COLORS,
        title="每月各類用電的變化",
        labels={"用電量_kWh": "用電量（度）"},
    )
    area.update_xaxes(dtick=1)
    st.plotly_chart(senior_chart(area), use_container_width=True, config=CHART_CONFIG)

    left, right = st.columns(2)
    with left:
        pie = px.pie(
            sums,
            values="用電量_kWh",
            names="用電類別",
            hole=0.42,
            color_discrete_sequence=COLORS,
            title="全年用電占比",
        )
        pie.update_traces(textposition="inside", textinfo="label+percent", textfont_size=17)
        st.plotly_chart(senior_chart(pie), use_container_width=True, config=CHART_CONFIG)
    with right:
        pivot = long_df.pivot(index="用電類別", columns="月份", values="用電量_kWh").reindex(sums["用電類別"])
        heatmap = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=[f"{int(month)}月" for month in pivot.columns],
                y=pivot.index,
                colorscale=[[0, "#F2F7FF"], [1, BLUE]],
                colorbar=dict(title="度", tickfont=dict(size=15)),
                hovertemplate="%{y}<br>%{x}<br>%{z:,.0f} 度<extra></extra>",
            )
        )
        heatmap.update_layout(title="深色表示用電較多")
        st.plotly_chart(senior_chart(heatmap), use_container_width=True, config=CHART_CONFIG)

elif page == "④ 可以省多少電":
    st.header("④ 可以省多少電")
    st.write("請拖動下面三個設定，圖表和金額會立即更新。")
    temp_raise = st.slider("冷氣溫度提高幾度？", 0.0, 3.0, 1.0, 0.5)
    lighting_saving = st.slider("照明預計節省百分比", 0, 50, 20, 5) / 100
    hotwater_saving = st.slider("熱水預計節省百分比", 0, 40, 10, 5) / 100

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
    s1.metric("一年可以省電", f"{saving:,.0f} 度", f"約 {saving_rate:.1f}%")
    s2.metric("一年可以省錢", f"{saving * electricity_price:,.0f} 元")
    s3.metric("一年可以減碳", f"{saving * carbon_factor / 1000:,.2f} 公噸")

    before_after = pd.DataFrame({"情況": ["改善前", "改善後"], "用電量": [total, total - saving]})
    compare = px.bar(
        before_after,
        x="情況",
        y="用電量",
        color="情況",
        text="用電量",
        color_discrete_map={"改善前": GRAY, "改善後": GREEN},
        title="改善前後用電比較",
        labels={"用電量": "全年用電（度）"},
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
        st.plotly_chart(senior_chart(compare), use_container_width=True, config=CHART_CONFIG)
    with right:
        st.plotly_chart(senior_chart(contributions), use_container_width=True, config=CHART_CONFIG)

elif page == "⑤ 下個月用電預測":
    st.header("⑤ 下個月用電預測")
    st.info("輸入下個月預估的人數、天數和溫度，系統會試算可能的用電量。")

    if len(occupied) < 3:
        st.warning("有住宿的月份少於 3 個，目前無法建立預測模型。")
    else:
        model_df = occupied.copy()
        features = model_df[["日平均溫度_C", "人日", "開館天數"]]
        target = model_df["總用電_kWh"]
        model = LinearRegression().fit(features, target)
        model_df["預測用電_kWh"] = np.maximum(0, model.predict(features))
        score = model.score(features, target)

        predicted_chart = go.Figure()
        predicted_chart.add_trace(
            go.Scatter(
                x=model_df["月份"],
                y=model_df["總用電_kWh"],
                name="實際用電",
                mode="lines+markers",
                line=dict(color=BLUE, width=5),
                marker=dict(size=11),
            )
        )
        predicted_chart.add_trace(
            go.Scatter(
                x=model_df["月份"],
                y=model_df["預測用電_kWh"],
                name="模型估計",
                mode="lines+markers",
                line=dict(color=ORANGE, width=4, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
            )
        )
        predicted_chart.update_layout(title="過去實際用電和模型估計", xaxis_title="月份", yaxis_title="用電量（度）")
        predicted_chart.update_xaxes(dtick=1)
        st.plotly_chart(senior_chart(predicted_chart), use_container_width=True, config=CHART_CONFIG)

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
            forecast_input = pd.DataFrame(
                [[forecast_temp, forecast_person_days, forecast_open_days]],
                columns=["日平均溫度_C", "人日", "開館天數"],
            )
            forecast = max(0, model.predict(forecast_input)[0])
            f1, f2, f3 = st.columns(3)
            f1.metric("下個月可能用電", f"{forecast:,.0f} 度")
            f2.metric("下個月可能電費", f"{forecast * electricity_price:,.0f} 元")
            f3.metric("下個月可能碳排", f"{forecast * carbon_factor / 1000:,.2f} 公噸")

        with st.expander("模型說明（需要時再看）"):
            st.write(
                f"模型參考日平均溫度、住宿人日與開館天數。現在的解釋力 R² 為 {score:.3f}。"
                "資料只有 12 個月時，結果適合做專題展示與情境比較，不應視為精準的電費預報。"
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

st.divider()
st.caption("提示：圖表中的『度』就是 kWh；把滑鼠移到圖上，可以看到詳細數字。")
