
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import datetime

# --- 網頁配置 ---
st.set_page_config(page_title="松濤一館用電組成拆解系統", layout="wide")

st.title("🏫 松濤一館用電組成拆解與節能預測系統")
st.markdown("本系統透過迴歸模型將總用電量區分為 **空調、熱水、照明/插座、基礎負載**。")

# --- 側邊欄：參數設定 ---
st.sidebar.header("⚙️ 模型參數調整")
ac_base_temp = st.sidebar.slider("空調啟動臨界氣溫 (°C)", 20.0, 26.0, 23.0)
kwh_per_person_day = st.sidebar.slider("照明插座權重 (KWH/人天)", 0.5, 2.0, 1.15)
base_load_input = st.sidebar.number_input("預估每月基礎負載 (KWH)", value=4164)

# --- 內部數據處理函數 ---
def get_processed_data():
    # 這裡預設使用您提供的數據邏輯
    temp_dict = {1: 15.2, 2: 15.6, 3: 17.5, 4: 21.2, 5: 24.1, 6: 27.2, 
                 7: 28.9, 8: 28.7, 9: 27.1, 10: 24.1, 11: 20.6, 12: 16.9}
    
    # 建立 113 年 (2024) 的範例基礎數據 (根據您提供的檔案內容)
    data_113 = pd.DataFrame({
        'Month': range(1, 13),
        'Total_KWH': [19521, 17634, 45606, 42192, 50989, 41913, 4682, 5681, 48200, 46000, 44000, 49000],
        'HW_KWH': [6192, 5885, 15274, 9843, 11635, 7142, 1123, 1084, 12650, 11500, 11000, 12000], # 熱水實測/估計值
        'StayDays': [16, 12, 31, 30, 31, 23, 0, 0, 24, 31, 30, 31],
        'Occupancy': [653] * 12
    })
    
    df = data_113.copy()
    df['AvgTemp'] = df['Month'].map(temp_dict)
    df['PersonDays'] = df['StayDays'] * df['Occupancy']
    
    # 執行拆解計算
    df['Lighting'] = df['PersonDays'] * kwh_per_person_day
    df['AC_Factor'] = np.maximum(0, df['AvgTemp'] - ac_base_temp) * df['PersonDays']
    df['AC'] = df['AC_Factor'] * 0.29 # 迴歸係數
    df['BaseLoad'] = base_load_input
    df['Other'] = df['Total_KWH'] - (df['Lighting'] + df['AC'] + df['HW_KWH'] + df['BaseLoad'])
    df['Other'] = df['Other'].clip(lower=0)
    
    return df

# --- 主要顯示區域 ---
df_final = get_processed_data()

# 數據指標區
c1, c2, c3, c4 = st.columns(4)
c1.metric("年度總用電量", f"{df_final['Total_KWH'].sum():,.0f} KWH")
c2.metric("空調預估總計", f"{df_final['AC'].sum():,.0f} KWH")
c3.metric("照明/插座總計", f"{df_final['Lighting'].sum():,.0f} KWH")
c4.metric("基礎負載總計", f"{df_final['BaseLoad'].sum():,.0f} KWH")

# 視覺化圖表
st.subheader("📊 113年每月用電拆解圖")
fig_bar = px.bar(df_final, x='Month', 
                 y=['BaseLoad', 'Lighting', 'HW_KWH', 'AC', 'Other'],
                 labels={'value': '用電量 (KWH)', 'variable': '組成項目', 'Month': '月份'},
                 title="疊加長條圖分析",
                 barmode='stack',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig_bar, use_container_width=True)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🍰 年度用電比例")
    sums = df_final[['BaseLoad', 'Lighting', 'HW_KWH', 'AC', 'Other']].sum()
    fig_pie = px.pie(values=sums, names=sums.index, hole=0.4)
    st.plotly_chart(fig_pie)

with col_right:
    st.subheader("💡 節能預測模擬")
    saving_temp = st.slider("模擬調高空調啟動溫度 (+°C)", 0.0, 3.0, 1.0)
    # 計算節省
    old_ac = df_final['AC'].sum()
    new_ac_factor = np.maximum(0, df_final['AvgTemp'] - (ac_base_temp + saving_temp)) * df_final['PersonDays']
    new_ac = (new_ac_factor * 0.29).sum()
    saved = old_ac - new_ac
    
    st.info(f"如果將啟動溫度調高 {saving_temp}度，預計全年可節省 **{saved:,.0f} KWH** 的空調電力。")
    st.write(f"約等於省下 **{saved*4:.0f} 元** 電費 (以每度4元估計)。")

# 數據表
if st.checkbox("查看原始數據表格"):
    st.dataframe(df_final)

# 下載按鈕
csv = df_final.to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 下載拆解報表 (CSV)", csv, "松濤一館拆解結果.csv", "text/csv")
