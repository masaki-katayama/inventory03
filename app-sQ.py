#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import streamlit as st
import inventorize as inv
from PIL import Image


# In[2]:


image1 = Image.open('logigeek-logo-short.png')
st.set_page_config(
    page_title="INV_Simulation03|LogiGeek", 
    page_icon=image1,
    layout="wide")

image2 = Image.open('logigeek-logo-long.png')
st.image(image2, caption='ロジスティクスをDXするための小ネタ集')
st.link_button(":blue[ロジギークのHPへ]", 
               "https://rikei-logistics.com/",
                use_container_width = True)

st.header(':blue[定量発注] & :blue[Min-Max発注]のシミュレーションアプリ')
st.text('')
st.subheader('このアプリでできること')
st.text('１．同じ需要データを用いて、定量発注方式、及びMin-Max発注方式のシミュレーションを行います。')
st.text('２．在庫推移の他、トータル物流コストの比較ができます。')
st.text('３．定量発注における発注数、及びMin-Max発注におけるMaxは１日の平均需要の倍数で設定できます。')
st.text('４．需要データはcsvファイルでアップロードできます。')
st.text('詳細な使い方については下記サイトをご覧下さい↓')
st.link_button(":blue[定量発注方式とMin-Max発注方式を同じ需要データで比較できるシミュレーションアプリ|ロジギーク]", 
               "https://rikei-logistics.com/app-sq")
st.text('')


# In[3]:


st.sidebar.header('◆条件設定画面◆')
st.sidebar.subheader('１．需要データの読み込み')
uploaded_file = st.sidebar.file_uploader('csvファイルをアップロードして下さい。',type='csv')

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_csv('default_data.csv')

st.sidebar.subheader('２．訓練データと検証データの比率')
split_rate = st.sidebar.number_input(label = '検証データ（％）', 
                                     value = 70, label_visibility="visible", 
                                     min_value=0, max_value=100)

st.sidebar.subheader('３．発注関連パラメータ')
ld = st.sidebar.number_input(label = '納品リードタイム（日）', 
                                     value = 3, label_visibility="visible", 
                                     min_value=0, max_value=90)
so = st.sidebar.number_input(label = '許容欠品率（％）', 
                                     value = 5, label_visibility="visible", 
                                     min_value=0, max_value=100)
xd = st.sidebar.number_input(label = '発注数を需要のｘ日分に設定:blue[（定量発注の場合）]', 
                                     value = 4, label_visibility="visible", 
                                     min_value=0, max_value=500)
mx = st.sidebar.number_input(label = 'MaxをMin＋需要のｘ日分に設定:blue[（Min-Max発注の場合）]', 
                                     value = 4, label_visibility="visible", 
                                     min_value=0, max_value=500)

st.sidebar.subheader('４．物流コスト')
ordering_cost1 = st.sidebar.number_input(label = '輸送コスト（円／発注）:blue[（定量発注の場合）]', 
                                     value = 90000, label_visibility="visible", 
                                     min_value=0, max_value=1000000)
ordering_cost2 = st.sidebar.number_input(label = '輸送コスト（円／発注）:blue[（Min-Max発注の場合）]', 
                                     value = 100000, label_visibility="visible", 
                                     min_value=0, max_value=1000000)
inventory_cost = st.sidebar.number_input(label = '保管コスト（円／個･日）', 
                                     value = 3, label_visibility="visible", 
                                     min_value=0, max_value=1000)
shortage_cost = st.sidebar.number_input(label = '欠品コスト（円／個）', 
                                     value = 1000, label_visibility="visible", 
                                     min_value=0, max_value=100000)
st.subheader('需要データ')
st.write('データ数　：', f'{len(raw_df)}個')
st.dataframe(raw_df)

raw_df = np.array(raw_df)

x = range(1, len(raw_df)+1)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(x, raw_df)
ax.set_xlabel('Days', weight ='bold', size = 14, color ='black')
ax.set_ylabel('Demand', weight ='bold', size = 14, color ='black')
st.pyplot(fig)

learn_df, test_df = train_test_split(raw_df, test_size=split_rate/100, shuffle=False)
    
av = learn_df.mean()
sd = learn_df.std(ddof = 1)

# sQ
result = inv.sim_min_Q_normal(
    test_df,
    av,
    sd,
    ld,
    1-so/100,
    xd * av,
    shortage_cost = shortage_cost,
    inventory_cost = inventory_cost,
    ordering_cost = ordering_cost1
)

sf_stock = int(result[1]['saftey_stock'])
av_stock = int(result[1]['average_inventory_level'])
o_point = int(result[0].iloc[1, 5])
ts_cost = int(result[1]['ordering_cost'])
st_cost = int(result[1]['inventory_cost'])
so_cost = int(result[1]['shortage_cost'])
fill_rate = result[1]['Item_fill_rate'] * 100
service_rate = result[1]['cycle_service_level'] * 100
o_qty = int(xd * av)

sf_stock_c = f'{sf_stock:,}個'
av_stock_c = f'{av_stock:,}個'
o_point_c = f'{o_point:,}個'
o_qty_c = f'{o_qty:,}個'
ts_cost_c = f'{ts_cost:,}円'
st_cost_c = f'{st_cost:,}円'
so_cost_c = f'{so_cost:,}円'
fill_rate_c = f'{fill_rate:.1f}％'
service_rate_c = f'{service_rate:.1f}％'

st.text('')
st.subheader(':blue[定量発注方式]のシミュレーション結果', divider='rainbow')
st.subheader(':mag: 入出庫推移表')
show_df = result[0].rename(columns={'period': '日', 'demand': '需要', 'sales': '出荷', 'inventory_level': '庫内在庫',
                                   'inventory_position': 'トータル在庫', 'min': '発注点', 'order': '発注', 'recieved': '入庫',
                                    'lost_order': '欠品'})
st.dataframe(show_df.astype('int'))

st.text('')
st.subheader(':mag: 庫内在庫推移グラフ')
x = range(1, len(test_df)+2)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(x, result[0]['inventory_level'])
ax.set_xlabel('Days', weight ='bold', size = 14, color ='black')
ax.set_ylabel('Stock', weight ='bold', size = 14, color ='black')
st.pyplot(fig)

st.text('')
st.subheader(':mag: 主要在庫指標')
st.write('安全在庫 ： ', sf_stock_c)
st.write('発注点 ： ', o_point_c)
st.write('発注数 ： ', o_qty_c)
st.write('平均在庫 ： ', av_stock_c)
st.write('サービス率 ： ', service_rate_c)
st.write('充足率 ： ', fill_rate_c)

st.text('')
st.subheader(':mag: 物流コスト')
st.write('トータル輸送コスト ： ', ts_cost_c)
st.write('トータル保管コスト ： ', st_cost_c)
st.write('トータル欠品コスト ： ', so_cost_c)
st.write('トータルロジスティクスコスト ： ', f'{ts_cost + st_cost + so_cost:,}円')

# Min-Max
rop_dic = inv.reorderpoint(
  av,
  sd,
  ld,
  1-so/100
)
rop = rop_dic['reorder_point']

result = inv.sim_min_max_normal(
    test_df,
    av,
    sd,
    ld,
    1-so/100,
    rop + mx * av,
    shortage_cost = shortage_cost,
    inventory_cost = inventory_cost,
    ordering_cost = ordering_cost2
)

sf_stock2 = int(result[1]['saftey_stock'])
av_stock2 = int(result[1]['average_inventory_level'])
mmin = int(result[0].iloc[1, 5])
mmax = int(result[0].iloc[1, 7])
ts_cost2 = int(result[1]['ordering_cost'])
st_cost2 = int(result[1]['inventory_cost'])
so_cost2 = int(result[1]['shortage_cost'])
fill_rate2 = result[1]['Item_fill_rate'] * 100
service_rate2 = result[1]['cycle_service_level'] * 100

sf_stock2_c = f'{sf_stock2:,}個'
av_stock2_c = f'{av_stock2:,}個'
mmin_c = f'{mmin:,}個'
mmax_c = f'{mmax:,}個'
ts_cost2_c = f'{ts_cost2:,}円'
st_cost2_c = f'{st_cost2:,}円'
so_cost2_c = f'{so_cost2:,}円'
fill_rate2_c = f'{fill_rate2:.1f}％'
service_rate2_c = f'{service_rate2:.1f}％'

st.text('')
st.subheader(':blue[Min-Max発注方式]のシミュレーション結果', divider='rainbow')
st.subheader(':mag: 入出庫推移表')
show_df = result[0].rename(columns={'period': '日', 'demand': '需要', 'sales': '出荷', 'inventory_level': '庫内在庫',
                                   'inventory_position': 'トータル在庫', 'min': 'Min', 'order': '発注', 'max': 'Max', 'recieved': '入庫',
                                    'lost_order': '欠品'})
st.dataframe(show_df.astype('int'))

st.text('')
st.subheader(':mag: 庫内在庫推移グラフ')
x = range(1, len(test_df)+2)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(x, result[0]['inventory_level'])
ax.set_xlabel('Days', weight ='bold', size = 14, color ='black')
ax.set_ylabel('Stock', weight ='bold', size = 14, color ='black')
st.pyplot(fig)

st.text('')
st.subheader(':mag: 主要在庫指標')
st.write('安全在庫 ： ', sf_stock2_c)
st.write('MIn ： ', mmin_c)
st.write('Max ： ', mmax_c)
st.write('平均在庫 ： ', av_stock2_c)
st.write('サービス率 ： ', service_rate2_c)
st.write('充足率 ： ', fill_rate2_c)

st.text('')
st.subheader(':mag: 物流コスト')
st.write('トータル輸送コスト ： ', ts_cost2_c)
st.write('トータル保管コスト ： ', st_cost2_c)
st.write('トータル欠品コスト ： ', so_cost2_c)
st.write('トータルロジスティクスコスト ： ', f'{ts_cost2 + st_cost2 + so_cost2:,}円')

