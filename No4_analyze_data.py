import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#import japanize_matplotlib

import os
from io import StringIO
import math
import random
import colorsys


# 比率と数値を両方表示するカスタム関数
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
    return my_format

def load_and_process(folder_path):
    dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

    # すべてのデータフレームを結合
    combined_df = pd.concat(dfs, ignore_index=True)

    # 前処理
    combined_df['P1_TotalDamage'] = pd.to_numeric(combined_df['P1_TotalDamage'], errors='coerce')
    combined_df['P2_TotalDamage'] = pd.to_numeric(combined_df['P2_TotalDamage'], errors='coerce')
    combined_df['P1_TotalDamage'].fillna(method='ffill', inplace=True)
    combined_df['P2_TotalDamage'].fillna(method='ffill', inplace=True)
    combined_df['P1_damage'] = combined_df['P1_TotalDamage'].diff(periods=1)
    combined_df['P2_damage'] = combined_df['P2_TotalDamage'].diff(periods=1)
    combined_df['P1_AttackPat'] = combined_df['P1_AttackPat'].str.replace(' ', '', regex=False)
    combined_df['P2_AttackPat'] = combined_df['P2_AttackPat'].str.replace(' ', '', regex=False)

    return combined_df


# P1が相手にダメージを与えたフレームから指定したフレーム数だけ直前に入力したコマンドを出力
def P1_attack_history(df, prev_frames, AttackPat):
    nonzero_indices = df[(df["P1_Damage"] > 0) & (df["P1_AttackPat"] == str(AttackPat))].index
    
    list_damage_values = []
    list_history_values = []

    # prev_framesで指定した直前レコードの "P1_history" 列の値を一意にして順番を保って出力
    for i in nonzero_indices:
        history_values = df.iloc[max(0, i - prev_frames):i]["P1_history"].unique()
        history_values = [val for val in history_values if pd.notna(val)]
        #print(f"P1_Damage: {df.iloc[i]['P1_Damage']}    {df.iloc[i]['P1_Damage']}")
        #print(", ".join(history_values))

        list_damage_values.append(int(df.iloc[i]['P1_Damage']))
        list_history_values.append(history_values)

    return list_damage_values, list_history_values


# P2がダメージを喰らったフレームから指定したフレーム数だけ直前に相手が入力したコマンドを出力
def P2_attack_history(df, prev_frames, AttackPat):
    nonzero_indices = df[(df["P2_Damage"] > 0) & (df["P2_AttackPat"] == str(AttackPat))].index

    list_damage_values = []
    list_history_values = []

    # prev_framesで指定した直前レコードの "P2_history" 列の値を一意にして順番を保って出力
    for i in nonzero_indices:
        history_values = df.iloc[max(0, i - prev_frames):i]["P2_history"].unique()
        history_values = [val for val in history_values if pd.notna(val)]
        #print(f"P2_Damage: {df.iloc[i]['P2_Damage']}    {df.iloc[i]['P2_Damage']}")
        #print(", ".join(history_values))

        list_damage_values.append(int(df.iloc[i]['P2_Damage']))
        list_history_values.append(history_values)

    return list_damage_values, list_history_values



def main():

    # データを読み込み
    folder_path = 'result'
    df = load_and_process(folder_path)

    print("############ 前処理 ############")
    # ラウンドが変わるごとに大ダメージが入るように見えてしまう部分を0リセット
    mask = df["Frame Number"] == 0
    next_index = df[mask].index + 1
    df.loc[next_index, ["P1_Damage", "P2_Damage"]] = 0

    # OCR誤検知で大ダメージが入るように見える部分を0リセット
    df.loc[df["P1_Damage"] >= 108, "P1_Damage"] = 0
    df.loc[df["P2_Damage"] >= 108, "P2_Damage"] = 0

    # 入力コマンド履歴のカラム作成
    df["P1_10key"] = df["P1_10key"].fillna('')
    df["P1_Button"] = df["P1_Button"].fillna('')
    df["P1_history"] = df["P1_10key"].astype(str) + df["P1_Button"].astype(str)
    df["P2_10key"] = df["P2_10key"].fillna('')
    df["P2_Button"] = df["P2_Button"].fillna('')
    df["P2_history"] = df["P2_10key"].astype(str) + df["P2_Button"].astype(str)

    print("############ 円グラフ表示 ############")

    # 攻撃成功/ダメージ喰らったタイミングから何フレーム前まで入力コマンドを取得するかの変数
    prev_frames = 90

    print("### P1側が攻撃成功した時のデータ ####")
    df_P1_Attack = df[ df['P1_damage'] > 0 ][['Frame Number', 'P1_damage', 'P1_AttackPat', 'P1_AttackStartupFrame', 'P1_Status', 'P1_BlockStun']]
    print(df_P1_Attack)

    grouped_sum = df_P1_Attack.groupby('P1_AttackPat')['P1_damage'].sum().reset_index()


    # OCRの読み取りに失敗した攻撃パターンを除外
    grouped_sum = grouped_sum[grouped_sum['P1_AttackPat'].isin(["上段", "中段", "下段", "投げ"])]
    grouped_sum = grouped_sum.reset_index(drop=True)


    # ダメージの合計を計算
    total_damage = grouped_sum['P1_damage'].sum()
    
    # 各攻撃パターンのダメージの割合を計算
    grouped_sum['damage_ratio'] = grouped_sum['P1_damage'] / total_damage * 100
    print(grouped_sum)

    # 色のリストを定義
    unique_attack_pats = df['P1_AttackPat'].unique()
    num_unique_attack_pats = len(unique_attack_pats)
    num_unique_attack_pats = len(unique_attack_pats)
    if num_unique_attack_pats <= 3:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 3色のカラーパレット
    elif num_unique_attack_pats <= 4:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 4色のカラーパレット
    elif num_unique_attack_pats <= 5:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5色のカラーパレット
    elif num_unique_attack_pats <= 6:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 6色のカラーパレット
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']  # 7色のカラーパレット


    # HTMLファイルを作成して出力
    with open("output_attack.html", "w", encoding="utf-8") as f:
        # HTMLファイルのヘッダー部分を書き込む
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write('<meta charset="UTF-8">\n')
        f.write('<meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        f.write("<style>\n")
    
        f.write(".pie-chart-container {\n")
        f.write("width: 600px;\n")
        f.write("height: 600px;\n")
        f.write("position: relative;\n")
        f.write("background-image: conic-gradient(\n")
        start_angle = 0
        for index, row in grouped_sum.iterrows():
            f.write(f"{colors[index]} {start_angle}%, {colors[index]} {start_angle + row['damage_ratio']}%")
            start_angle += row['damage_ratio']
            if index < len(grouped_sum) - 1:
                f.write(", ")
        f.write("\n); border-radius: 50%;\n")
    
        f.write("}\n")
    
    
        f.write(".pie-labels {\n")
        f.write("}\n")
    
        f.write(".legend-item {\n")
        f.write("position: absolute;\n")
        f.write("color: #333;\n")
        f.write("}\n")

        f.write(".data_tab{\n")
        f.write("font-size: 50%;\n")
        f.write("}\n")


        f.write("</style>\n</head>\n<body>\n")

        f.write("<h2 id='#top'>P1が相手に与えたダメージの種類<h2>\n")

        f.write("<div class='pie-chart-container'>\n")
        f.write("<div class='pie-labels'>\n")
    
        # 円グラフの中心座標
        center_x = 300
        center_y = 300
        
        # 各凡例を追加
        start_ratio = 0
        angle = 0
        for index, row in grouped_sum.iterrows():
            angle = (3.6 * start_ratio) + (3.6 * row['damage_ratio']) / 2
            radius = 200  # 円グラフの半径
            
            # 凡例の位置を計算
            legend_x = center_x + radius * 0.8 * math.cos(math.radians(angle - 90))
            legend_y = center_y + radius * 0.8 * math.sin(math.radians(angle - 90))
            
            #f.write(f"<div class='legend-item' style='left: {legend_x}px; top: {legend_y}px;'>{row['P1_AttackPat']}</div>\n")
            f.write(f"<div class='legend-item' style='left: {legend_x}px; top: {legend_y}px;'><a href='#{row['P1_AttackPat']}'>{row['P1_AttackPat']}</a></div>\n")
            start_ratio += row['damage_ratio']

        f.write("</div>\n")
        f.write("</div>\n")

        for index, row in grouped_sum.iterrows():
            f.write(f"<div id='{row['P1_AttackPat']}'>\n")
            f.write(f"<h3>{row['P1_AttackPat']}の内容<h3>\n")

            list_damage_values, list_history_values = P1_attack_history(df, prev_frames, row['P1_AttackPat'])
            f.write(f"\n")
            f.write(f"<table border class='data_tab'>\n")
            f.write(f"   <tr><th>与えたダメージ</th><th>直前{str(prev_frames)}フレームで入力したコマンド</th></tr>\n")
            for i in range(len(list_history_values)):
                f.write(f"   <tr><td>{str(list_damage_values[i])}</td><td>{str(list_history_values[i]).replace('[','').replace(']','').replace('5','')}</td></tr>\n")
            #f.write(f"\n")

            f.write(f"</table>\n")
            f.write(f"<a border class='data_tab' href='#top'>トップに戻る</a>\n")

        f.write("</body>\n</html>")



    print("### P2側が攻撃成功した時のデータ ####")
    df_P2_Attack = df[ df['P2_damage'] > 0 ][['Frame Number', 'P2_damage', 'P2_AttackPat', 'P2_AttackStartupFrame', 'P2_Status', 'P1_BlockStun']]
    print(df_P2_Attack)

    grouped_sum = df_P2_Attack.groupby('P2_AttackPat')['P2_damage'].sum().reset_index()

    # OCRの読み取りに失敗した攻撃パターンを除外
    grouped_sum = grouped_sum[grouped_sum['P2_AttackPat'].isin(["上段", "中段", "下段", "投げ"])]
    grouped_sum = grouped_sum.reset_index(drop=True)


    # ダメージの合計を計算
    total_damage = grouped_sum['P2_damage'].sum()
    
    # 各攻撃パターンのダメージの割合を計算
    grouped_sum['damage_ratio'] = grouped_sum['P2_damage'] / total_damage * 100
    print(grouped_sum)

    # 色のリストを定義
    unique_attack_pats = df['P2_AttackPat'].unique()
    num_unique_attack_pats = len(unique_attack_pats)
    num_unique_attack_pats = len(unique_attack_pats)
    if num_unique_attack_pats <= 3:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 3色のカラーパレット
    elif num_unique_attack_pats <= 4:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 4色のカラーパレット
    elif num_unique_attack_pats <= 5:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5色のカラーパレット
    elif num_unique_attack_pats <= 6:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 6色のカラーパレット
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']  # 7色のカラーパレット


    # HTMLファイルを作成して出力
    with open("output_damage.html", "w", encoding="utf-8") as f:
        # HTMLファイルのヘッダー部分を書き込む
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write('<meta charset="UTF-8">\n')
        f.write('<meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        f.write("<style>\n")
    
        f.write(".pie-chart-container {\n")
        f.write("width: 600px;\n")
        f.write("height: 600px;\n")
        f.write("position: relative;\n")
        f.write("background-image: conic-gradient(\n")
        start_angle = 0
        for index, row in grouped_sum.iterrows():
            f.write(f"{colors[index]} {start_angle}%, {colors[index]} {start_angle + row['damage_ratio']}%")
            start_angle += row['damage_ratio']
            if index < len(grouped_sum) - 1:
                f.write(", ")
        f.write("\n); border-radius: 50%;\n")
    
        f.write("}\n")
    
    
        f.write(".pie-labels {\n")
        f.write("}\n")
    
        f.write(".legend-item {\n")
        f.write("position: absolute;\n")
        f.write("color: #333;\n")
        f.write("}\n")

        f.write(".data_tab{\n")
        f.write("font-size: 50%;\n")
        f.write("}\n")


        f.write("</style>\n</head>\n<body>\n")

        f.write("<h2 id='#top'>P1が喰らったダメージの種類<h2>\n")

        f.write("<div class='pie-chart-container'>\n")
        f.write("<div class='pie-labels'>\n")
    
        # 円グラフの中心座標
        center_x = 300
        center_y = 300
        
        # 各凡例を追加
        start_ratio = 0
        angle = 0
        for index, row in grouped_sum.iterrows():
            angle = (3.6 * start_ratio) + (3.6 * row['damage_ratio']) / 2
            radius = 200  # 円グラフの半径
            
            # 凡例の位置を計算
            legend_x = center_x + radius * 0.8 * math.cos(math.radians(angle - 90))
            legend_y = center_y + radius * 0.8 * math.sin(math.radians(angle - 90))
            
            #f.write(f"<div class='legend-item' style='left: {legend_x}px; top: {legend_y}px;'>{row['P2_AttackPat']}</div>\n")
            f.write(f"<div class='legend-item' style='left: {legend_x}px; top: {legend_y}px;'><a href='#{row['P2_AttackPat']}'>{row['P2_AttackPat']}</a></div>\n")
            start_ratio += row['damage_ratio']

        f.write("</div>\n")
        f.write("</div>\n")

        for index, row in grouped_sum.iterrows():
            f.write(f"<div id='{row['P2_AttackPat']}'>\n")
            f.write(f"<h3>{row['P2_AttackPat']}の内容<h3>\n")

            list_damage_values, list_history_values = P2_attack_history(df, prev_frames, row['P2_AttackPat'])
            f.write(f"\n")
            f.write(f"<table border class='data_tab'>\n")
            f.write(f"   <tr><th>被ダメージ</th><th>相手が直前{str(prev_frames)}フレームで入力したコマンド</th></tr>\n")
            for i in range(len(list_history_values)):
                f.write(f"   <tr><td>{str(list_damage_values[i])}</td><td>{str(list_history_values[i]).replace('[','').replace(']','').replace('5','')}</td></tr>\n")
            #f.write(f"\n")

            f.write(f"</table>\n")
            f.write(f"<a border class='data_tab' href='#top'>トップに戻る</a>\n")

        f.write("</body>\n</html>")


if __name__ == "__main__":
    main() 
