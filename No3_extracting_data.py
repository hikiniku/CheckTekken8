import cv2
import pytesseract
from pytesseract import Output
import csv
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
import os

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import pandas as pd
import numpy as np
import numpy.core.multiarray

import warnings
warnings.simplefilter('ignore')

import PySimpleGUI as sg


#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# モデルの読み込み
model = torchvision.models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 9)
model.load_state_dict(torch.load("arrow_classifier.pth", torch.device('cpu')))
model = model.to(device)

model2 = torchvision.models.resnet18()
num_ftrs2 = model.fc.in_features
model2.fc = torch.nn.Linear(num_ftrs2, 16)
model2.load_state_dict(torch.load("button_classifier.pth", torch.device('cpu')))
model2 = model2.to(device)


def num_to_string(num_str):
    # 数字に対応する文字列の辞書
    num_to_str_dict = {
        "0": "ALL",
        "1": "",
        "2": "LK",
        "3": "LP",
        "4": "LPLK",
        "5": "LPRK",
        "6": "LPWK",
        "7": "RK",
        "8": "RP",
        "9": "RPLK",
        "10": "RPRK",
        "11": "RPWK",
        "12": "WK",
        "13": "WP",
        "14": "WPLK",
        "15": "WPRK",
    }

    return num_to_str_dict.get(num_str, "エラー")

def detecting_arrow(device, transform, model, img):
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 予測
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        #print(f"Predicted label: {pred.item()}")

    return str(pred.item())


def process_frame(args):
    frame_number, frame, P1_TotalDamage, P1_MaxComboDamage, P1_AttackPat, P1_AttackDamage, P1_AttackStartupFrame, P1_BlockStun, P1_Status, P1P2_Distance, P2_TotalDamage, P2_MaxComboDamage, P2_AttackPat, P2_AttackDamage, P2_AttackStartupFrame, P2_Status, P1_10key, P1_Button, P2_10key, P2_Button, P1_RecoverableDamage, P2_RecoverableDamage, tesseract_cmd = args

    # 指定した四角形領域を切り出し
    cropped_frame01 = frame[P1_TotalDamage[1]:P1_TotalDamage[3], P1_TotalDamage[0]:P1_TotalDamage[2]]
    cropped_frame02 = frame[P1_MaxComboDamage[1]:P1_MaxComboDamage[3], P1_MaxComboDamage[0]:P1_MaxComboDamage[2]]
    cropped_frame03 = frame[P1_AttackPat[1]:P1_AttackPat[3], P1_AttackPat[0]:P1_AttackPat[2]]
    cropped_frame04 = frame[P1_AttackDamage[1]:P1_AttackDamage[3], P1_AttackDamage[0]:P1_AttackDamage[2]]
    cropped_frame05 = frame[P1_AttackStartupFrame[1]:P1_AttackStartupFrame[3], P1_AttackStartupFrame[0]:P1_AttackStartupFrame[2]]
    cropped_frame06 = frame[P1_BlockStun[1]:P1_BlockStun[3], P1_BlockStun[0]:P1_BlockStun[2]]
    cropped_frame07 = frame[P1_Status[1]:P1_Status[3], P1_Status[0]:P1_Status[2]]

    cropped_frame08 = frame[P1P2_Distance[1]:P1P2_Distance[3], P1P2_Distance[0]:P1P2_Distance[2]]

    cropped_frame09 = frame[P2_TotalDamage[1]:P2_TotalDamage[3], P2_TotalDamage[0]:P2_TotalDamage[2]]
    cropped_frame10 = frame[P2_MaxComboDamage[1]:P2_MaxComboDamage[3], P2_MaxComboDamage[0]:P2_MaxComboDamage[2]]
    cropped_frame11 = frame[P2_AttackPat[1]:P2_AttackPat[3], P2_AttackPat[0]:P2_AttackPat[2]]
    cropped_frame12 = frame[P2_AttackDamage[1]:P2_AttackDamage[3], P2_AttackDamage[0]:P2_AttackDamage[2]]
    cropped_frame13 = frame[P2_AttackStartupFrame[1]:P2_AttackStartupFrame[3], P2_AttackStartupFrame[0]:P2_AttackStartupFrame[2]]
    cropped_frame14 = frame[P2_Status[1]:P2_Status[3], P2_Status[0]:P2_Status[2]]

    #cropped_frame15 = frame[P1_Frames[1]:P1_Frames[3], P1_Frames[0]:P1_Frames[2]] #割愛
    #cropped_frame16 = frame[P2_Frames[1]:P2_Frames[3], P2_Frames[0]:P2_Frames[2]] #割愛

    cropped_frame17 = frame[P1_10key[1]:P1_10key[3], P1_10key[0]:P1_10key[2]]
    cropped_frame18 = frame[P1_Button[1]:P1_Button[3], P1_Button[0]:P1_Button[2]]
    cropped_frame19 = frame[P2_10key[1]:P2_10key[3], P2_10key[0]:P2_10key[2]]
    cropped_frame20 = frame[P2_Button[1]:P2_Button[3], P2_Button[0]:P2_Button[2]]

    cropped_frame21 = frame[P1_RecoverableDamage[1]:P1_RecoverableDamage[3], P1_RecoverableDamage[0]:P1_RecoverableDamage[2]]
    cropped_frame22 = frame[P2_RecoverableDamage[1]:P2_RecoverableDamage[3], P2_RecoverableDamage[0]:P2_RecoverableDamage[2]]


    # OCRでテキスト認識
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    res01 = pytesseract.image_to_string(cropped_frame01, lang='eng', output_type=Output.STRING)
    res02 = pytesseract.image_to_string(cropped_frame02, lang='eng', output_type=Output.STRING)
    res03 = pytesseract.image_to_string(cropped_frame03, lang='jpn', output_type=Output.STRING)
    res04 = pytesseract.image_to_string(cropped_frame04, lang='eng', output_type=Output.STRING)
    res05 = pytesseract.image_to_string(cropped_frame05, lang='eng', output_type=Output.STRING)
    res06 = pytesseract.image_to_string(cropped_frame06, lang='eng', output_type=Output.STRING)
    res07 = pytesseract.image_to_string(cropped_frame07, lang='jpn', output_type=Output.STRING)

    res08 = pytesseract.image_to_string(cropped_frame08, lang='eng', output_type=Output.STRING)

    res09 = pytesseract.image_to_string(cropped_frame09, lang='eng', output_type=Output.STRING)
    res10 = pytesseract.image_to_string(cropped_frame10, lang='eng', output_type=Output.STRING)
    res11 = pytesseract.image_to_string(cropped_frame11, lang='jpn', output_type=Output.STRING)
    res12 = pytesseract.image_to_string(cropped_frame12, lang='eng', output_type=Output.STRING)
    res13 = pytesseract.image_to_string(cropped_frame13, lang='eng', output_type=Output.STRING)
    res14 = pytesseract.image_to_string(cropped_frame14, lang='jpn', output_type=Output.STRING)

    #res15 = pytesseract.image_to_string(cropped_frame15, lang='eng', output_type=Output.STRING) #割愛
    #res16 = pytesseract.image_to_string(cropped_frame16, lang='eng', output_type=Output.STRING) #割愛

    res21 = pytesseract.image_to_string(cropped_frame21, lang='eng', output_type=Output.STRING)
    res22 = pytesseract.image_to_string(cropped_frame22, lang='eng', output_type=Output.STRING)


    # 画像識別で入力コマンド認識
    model.eval()
    model2.eval()
    res17 = detecting_arrow(device, transform, model, cropped_frame17)
    res18 = detecting_arrow(device, transform, model2, cropped_frame18)
    res19 = detecting_arrow(device, transform, model, cropped_frame19)
    res20 = detecting_arrow(device, transform, model2, cropped_frame20)
    # ボタン入力の識別結果を数値から文字列に変換
    res18 = num_to_string(res18)
    res20 = num_to_string(res20)

    return frame_number, res01.strip(), res02.strip(), res03.strip(), res04.strip(), res05.strip(), res06.strip(), res07.strip(), res08.strip(), res09.strip(), res10.strip(), res11.strip(), res12.strip(), res13.strip(), res14.strip(), res17.strip(), res18.strip(), res19.strip(), res20.strip(), res21.strip(), res22.strip()

def process_all_clips_in_folder(folder_path, num_processes, n_split, tesseract_cmd):
    # folder_path 内の 'clip' で始まるすべての動画ファイルをリストアップ
    video_files = [f for f in os.listdir(folder_path) if f.startswith('clip') and f.endswith('.mp4')]

    for video_file in video_files:
        input_video_path = os.path.join(folder_path, video_file)
        output_csv_path = os.path.join(folder_path, 'detect_results_' + os.path.splitext(video_file)[0] + '.csv')
        process_video(input_video_path, output_csv_path, num_processes, n_split, tesseract_cmd)

def process_video(input_video_path, output_csv_path, num_processes, n_split, tesseract_cmd):

    # 四角形の左上と右下の頂点座標
    totaldamage = (540, 1325, 818, 1375) #使わない
    P1_TotalDamage = (1280, 1325, 1395, 1375)
    P1_MaxComboDamage = (1280, 1393, 1395, 1445)
    P1_AttackPat = (1294, 1448, 1395, 1502)
    P1_AttackDamage = (1200, 1520, 1395, 1570)
    P1_AttackStartupFrame = (1155, 1720, 1395, 1795)
    P1_BlockStun = (1155, 1800, 1395, 1855)
    P1_Status = (1000, 1870, 1395, 1920)

    P1P2_Distance = (1200, 1927, 1395, 1978)

    P2_TotalDamage = (3215, 1325, 3330, 1375)
    P2_MaxComboDamage = (3215, 1393, 3330, 1445)
    P2_AttackPat = (3218, 1448, 3330, 1502)
    P2_AttackDamage = (3135, 1520, 3330, 1570)
    P2_AttackStartupFrame = (3090, 1740, 3330, 1795)
    P2_Status = (2935, 1870, 3330, 1920)

    # 精度悪いし後処理で計算できるので削除
    #P1_Frames = (385, 465, 445, 525)
    #P2_Frames = (3680, 465, 3740, 525)

    P1_10key = (265, 465, 325, 525)
    P1_Button = (325, 465, 385, 525)
    P2_10key = (3543, 465, 3603, 525)
    P2_Button = (3603, 465, 3663, 525)

    P1_RecoverableDamage = (1200, 1580, 1395, 1630)
    P2_RecoverableDamage = (3135, 1580, 3330, 1630)


    # VideoCaptureオブジェクトの作成
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_segment = total_frames // n_split


    # CSVファイルへの書き込み準備
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "P1_TotalDamage", "P1_MaxComboDamage", "P1_AttackPat", "P1_AttackDamage", "P1_AttackStartupFrame", "P1_BlockStun", "P1_Status", "P1P2_Distance", "P2_TotalDamage", "P2_MaxComboDamage", "P2_AttackPat", "P2_AttackDamage", "P2_AttackStartupFrame", "P2_Status", "P1_10key", "P1_Button", "P2_10key", "P2_Button", "P1_RecoverableDamage", "P2_RecoverableDamage"])

        for segment in range(n_split):
            start_frame = segment * frames_per_segment
            end_frame = (segment + 1) * frames_per_segment if segment < n_split - 1 else total_frames

            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)


            # フレーム処理
            pool = Pool(processes=num_processes)
            args = [(i + start_frame, frames[i], P1_TotalDamage, P1_MaxComboDamage, P1_AttackPat, P1_AttackDamage, P1_AttackStartupFrame, P1_BlockStun, P1_Status, P1P2_Distance, P2_TotalDamage, P2_MaxComboDamage, P2_AttackPat, P2_AttackDamage, P2_AttackStartupFrame, P2_Status, P1_10key, P1_Button, P2_10key, P2_Button, P1_RecoverableDamage, P2_RecoverableDamage, tesseract_cmd) for i in range(len(frames))]

            for result in tqdm(pool.imap_unordered(process_frame, args), total=len(args), desc=f"Processing segment {segment + 1}/{n_split}"):
                writer.writerow(result)


    print("後処理、データ整形処理を実行")
    # CSVファイルの読み込み
    df = pd.read_csv(output_csv_path)

    # 1. "Frame Number"で昇順にソート
    df.sort_values(by='Frame Number', ascending=True, inplace=True)

    # 2. "P1_10key"と"P2_10key"に1を足す
    df['P1_10key'] = df['P1_10key'] + 1
    df['P2_10key'] = df['P2_10key'] + 1

    # 3 & 4. "P1_TotalDamage"と"P2_TotalDamage"で数値以外を直前のレコードの値に置換、空または数値以外の最初のレコードは0に置換
    df['P1_TotalDamage'] = pd.to_numeric(df['P1_TotalDamage'], errors='coerce').fillna(method='ffill').fillna(0)
    df['P2_TotalDamage'] = pd.to_numeric(df['P2_TotalDamage'], errors='coerce').fillna(method='ffill').fillna(0)

    # "P1_Damage"と"P2_Damage"カラムを計算して追加
    df['P1_Damage'] = df['P1_TotalDamage'].diff().fillna(0)
    df['P2_Damage'] = df['P2_TotalDamage'].diff().fillna(0)

    # 5. 特定のカラムからスペースを除去
    columns_to_strip = ['P1_AttackPat', 'P1_Status', 'P2_AttackPat', 'P2_Status']
    df[columns_to_strip] = df[columns_to_strip].apply(lambda x: x.str.replace(' ', ''))

    # 6. "P2_10key_reverse"カラムを計算して追加
    reverse_mapping = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}
    df['P2_10key_reverse'] = df['P2_10key'].map(reverse_mapping)

    # 結果を新しいCSVファイルに出力
    df[['Frame Number', 'P1_10key', 'P1_Button', 'P2_10key', 'P2_Button', 'P1_Damage', 'P2_Damage', 'P2_10key_reverse', 'P1_RecoverableDamage', 'P2_RecoverableDamage', 'P1_AttackPat', 'P2_AttackPat', 'P1_Status', 'P2_Status', 'P1_TotalDamage', 'P1_MaxComboDamage', 'P1_AttackDamage', 'P1_AttackStartupFrame', 'P1_BlockStun', 'P1P2_Distance', 'P2_TotalDamage', 'P2_MaxComboDamage', 'P2_AttackDamage', 'P2_AttackStartupFrame']].to_csv(output_csv_path, index=False)


    cap.release()
    print("OCR処理が完了しました: " + input_video_path)

def main():
    sg.theme('DarkGrey13')
    layout = [
        [sg.Text('Tesseractの実行ファイル:'), sg.InputText(default_text=r'C:\Program Files\Tesseract-OCR\tesseract.exe', key='tesseract_cmd')],
        [sg.Text('スレッド数:'), sg.InputText(default_text='8', size=(5, 1), key='num_processes')],
        [sg.Text('動画の分割処理数:'), sg.InputText(default_text='4', size=(5, 1), key='n_split')],
        [sg.Button('HUDデータ抽出処理開始')]
    ]

    window = sg.Window('リプレイHUDデータ抽出', layout)

    while True:
        event, values = window.read()
    
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'HUDデータ抽出処理開始':
            num_processes = int(values['num_processes'])
            n_split = int(values['n_split'])
            tesseract_cmd = values['tesseract_cmd']
    
            # pytesseractのtesseract_cmdを設定
            print(tesseract_cmd)
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

            folder_path = 'work'  # 既定のフォルダパス

            process_all_clips_in_folder(folder_path, num_processes, n_split, tesseract_cmd)

    window.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
