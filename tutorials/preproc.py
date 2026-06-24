import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime, timedelta

notebook_dir = os.getcwd()
parent_dir = os.path.join(notebook_dir, '..')
sys.path.append(parent_dir)
from aether.utils_data import extract_data, extract_apnee, normalize_signal

csv_files_tot =  ['AA20211218_214726_0_HRD.edf-with-date.csv', 'AA20220219_235834_0_HRD.edf-with-date.csv', 
    'AA20211220_224125_0_HRD.edf-with-date.csv', 'AA20210921_214319_0_HRD.edf-with-date.csv', 
    'AA20210622_210513_0_HRD.edf-with-date.csv', 
    '20220913_064933_0_HRD.edf-with-date.csv',  
    '2.edf-with-date.csv', 
    '3.edf-with-date.csv', '4.edf-with-date.csv', 
    '5.edf-with-date.csv', '6.edf-with-date.csv', 
    '7.edf-with-date.csv']

file_listapnee_tot = [['18 dec AA.csv', '18 dec SG.csv'], ['19 feb 22 AA.csv', '19 feb 22 SG.csv'], 
    ['20 dec 21 AA.csv', '20 dec 21 SG.csv'], ['21 sept 21 AA.csv', '21 sept 21 SG.csv'], 
    ['22 jun 21 AA.xlsx', '22 jun 21 SG.xlsx'] , 
    ['file_marcato_test.xlsx'], 
    ['secondo.csv'], ['terzo.csv'], ['quarto.csv'],
    ['quinto.csv'], ['sesto.csv'], 
    ['settimo.csv']]


csv_files_eval = [ '20220913_064933_0_HRD.edf-with-date.csv' ]
file_listapnee_eval = [ ['file_marcato_test.xlsx']]


def extract_apnee1(file_apnee, folder_path):
    """
    Fubnction extracting apnea events from a scoring file.

    :param list[str] file_apnee: list containing the path to the scoring files
    :param str folder_path: path to the folder with the files under consideration
    :return: list containing apnea events, i.e. (start, end) of each apnea event
    :rtype: list
    """
    apnea_events = []
    len_events = []

    for file_path in file_apnee:
        filepath = os.path.join(folder_path, file_path)
        print(f"Processing file: {filepath}")

        try:
            if filepath.endswith('.csv'):
                # Legge file CSV, gestendo separatori comuni (;, e virgola)
                apnea_file = pd.read_csv(filepath, sep='[;,]', engine='python')
            elif filepath.endswith(('.xls', '.xlsx')):
                apnea_file = pd.read_excel(filepath)
            else:
                print(f"Skipping unsupported file type: {filepath}")
                continue
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            continue

        # Remove blanck spaces from column names
        apnea_file.columns = apnea_file.columns.str.strip()
        
        # In our files there are two different way of storing data: using start_time
        # and end_time, or start_time and duration related to the apnea event
        if 'Start_time' in apnea_file.columns and 'End_time' in apnea_file.columns:
            print("Detected format: Start_time / End_time (Format 1)")
            
            for _, row in apnea_file.iterrows():
                try:
                    start_str = str(row['Start_time'])
                    end_str = str(row['End_time'])
                    
                    # expected data in format "AAAA-MM-GG HH:MM:SS.fff+00:00"
                    start_str = start_str.split(" ")[1].split("+")[0]
                    end_str = end_str.split(" ")[1].split("+")[0]

                    time_format = "%H:%M:%S.%f" if "." in start_str else "%H:%M:%S"
                    start_time = datetime.strptime(start_str, time_format)
                    
                    time_format = "%H:%M:%S.%f" if "." in end_str else "%H:%M:%S"
                    end_time = datetime.strptime(end_str, time_format)

                    len_events.append(end_time - start_time)

                    apnea_events.append((start_time.strftime("%H:%M:%S.%f"), end_time.strftime("%H:%M:%S.%f")))
                except Exception as e:
                    print(f"Skipping row due to parsing error (Format 1): {e}")
                    
        elif 'Time' in apnea_file.columns and 'Duration' in apnea_file.columns:
            print("Detected format: Time / Duration (Format 2)")

            # clean column data and conversion as string
            apnea_file['Time'] = apnea_file['Time'].astype(str).str.strip()
            apnea_file['Duration'] = apnea_file['Duration'].astype(str).str.strip()

            for _, row in apnea_file.iterrows():
                try:
                    start_time_str = row['Time']
                    duration_str = row['Duration']
                    
                    time_format = "%H:%M:%S.%f" if "." in start_time_str else "%H:%M:%S"
                    start_time = datetime.strptime(start_time_str, time_format)

                    duration = timedelta(seconds=float(duration_str))
                    len_events.append(duration)

                    end_time = start_time + duration
                    
                    apnea_events.append((start_time.strftime("%H:%M:%S.%f"), end_time.strftime("%H:%M:%S.%f")))
                except Exception as e:
                    print(f"Skipping row due to parsing error (Format 2): {e}")
        
        else:
            print(f"Skipping file {filepath}: Required columns ('Start_time'/'End_time' or 'Time'/'Duration') not found.")
    if len(len_events) > 0:
        print(min(len_events))
    return apnea_events

from datetime import datetime

def create_windows(signal, labels, fs, t_window, overlap):
    points = int(fs * t_window)
    step = int(points * (1 - overlap))

    windows_signal = []
    windows_label = []
    labels_1 = []
    for i in range(0, len(signal) - points, step):
        # window extraction
        window_i = signal[i : i + points]
        label_i = labels[i : i + points]
        # window labelled positive (1) if 50% of them is 1
        label_window = 1 if np.mean(label_i) >= 0.5 else 0

        label1 = 1 if np.any(label_i != 0) else 0
        labels_1.append(label1)

        windows_signal.append(window_i)
        windows_label.append(label_window)

    return windows_signal, windows_label, labels_1

def create_labels(timestamps, events):
    labels = np.zeros(len(timestamps))
    timestamps = pd.to_datetime(pd.Series(timestamps).astype(str), format='mixed')

    for start, end in events:
        start = pd.to_datetime(start, format='mixed')
        end = pd.to_datetime(end, format='mixed')
        mask_apnea = (timestamps >= start) & (timestamps <= end)

        labels[mask_apnea] = 1

    return labels

def prep_data(csv_files,file_listapnee, test=False):
    '''
    Function extracting and pre-processing the data of interest.

    :param list[str] csv_files: list containing the path to the data files
    :param list[str] file_listapnee: list containting the path to the scoring
           files
    :param bool test: If True, the file under consideration is for the evaluation
           phase. If False, for the training/testing phase.
    :return:
    :rtype: 
    '''

    X = []
    y = []
    y1 = []
    pazienti_tot = []

    id_patients = []
    times_tot = []
    id_times = []
    resp_int = []

    for i, file in enumerate(csv_files):
        
        file_name =file
        file = os.path.join('../aether/tracciati_validazione/',  file)
        print('File:', file)
        file = pd.read_csv(file)
        
        start = 30675
        end = len(file) - 2323

        mask_press, resp_flow, timestamps = extract_data(file, start, end)

        resp_flow = [float(x) for x in resp_flow]
        mask_press = [float(x) for x in mask_press]

        file_apnee = file_listapnee[i]
        path_apnee = '../aether/Riscorati/'

        apnee_events = extract_apnee(file_apnee, path_apnee)
        print('Number apnee', len(apnee_events))

        resp_flow_scaled = normalize_signal(resp_flow)
        mask_press_scaled = normalize_signal(mask_press)

        signal_tot = np.column_stack((resp_flow_scaled, mask_press_scaled))
        print(signal_tot.shape)

        labels_tot = create_labels(timestamps, apnee_events)
        #df = pd.DataFrame({'time': timestamps, 'labels': labels_tot})
        #df.to_csv('data_new.csv', index=False)

        # windowing: 5 seconds, 50% overlap, fs = 24Hz
        signal_seg, label_seg, labels1= create_windows(signal_tot, labels_tot, fs=24, t_window=9, overlap=0.5)

        # --- NUOVO: Creiamo l'array di ID per questo specifico paziente ---
        # Assegna l'indice numerico 'i' a ciascuna finestra di questo file
        id_paziente_seg = [i] * len(signal_seg)
        pazienti_tot.append(id_paziente_seg)


        # id_patient = [i] * len(resp_events)
        # print('data_resp', len(resp_data))
        print((len(signal_seg),len(signal_seg[0][1]), len(signal_seg[1])))

        X.append(signal_seg)

        y.append(label_seg)
        y1.append(labels1)

        print((len(X),len(X[0])))


        # id_patients.extend(id_patient)

        print('N. respiratory windows', len(signal_seg))
        # print('N. respiratory data', len(resp_data))
        # print('N. labels ID', len(id_patient))
    print('N. respiratory data', len(X))
    print('N. labels', len(y))

    X = np.concatenate(X, axis=0) # [n_windows, points, n_channels] point=len(window)
    #print('xx', X.shape)
    y = np.concatenate(y, axis=0) # [n_windows]
    #print(y.shape)
    y1 = np.concatenate(y1, axis=0)
    pazienti = np.concatenate(pazienti_tot, axis=0)

    return X,  np.array(y), np.array(y1), signal_tot, pazienti

X,y,y1,  signal_tot, pazienti =prep_data(csv_files_tot,file_listapnee_tot,test=False)
np.save('data_window_9s.npy', X)
np.save('label_window_9s.npy', y)
np.save('label_window_1_9s.npy', y1)
np.save('pazienti_id_9s.npy', pazienti)


X,y,y1,  signal_tot, pazienti =prep_data(csv_files_eval,file_listapnee_eval,test=True)
np.save('data_window_eval9.npy', X)
np.save('label_window_eval9.npy', y)
np.save('label_window_1_eval9.npy', y1)
np.save('pazienti_id_eval_9s.npy', pazienti)
