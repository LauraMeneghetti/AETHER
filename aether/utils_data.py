'''Module for utility functions'''

import numpy as np
import torch
from collections import Counter
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from scipy.ndimage import label
from datetime import datetime, timedelta
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scipy.fft import fft
from scipy.integrate import simpson
import pickle
import matplotlib.pyplot as plt


def extract_data(data_time, start, end):
    '''
    Function extracting useful data from the data file provided.
    :param pd.DataFrame file: input data file
    :param int: starting point
    :param int: ending point 
    :return: list containing data of interest, i.e. mask pressure, 
             respiratory airflow and times
    :rtype: list, list, list
    '''

    mask_press = []
    for i in range(start, end):
        data = data_time.loc[i, 'MASK_PRESS [hPa]']
        mask_press.append(data)

    resp_flow = []
    for i in range(start, end):
        data = data_time.loc[i, 'RESP_FLOW [?]']
        resp_flow.append(data)

    datastamps = []
    for i in range(start, end):
        data = data_time.loc[i, 'DATE_TIME_STAMP']
        datastamps.append(data)

    timestamps = []
    for datatime_str in datastamps:
        time_str = datatime_str.split(" ")[1]
        time_str = time_str.split("+")[0]
        if "." in time_str:
            time_str = datetime.strptime(time_str, "%H:%M:%S.%f").time()
        else:
            time_str = datetime.strptime(time_str, "%H:%M:%S").time()
        timestamps.append(time_str)


    return mask_press, resp_flow, timestamps



def extract_apnee(file_apnee, folder_path):
    """
    Fubnction extracting apnea events from a scoring file.

    :param list[str] file_apnee: list containing the path to the scoring files
    :param str folder_path: path to the folder with the files under consideration
    :return: list containing apnea events, i.e. (start, end) of each apnea event
    :rtype: list
    """
    apnea_events = []

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

                    end_time = start_time + duration
                    
                    apnea_events.append((start_time.strftime("%H:%M:%S.%f"), end_time.strftime("%H:%M:%S.%f")))
                except Exception as e:
                    print(f"Skipping row due to parsing error (Format 2): {e}")
        
        else:
            print(f"Skipping file {filepath}: Required columns ('Start_time'/'End_time' or 'Time'/'Duration') not found.")

    return apnea_events



def freq_val(lst):
    '''
    Function evaluating the most common value in a signal, representing
    in this case the baseline value

    :param list lst: input signal
    :return: normalized list based on the most frequent value and the 
             value itself
    :rtype: lst, float
    '''
    counter_resp = Counter(lst)
    most_common_value = counter_resp.most_common(1)[0][0]
    list_common_value = [1 * most_common_value for _ in range(len(lst))]
    lst_norm = [a - b for a, b in zip(lst, list_common_value)]

    return lst_norm, most_common_value


def find_low_oscill(lst):
    '''
    Function detecting the central or baseline value (low oscillation)
    of a signal by removing values considered too extreme or noisy.

    :param list lst: input signal
    :return: mean value of the low oscillations
    :rtype: float
    '''
    # if the list is empty return nan
    if len(lst) == 0:
        return np.nan
    threshold = np.std(lst)
    mean_val = np.mean(lst)
    oscillating_val = [x for x in lst if abs(x - mean_val) < threshold]
    # if all the values are removed return the original mean
    if not oscillating_val:
        return mean_val 
    oscillating_center_mean = np.mean(oscillating_val)

    return oscillating_center_mean



def remove_outliers(start, end, val_out, base_mask, signal, peaks):
    '''
    Function detecting and removing outlier values.
    
    :param int start: start value
    :param int end: last value
    :param float val_out: outlier value
    :param float base_mask: baseline value of the signal
    :param list signal: input signal
    :param numpy.ndarray peaks: peaks contained in the signal
    :return: new starting and ending points of the signal, identyfing a respiratory event
    :rtype: int, int
    '''

    index_out = np.where(np.isclose(signal, val_out - base_mask)) 
    if 0 < index_out[0].size < len(signal): # outliers exists
        for peak in peaks:
            for idx in index_out[0]:
                idx = idx + start
                if idx <= peak:
                    start = idx + 1
                elif peak <= idx:
                    end_t = idx - 1
                    if end_t < end:
                        end = end_t

    if index_out[0].size == len(signal):
        start = None
        end = None


    return start, end

def find_mask_intervals(signal, base_mask):
    '''
    Function performing the segmentation of the signal related to the
    mask pressure.
    
    :param list signal: input mask signal
    :param float: baseline value of the signal
    :return: intervals of the mask signal, identyfing the respiratory events
    :rtype: list
    '''
    # identify the positive peaks of the signal, i.e. above the baseline val 
    pos_peaks, properties = find_peaks(signal, height=0, distance=25)

    # discard the smaller peaks (noise) and keep only the highest one
    # (highest centroid) by dividing them in two groups
    data = np.array(properties['peak_heights']).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(data)
    labels = kmeans.labels_

    high_cluster = np.argmax(kmeans.cluster_centers_)
    high_value_indices = np.where(labels == high_cluster)[0]
    pos_peaks = pos_peaks[high_value_indices]

    # determine the mean of the lowest peaks to retain only regions above
    # this threshold
    low_cluster = np.argmin(kmeans.cluster_centers_)
    low_values = data[labels == low_cluster]
    val_peaks = [float(f"{x[0]:.1f}") for x in low_values]
    threshold = sum(val_peaks)/len(val_peaks)
    print('t', threshold)
    above_threshold = np.abs(signal) > threshold  # Identify large movements

    # Label continuous regions where signal exceeds threshold
    labeled_array, num_features = label(above_threshold)

    # Extract start and end indices of the interval of each peak
    mask_intervals = []
    duration = []
 
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        filtered_peaks = pos_peaks[(pos_peaks >= indices[0]) & (pos_peaks <= indices[-1])]
        
        if len(filtered_peaks) != 0:
            int_signal = signal[indices[0]:indices[-1]]
            start_mask, end_mask = remove_outliers(indices[0], indices[-1], -3276.8, base_mask,  int_signal, filtered_peaks)

            if start_mask is not None and end_mask is not None: 
                if i!=1 and len(mask_intervals)>0:
                    # check for autotriggers ---> modified, sposto a dopo
                    #if start_mask - mask_intervals[-1][1] <= 8: 
                    #    # store (start,end)
                    #    mask_intervals[-1] = (mask_intervals[-1][0], int(end_mask)) 
                    #    duration[-1] = end_mask - mask_intervals[-1][0]

                    # Soglia di puro rumore dello strumento (es. 3 campioni = ~120ms)
                    # if start_mask - mask_intervals[-1][1] <= 3: 
                    #     # Questo è rumore del sensore: fondiamo i segmenti
                    #     mask_intervals[-1] = (mask_intervals[-1][0], int(end_mask))
                    #     duration[-1] = end_mask - mask_intervals[-1][0]
                    #     # altriemnti potrebbe essere un Double Trigger o un Auto-trigger reale
                    #     # Li teniamo SEPARATI così la funzione delle asincronie può misurarli
                    #     # lo verifico dopo
                    # else:
                    #     mask_intervals.append((int(start_mask), int(end_mask)))
                    #     duration.append(end_mask - start_mask)

                    mask_intervals.append((int(start_mask), int(end_mask)))
                    duration.append(end_mask - start_mask)
                else:
                    mask_intervals.append((int(start_mask), int(end_mask)))
                    duration.append(end_mask - start_mask)

    return mask_intervals



from scipy.ndimage import label, median_filter  # Aggiunto median_filter
from sklearn.cluster import KMeans

def find_mask_intervals_local(signal, base_mask, fs=24):
    """
    Function performing the segmentation of the signal related to the
    mask pressure using an adaptive local threshold.
    """
    # 1. Identifichiamo i picchi positivi grossolani
    pos_peaks, properties = find_peaks(signal, height=0, distance=25)

    # 2. Mantieni il tuo KMeans per scremare i picchi macro dagli artefatti minimi
    data = np.array(properties['peak_heights']).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(data)
    labels = kmeans.labels_

    high_cluster = np.argmax(kmeans.cluster_centers_)
    high_value_indices = np.where(labels == high_cluster)[0]
    pos_peaks = pos_peaks[high_value_indices]

    # =========================================================================
    # NUOVA LOGICA: SOGLIA ADATTIVA LOCALE (Sostituisce il threshold statico)
    # =========================================================================
    # Definiamo una finestra mobile (es. 10 secondi di segnale)
    # 10 secondi * 24 Hz = 240 campioni. Regolala se vuoi analisi più o meno strette.
    window_size = 10 * fs 
    
    # Il filtro mediano estrae la linea di base (rumore di fondo/tendenza) istante per istante
    baseline_mobile = median_filter(np.abs(signal), size=window_size)
    
    # Calcoliamo l'ampiezza media del cluster basso (il rumore residuo dei picchi)
    low_cluster = np.argmin(kmeans.cluster_centers_)
    low_values = data[labels == low_cluster]
    val_peaks = [float(f"{x[0]:.1f}") for x in low_values]
    offset_rumore = sum(val_peaks) / len(val_peaks) if len(val_peaks) > 0 else 0.5
    
    # La soglia ora è un vettore che si alza e si abbassa insieme alla baseline mobile
    threshold_adaptive = baseline_mobile + offset_rumore
    
    # Identifichiamo i movimenti ampi usando la soglia locale punto per punto
    above_threshold = np.abs(signal) > threshold_adaptive
    # =========================================================================

    # Label continuous regions where signal exceeds threshold (Resta identico!)
    labeled_array, num_features = label(above_threshold)

    # Extract start and end indices of the interval of each peak
    mask_intervals = []
    duration = []
 
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        filtered_peaks = pos_peaks[(pos_peaks >= indices[0]) & (pos_peaks <= indices[-1])]
        
        if len(filtered_peaks) != 0:
            int_signal = signal[indices[0]:indices[-1]]
            start_mask, end_mask = remove_outliers(indices[0], indices[-1], -3276.8, base_mask,  int_signal, filtered_peaks)

            if start_mask is not None and end_mask is not None: 
                if i != 1 and len(mask_intervals) > 0:
                    mask_intervals.append((int(start_mask), int(end_mask)))
                    duration.append(end_mask - start_mask)
                else:
                    mask_intervals.append((int(start_mask), int(end_mask)))
                    duration.append(end_mask - start_mask)

    return mask_intervals


def find_noise(intervals, signal, oscillating_center_mean):
    '''
    Function identifying the noise value between two identified respiratory intervals.
    This value is needed to refine the baseline value for each event and thus the 
    related starting and ending points.

    :param list intervals: list of intervals (start, end)
    :param list signal: input signal
    :param float oscillating_center_mean: mean value of the low oscillations 
                                          of the signal
    :return: list of tuples (baseline_sx, baseline_dx) for each respiratory event, 
              where baseline_x is identifying the noise value
    :rtype: list[tuple]
    '''
    noise_vals = []
    for i in range(len(intervals)):
        inter = signal[intervals[i][0]:intervals[i][1]]
        max_val = np.max(inter)
        if i != 0 and i!= len(intervals)-1:
            
            # right interval (between resp events i and i+1)
            segment_dx = np.array(signal[intervals[i][1]:intervals[i+1][0]])
            # apply a mask to consider only portion below 30% of the signal
            mask_dx = np.abs(segment_dx) < (0.3 * max_val)
            segment_dx_filtered = segment_dx[mask_dx]
            # find baseline (center low oscillations)
            if segment_dx_filtered.size > 0:
                oscillating_center_dx = find_low_oscill(segment_dx_filtered)
            else:
                # if the array is empty, the value is set to nan
                oscillating_center_dx = np.nan

            # left interval (between resp events i-1 and i)
            segment_sx = np.array(signal[intervals[i-1][1]:intervals[i][0]])
            # apply a mask to consider only portion below 30% of the signal
            mask_sx = np.abs(segment_sx) < (0.3 * max_val)
            segment_sx_filtered = segment_sx[mask_sx]
            # find baseline (center low oscillations)
            if segment_sx_filtered.size > 0:
                oscillating_center_sx = find_low_oscill(segment_sx_filtered)
            else:
                # if the array is empty, the value is set to nan
                oscillating_center_sx = np.nan


            # check for nan values. If found, the other noise value will be used
            # check first for baseline dx previous interval, that in case will be
            # changed with the baseline sx of the current one, and if also this is
            # None, the baseline sx of the previous interval will be used
            if np.isnan(noise_vals[i-1][1]):
                #print('iiii', i-1)
                if np.isnan(oscillating_center_sx):
                    noise_dx = noise_vals[i-1][0]
                else: 
                    noise_dx = oscillating_center_sx
                noise_vals[i-1] = (noise_vals[i-1][0], noise_dx) 

            # check the baseline sx on the current interval. If None, the baseline
            # sx of the previous interval (corrected previously) will be used
            if np.isnan(oscillating_center_sx):
                noise_val_sx = noise_vals[i-1][1] # Usa il valore imputato/calcolato
            else:
                noise_val_sx = oscillating_center_sx

            noise_val_dx = oscillating_center_dx

            noise_val = np.mean([oscillating_center_dx, oscillating_center_sx])

        elif i == 0: #first interval

             # right baseline
            segment_dx = np.array(signal[intervals[i][1]:intervals[i+1][0]])
            mask_dx = np.abs(segment_dx) < (0.3 * max_val)
            segment_dx_filtered = segment_dx[mask_dx]
            if segment_dx_filtered.size > 0:
                oscillating_center_dx = find_low_oscill(segment_dx_filtered)
            else:
                # if the array is empty, the value is set to nan
                oscillating_center_dx = np.nan
            
            if np.isnan(oscillating_center_dx):
                # baseline dx None --> use oscillating_center_mean
                noise_val_dx = oscillating_center_mean
            else:
                noise_val_dx = oscillating_center_dx

            # baseline sx = dx in the first interval
            noise_val_sx = noise_val_dx 
            noise_val = noise_val_dx

        elif i == len(intervals)-1: #last interval

            # left baseline
            segment_sx = np.array(signal[intervals[i-1][1]:intervals[i][0]])
            mask_sx = np.abs(segment_sx) < (0.3 * max_val)
            segment_sx_filtered = segment_sx[mask_sx]
            if segment_sx_filtered.size > 0:
                oscillating_center_sx = find_low_oscill(segment_sx_filtered)
            else:
                # if the array is empty, the value is set to nan
                oscillating_center_sx = np.nan
            
            if np.isnan(oscillating_center_sx):
                #  baseline sx None --> use baseline dx previous interval
                noise_val_sx = noise_vals[i-1][1]
            else:
                noise_val_sx = oscillating_center_sx
            
            # baseline dx = sx in the last interval
            noise_val_dx = noise_val_sx 
            noise_val = noise_val_sx

        noise_vals.append((noise_val_sx, noise_val_dx))

    return noise_vals



def find_left_extreme(signal, start_mask, start_resp):
    """
    Function finding new left extreme by checking when there is a change
    of sign in the signal under consideration.
    
    :param list signal: input signal
    :param int start_mask: initial starting point (provided by mask pressure)
    :param int start_resp: possible new starting point
    :return: index of the new left extreme
    :rtype: int

    """
    signal = np.array(signal)
    max_idx = np.argmax(signal)

    left_extreme_idx = start_resp
    left_idx = []

    # reverse checking for left extreme
    # starting from the maximum value of the signal, find the point in which there
    # is a change of sign
    for i in range(max_idx, 0, -1):
        if np.sign(signal[i]) != np.sign(signal[i - 1]):
            left_idx.append(i + start_resp)

    if len(left_idx) != 0:
        left_extreme_idx = np.max(left_idx)

    if left_extreme_idx > start_mask:
        left_extreme_idx = start_mask

    return left_extreme_idx

def find_right_extreme(signal, end_mask, end_resp, start_resp):
    """
    Function finding new right extreme by checking when there is a change
    of sign in the signal under consideration.
    
    :param list signal: input signal
    :param int end_mask: initial ending point (provided by mask pressure)
    :param int end_resp: possible new ending point
    :param int start_resp: possible new starting point
    :return: index of the new right extreme
    :rtype: int

    """
    signal = np.array(signal)
    min_idx = np.argmin(signal)

    right_extreme_idx = end_resp
    right_idx = []

 
    # reverse checking for right extreme
    # starting from the minimum value of the signal, find the point in which there
    # is a change of sign   
    for i in range(min_idx, len(signal)-1):
        if np.sign(signal[i]) != np.sign(signal[i + 1]):
            right_idx.append(i + start_resp)

    if len(right_idx) != 0:
        right_extreme_idx = np.min(right_idx)

    if right_extreme_idx < end_mask:
        right_extreme_idx = end_mask

    return right_extreme_idx

def find_resp_intervals(signal, intervals, oscillating_center_mean):
    '''
    Function performing the segmentation of the signal related to the
    respiratory airflow.
    
    :param list signal: input respiratory signal
    :param list intervals: starting intervals for repsiratory events
    :param float oscillating_center_mean: mean value of the low oscillations 
                                          of the signal
    :return: intervals of the respiratory signal, identyfing the respiratory events
            and related noise values (noise_sx, noise_dx)
    :rtype: list[tuple], list[tuple]
    '''
    # find left and right noise vals for each interval
    noise_vals = find_noise(intervals, signal, oscillating_center_mean)

    resp_intervals = []
    duration=[]
    noise_int = []
    aligned_intervals = []
    for i in range(len(intervals)):
        noise_val_sx, noise_val_dx = noise_vals[i]
        noise_val = np.mean([noise_val_sx, noise_val_dx])

        # determine right extreme (end_resp)
        # define the interval between the current and subsequent ones
        # or if this is the last interval, till the end of the signal
        if i!= len(intervals)-1:
            inter_mezzo_dx = signal[intervals[i][1]:intervals[i+1][0]]
        else:
            delta = len(signal) - intervals[i][1]
            inter_mezzo_dx = signal[intervals[i][1]:intervals[i][1]+delta]
        
        if not len(inter_mezzo_dx) > 0:
            #if is empty, set end_resp to the end of the interval
            end_resp = intervals[i][1]
        else:
            # determine point above baseline dx (if none i will use the sx val)
            threshold_dx = noise_val_dx if noise_val_dx is not None else noise_val_sx
            end_resp = inter_mezzo_dx > threshold_dx # boolean filter

            # find first value above the threshold
            if np.any(end_resp):
                end_resp = np.argmax(end_resp) + intervals[i][1]
            else:
                # no value above threshold, use the initial guess
                if inter_mezzo_dx:
                    end_resp = np.argmax(inter_mezzo_dx)  + intervals[i][1]
                else:
                    end_resp = intervals[i][1]

        # determine left extreme (start_resp)
        # determine the interval between the current and previous ones
        # or if this is the first one, from the beginning of the signal
        if i!= 0:
            inter_mezzo_sx = signal[resp_intervals[-1][1]:intervals[i][0]]
        else:
            inter_mezzo_sx = signal[0 :intervals[i][0]]

        if not len(inter_mezzo_sx) > 0:
            start_resp = intervals[i][0]
        else:
            # the search here shuold be in the reverse order
            if len(inter_mezzo_sx) > 0:
                inter_mezzo_sx.reverse()

            # determine point above baseline sx (if none i will use the dx val)
            threshold_sx = noise_val_sx if noise_val_sx is not None else noise_val_dx
            start_resp = inter_mezzo_sx > threshold_sx

            # find the first value above threshold
            if np.any(start_resp):
                start_resp_offset = len(inter_mezzo_sx) - np.argmin(start_resp) + 1 
                if i != 0:   
                    start_resp = start_resp_offset + resp_intervals[-1][1]
                else:
                    start_resp = start_resp_offset
            else:
                #No value above threshold, use the initial value 
                start_resp = intervals[i][0]

        # ensure the new start is never after the initial value for it
        if start_resp > intervals[i][0]:
            start_resp = intervals[i][0]

        # refine the interval and the points start and end

        inter_new = signal[start_resp: end_resp]
        # find the positive peak using the noise val as threshold height
        pos_peaks, _ = find_peaks(inter_new, height=noise_val, distance=25)

        if len(pos_peaks) > 0:  # Check if peaks were found
            max_peak_idx = pos_peaks[np.argmax(np.array(inter_new)[pos_peaks])]
        else:
            # otherwise use the max value
            max_peak_idx = np.argmax(inter_new)

        # absolute idx found by adding start resp 
        absolute_peak_idx = max_peak_idx + start_resp
        # refine borders by removing outliers
        start_resp, end_resp = remove_outliers(start_resp, end_resp, -327.68, 0,  inter_new, [absolute_peak_idx])

        # final extension starting and ending points
        left_extreme = find_left_extreme(inter_new, intervals[i][0], start_resp)
        if left_extreme is not None:
            start_resp = left_extreme

        right_extreme = find_right_extreme(inter_new, intervals[i][1], end_resp,start_resp)
        if right_extreme is not None:
            end_resp = right_extreme

        # fix right extreme if end_resp is too far from intervals[i][1]
        if start_resp is not None and end_resp is not None:
            if end_resp - intervals[i][1] >= 25: #1 second of difference
                end_resp = intervals[i][1] + 25 #(end_resp - intervals[i][1])/3


            # # autotrigger check, merge tow intervals if satisfied
            # if i!=0 and start_resp - resp_intervals[-1][1]  <= 4: 
            #     resp_intervals[-1] = (resp_intervals[-1][0], int(end_resp))
            #     duration[-1] = end_resp - resp_intervals[-1][0]
            #     noise_int[-1] = (noise_int[-1][0], noise_val_dx)
            #     # Uniamo anche il rispettivo intervallo macchina per non perdere il sincronismo
            #     aligned_intervals[-1] = (aligned_intervals[-1][0], intervals[i][1])
            # else:
            #     # new interval refined
            #     resp_intervals.append((int(start_resp), int(end_resp)))
            #     duration.append(end_resp - start_resp)
            #     noise_int.append((noise_val_sx, noise_val_dx))
            #     aligned_intervals.append((intervals[i][0], intervals[i][1]))

            resp_intervals.append((int(start_resp), int(end_resp)))
            duration.append(end_resp - start_resp)
            noise_int.append((noise_val_sx, noise_val_dx))
            aligned_intervals.append((intervals[i][0], intervals[i][1]))

    return resp_intervals, noise_int, aligned_intervals

def plot_mask_peaks(signal, intervals):
    signal = np.array(signal)

    plt.figure(figsize=(10, 5))
    plt.plot(signal, label="Mask Pressure Signal", alpha=0.7)
    plt.legend(fontsize=16)
    #plt.plot(pos_peaks, signal[pos_peaks], "x")
    for start, end in intervals:
        plt.axvspan(start, end, color="#F08080", alpha=0.3)  # Highlight peak intervals

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Mask Pressure", fontsize=16)
    plt.title("Mask Pressure Signal",fontsize=16) #Detected Mask Pressure Intervals", fontsize=16)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('mask.png')
    plt.show()


def plot_resp_int(signal, resp_intervals, noise_vals, mask_intervals):
    time_x = np.arange(0, len(signal))
    plt.figure(figsize=(10, 5))
    plt.plot(time_x, signal, label="Respiratory Signal", alpha=0.7)
    plt.legend(fontsize=16)
    # for start, end in mask_intervals:
    #    plt.axvspan(start, end, color="red", alpha=0.3)  # Highlight peak intervals
    # for start, end in mask_intervals:
    #     plt.axvline(x=start, color="red", linestyle="--")
    #     plt.axvline(x=end, color="red", linestyle="--")
    # for start, end in mask_intervals:
    #     plt.axvline(x=start, color="red", linestyle="--", label="Start" if start == mask_intervals[0][0] else "")
    #     plt.axvline(x=end, color="red", linestyle="--", label="End" if end == mask_intervals[0][1] else "")

    for i in range(len(resp_intervals)):
        start = resp_intervals[i][0]
        end = resp_intervals[i][1]
        time_int = [x for x in range(start, end +1)]
        noise_int = [noise_vals[i] for _ in range(len(time_int))]
        plt.plot(time_int, noise_int)

    for start, end in resp_intervals:
        plt.axvspan(start, end, color="#90EE90", alpha=0.3)  # Highlight peak intervals

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Respiratory Flow (L/min)", fontsize=16)
    plt.title("Respiratory Flow Signal",fontsize=16)# Detected Respiratory Intervals",fontsize=16)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('Resp.png')
    plt.show()

    

def segmentation(mask_press, resp_flow):
    '''
    Function performing the segmentation of the input signals, 
    identifying strating and ending points of respiratory events. 
    
    :param list mask_press: complete signal for the pressure of the mask
    :param list resp_flow: complete signal for the respiratory flow
    :return: respiratory intervals (start, end) and values of baseline noise
             for each respiratory interval
    :rtype: list[tuple], list[tuple]
    '''

    mask_press_norm, base_mask = freq_val(mask_press)
    oscillating_center_mean = find_low_oscill(resp_flow)

    #interval = (1002200, 1077275) #(596085, 596922) #(596085, 596922) #(596555, 596705) #(596085, 596922) #(275545, 276350) #(596085, 596922) #(36065, 36596) #(99477, 101722)   #(242277, 243100) # #(42277, 43380) 
    #mask_press_norm = mask_press_norm[interval[0]: interval[1]] 
    mask_intervals = find_mask_intervals_local(mask_press_norm, base_mask)
    #plot_mask_peaks(mask_press_norm, mask_intervals)

    #resp_flow = resp_flow[interval[0]: interval[1]]
    resp_intervals, noise_int, aligned_mask_intervals = find_resp_intervals(
        resp_flow, mask_intervals, oscillating_center_mean)
    #plot_resp_int(resp_flow, resp_intervals, noise_int, mask_intervals)

    return resp_intervals, noise_int, aligned_mask_intervals


def label_segments(events, apnea_events, timestamps):
    """
    Function labelling the respiratory events found based on the scoring
    files.

    :param list events: list of tuples [(start_time, end_time), ...] defining
           the respiratory events
    :param list apnea_events: List of tuples [(apnea_start_time, apnea_end_time), ...]
           defining the apnea events scored.
    :param list timestamps: list containing the data related to real times.
    :return: lists containing the labels and the (start, end) for each respiratory
            event
    :rtype: list, list[tuple] 
    """
    labels = []
    times = []
    for seg_start, seg_end in events:
        seg_start = timestamps[seg_start]
        seg_end = timestamps[seg_end]
        times.append((seg_start, seg_end))
        label = 0  # Default: non-apnea
        for apnea_start, apnea_end in apnea_events:
            apnea_start = datetime.strptime(apnea_start, "%H:%M:%S.%f").time()
            apnea_end = datetime.strptime(apnea_end, "%H:%M:%S.%f").time()

            # Check if segment overlaps with any apnea event
            if seg_start <= apnea_end and seg_end >= apnea_start:
                label = 1  # Apnea
                break
        labels.append(label)
    return labels, times


def save_testevent(resp_events, labels):
    '''
    Function saving each respiratory event with the related label.
    :param list resp_events: list containing each respiratory event detected
    :param list labels: list containing the label for each respiratory event
    '''

    if len(resp_events) != len(labels):
        raise ValueError("Both lists must have the same length.")

    # Create a DataFrame with the two lists
    data = pd.DataFrame({
        "Column1": resp_events,
        "Column2": labels
    })

    # Save the DataFrame to a file
    data.to_csv("Data_eval.csv", index=False)


def normalize_signal(signal):
    '''
    Function normalizing the signal under consideration.
    :param list signal: input signal
    :return: scaled signal
    :rtype: np.array
    '''
    
    scaler = RobustScaler()
    signal = np.array(signal)
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    return signal_scaled

def area_pos(signal, noise_val):
    signal_pos = np.maximum(signal - noise_val[0], 0)
    area_val = simpson(signal_pos)
    return area_val

def extract_feat(segment_flow, noise_val):
    '''
    Function extracting the feature from a segment:
    :param list segment_flow: input segment/signal
    :param tuple noise_int: noise values (right and left) related to the
           input signal
    :return: features of the segment under consideration
    :rtype: list
    '''
    features = []
    # Statistical features (flow)
    features.append(np.mean(segment_flow))
    features.append(np.var(segment_flow))
    features.append(np.std(segment_flow))
    min_seg = np.min(segment_flow)
    max_seg = np.max(segment_flow)
    features.append(min_seg)
    features.append(max_seg)
    features.append(max_seg - min_seg)
    features.append(np.sum(segment_flow**2))
    #features.append(spectral_entropy_wavelet(segment_flow))
    features.append(area_pos(segment_flow, noise_val))

    # Frequency-domain features (flow)
    fft_flow = np.abs(fft(segment_flow))
    features.append(np.mean(fft_flow))
    features.append(np.max(fft_flow))

    return features

def extract_feat_segments(resp_events, resp_flow, noise_int):
    '''
    Function extracting from the respiratory flow, the feature connected to 
    each respiratory event detected previously

    :param list resp_events: starting and ending points of each respiratory
           event detected
    :param list resp_flow: input signal
    :param list noise_int: noise vals (dx, sx) for each respiratory event
    :return: features for each respiratory event
    :rtype: list
    '''

    feat_segments = []
    i = 0
    for seg_start, seg_end in resp_events:
        start_idx = seg_start
        end_idx = seg_end

        segment_flow = resp_flow[start_idx:end_idx]
        feat_segments.append(extract_feat(segment_flow, noise_int[i]))
        i +=1 

    return feat_segments

def extract_resp_segments(resp_events, resp_flow):
    '''
    Function extracting from the respioratory flow, the respiratory
    events using the one detected previously.

    :param list resp_events: starting and ending points of each respiratory
           event detected
    :param list resp_flow: input signal
    :return: respiratory segments (values of each respiratory event found)
    :rtype: list
    '''

    # Prepare data for model training/testing
    resp_segments = [] 
    for seg_start, seg_end in resp_events:
        start_idx = seg_start
        end_idx = seg_end

        segment_flow = resp_flow[start_idx:end_idx]
        resp_segments.append(segment_flow)  

    return resp_segments


def prep_data(csv_files,file_listapnee, test=False):
    '''
    Function extracting and pre-processing the data of interest.

    :param list[str] csv_files: list containing the path to the data files
    :param list[str] file_listapnee: list containting the path to the scoring
           files
    :param bool test: If True, the file under consideration is for the evaluation
           phase. If False, for the training/testing phase.
    :return: respiratory segments extracted from the input respiratory signal with the
             related labels, feature extracted, patient id, timestamps (with related 
             patient id list) and starting/ending points of each segment.
    :rtype: list, list, list, list, list, list, list
    '''

    resp_segments = []
    labels_segments = []
    feat_segments = []
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

        apnee_events = extract_apnee(file_apnee)
        print('Number apnee', len(apnee_events))

        #Data Segmentation
        resp_events, noise_int = segmentation(mask_press, resp_flow)

        #labelling of the resp events
        label_events, times = label_segments(resp_events, apnee_events, timestamps)

        times_tot.extend(timestamps)
        id_times.extend([i] * len(timestamps))
        resp_int.extend(resp_events)

        if test == True:
            save_testevent(resp_events, label_events)
            with open('time_data_eval.pkl', 'wb') as f:
                pickle.dump(timestamps, f)

            with open('resp_int_eval.pkl', 'wb') as f:
                pickle.dump(resp_events, f)

            with open('apnee_true_eval.pkl', 'wb') as f:
                pickle.dump(apnee_events, f)


        n_apnee = label_events.count(1)
        print('Number of Apnea Events:', n_apnee)

        resp_flow_scaled = normalize_signal(resp_flow)
        resp_data = extract_resp_segments(resp_events, resp_flow_scaled)
        resp_data_feat = extract_feat_segments(resp_events, resp_flow_scaled, noise_int)

        id_patient = [i] * len(resp_events)
        print('data_resp', len(resp_data))

        resp_segments.extend(resp_data)

        labels_segments.extend(label_events)

        feat_segments.extend(resp_data_feat)

        id_patients.extend(id_patient)

        print('N. respiratory intervals', len(resp_events))
        print('N. respiratory data', len(resp_data))
        print('N. labels ID', len(id_patient))
    print('N. respiratory data/segments', len(resp_segments))
    print('N. labels', len(labels_segments))


    return resp_segments,  np.array(labels_segments), np.array(feat_segments), np.array(id_patients), np.array(times_tot), np.array(id_times), np.array(resp_int)


def check_apnea_criteria(X_test, y_pred, time_data, resp_int, sorted_idx, save_csv=False):
    """
    Function applying the apnea criteria of having minimum two conscutive events
    classified as apnea to have a real apnea event.

    :param np.array X_test: input sequence
    :param np.array y_pred: binary predictions fro the respiratory events
    :param np.array time_data: timestamps related to repsiratory intervals
    :param np.array resp_int: reference time intervals for respiratory segments
    :param torch.Tensor/np.array sorted_idx: original order (to be restored)
    :param bool save_csv: If True, the output is saved in a csv file.
    :return: new predictions filtered based on the apnea criteria, apnea event details
             such as index of the starting and ending points and duration, and then 
             related starting and ending times with duration
    :rtype: lst, lst, lst
    """
    y_pred = np.asarray(y_pred).ravel()
    
    # sequence duration
    seq_duration = np.array([len(seq) for seq in X_test])
    
    if sorted_idx is not None:
        original_idx = torch.argsort(sorted_idx).cpu().numpy()
        y_pred = y_pred[original_idx]
        seq_duration = seq_duration[original_idx]

    y_pred_apneas = np.zeros_like(y_pred, dtype=int)
    apnea_events = []
    apnea_events_time = []

    current_apnea_start_idx = None
    current_apnea_duration = 0

    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if current_apnea_start_idx is None:
                current_apnea_start_idx = i
            current_apnea_duration += seq_duration[i]
        
        # end event if last element has been encountered  or the 0 value is found.
        is_end = (y_pred[i] == 0) or (i == len(y_pred) - 1)
        
        if current_apnea_start_idx is not None and is_end:

            end_idx_segment = i - 1
            if y_pred[i] == 1:
                 end_idx_segment = i

            segment_count = end_idx_segment - current_apnea_start_idx + 1

            if segment_count >= 2:
                apnea_events.append((current_apnea_start_idx, end_idx_segment, current_apnea_duration))
                y_pred_apneas[current_apnea_start_idx : end_idx_segment + 1] = 1
                
                # time: use index of resp_int and time_data
                start_time_idx = resp_int[current_apnea_start_idx][0]
                end_time_idx = resp_int[end_idx_segment][1]
                apnea_events_time.append((
                    time_data[start_time_idx], 
                    time_data[end_time_idx], 
                    current_apnea_duration
                ))

            # Reset tracking variables
            current_apnea_start_idx = None
            current_apnea_duration = 0

    # csv file
    if save_csv:
        df = pd.DataFrame({'Original_Pred': y_pred.ravel(), 'Qualified_Apnea': y_pred_apneas})
        df.to_csv('apneas.csv', index=False)
        
    return y_pred_apneas, apnea_events, apnea_events_time

def convert_to_datetime(time_str):
    '''
    Function converting time strings in the correct format.

    :param  str time_str: string indicating a specific time
    :return: converted time string
    :rtype: datatime
    '''
    # Add microseconds if missing
    if '.' not in time_str:
        time_str += '.000000'  # Add zero microseconds
    return datetime.strptime(time_str, "%H:%M:%S.%f")


def to_datetime(intervals):
    '''
    Function ensuring all time strings are in the correct format (with microseconds).

    :param lst intervals: list of strating/ending point of intervals of interest
    :return: list of intervals in the correct time format
    :rtype: lst
    '''
    return [(convert_to_datetime(start), convert_to_datetime(end)) for start, end in intervals]

def intersects(interval1, interval2):
    '''
    Function detecting the intersection between two intervals

    :param lst interval1: first interval
    :param lst interval2: second interval
    :return: True or False value,  indicating the presence or not of an intersection
    :rtype: bool
    '''
    return interval1[0] <= interval2[1] and interval2[0] <= interval1[1]

def compare_intervals(list_true, list_pred):
    '''
    Function comparing true apneas with the predicted ones to understand
    how many events are missing, detected, ...

    :param lst list_true
    :param lst list_pred
    :return: tuple containing the true positive, the false negatives and the false
             positives.
    :rtype: tuple(lst, lst, lst)
    '''
    true_dt, pred_dt = to_datetime(list_true), to_datetime(list_pred)
    
    intersecting = []

    matched_trueidx = set()
    matched_predidx = set()

    # Iterate through true intervals and find corresponding predicted intervals
    for i, true_int in enumerate(true_dt):
        for j, pred_int in enumerate(pred_dt):
            # If the predicted interval has not been matched yet AND it intersects with the true interval
            if j not in matched_predidx and intersects(true_int, pred_int):
                intersecting.append((list_true[i], list_pred[j]))  # Salva le stringhe originali
                matched_trueidx.add(i)
                matched_predidx.add(j)
                # Break the inner loop because we've found a match for this true apnea.
                # This ensures one-to-one counting, even with multiple overlaps.
                break 


    # Find the unique intervals
    # false negatives
    unique_true = [list_true[i] for i in range(len(list_true)) if i not in matched_trueidx]
    # false positives
    unique_pred = [list_pred[j] for j in range(len(list_pred)) if j not in matched_predidx]

    return intersecting, unique_true, unique_pred



def merge_time_intervals(intervals):
    '''
    Function checking if there are intervals with an intersection
    and in caseit merges them.

    :param lst intervasl: list of intervals
    :return: list of intervals in datatime format
    :rtype: lst(tuple)
    '''
    # convert str in datatime obj
    time_intervals = [(datetime.strptime(start, "%H:%M:%S.%f"), 
                       datetime.strptime(end, "%H:%M:%S.%f")) for start, end in intervals]

    # order interval based on initial time
    time_intervals.sort(key=lambda x: x[0])

    merged = [time_intervals[0]]

    for start, end in time_intervals[1:]:
        last_start, last_end = merged[-1]

        # if there is an intersection, this value is updated
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return [(start.strftime("%H:%M:%S.%f"), end.strftime("%H:%M:%S.%f")) for start, end in merged]

def check_stringa(t):
    '''
    Function convertime in correct data format a string

    :param datetime/str t: object under consideration
    '''
    if isinstance(t, str):
        return t
    return t.strftime("%H:%M:%S.%f")



