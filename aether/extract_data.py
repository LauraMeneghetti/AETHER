import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import pickle

from aether.utils_data import segmentation, label_segments, extract_data, extract_apnee, extract_apnee1, sample_data, prepare_data, save_testevent, normalize_signal, extract_resp_segments, FIR_filter, extract_feat_segments

# Load respiratory measurements data and apnea
# PLACE HERE THE NAME OF THE DATA FILES FOR TRAINING/TESTING
# NEED FILES WITH ALL DATA + SCORING FILES (ONLY APNEAS)
csv_files_tot =  ['2.edf-with-date.csv', '3.edf-with-date.csv']

file_listapnee_tot = [['secondo.csv', 'secondo1.csv'], ['terzo.csv']]


csv_files_eval = [ '4.edf-with-date.csv' ]
file_listapnee_eval = [ ['quarto.csv']]

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
    X_feat = []
    id_patients = []
    times_tot = []
    id_times = []
    resp_int = []

    for i, file in enumerate(csv_files):
        
        file_name =file
        file = os.path.join('tracciati_validazione/',  file)
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

        X.extend(resp_data)

        y.extend(label_events)

        X_feat.extend(resp_data_feat)

        id_patients.extend(id_patient)

        print('N. respiratory intervals', len(resp_events))
        print('N. respiratory data', len(resp_data))
        print('N. labels ID', len(id_patient))
    print('N. respiratory data', len(X))
    print('N. labels', len(y))


    return X,  np.array(y), np.array(X_feat), np.array(id_patients), np.array(times_tot), np.array(id_times), np.array(resp_int) #np.array(X), np.array(y)

X,y, X_feat, id_patients, times_tot, id_times, resp_int=prep_data(csv_files_tot,file_listapnee_tot,test=False)
np.save("resp_labels_feat_nofir_new1.npy",y)
with open('resp_events_feat_nofir_new1.pkl', 'wb') as f:
    pickle.dump(X, f)
np.save('X_data_feat_nofir_new1.npy', X_feat)
np.save('id_list_feat_nofir_new1.npy', id_patients)
np.save('id_list_times1.npy', id_times)

with open('times_tot1.pkl', 'wb') as f:
    pickle.dump(times_tot, f)
with open('resp_int1.pkl', 'wb') as f:
    pickle.dump(resp_int, f)

X_test,y_test, X_feat_test, id_patients, _, _, _=prep_data(csv_files_eval,file_listapnee_eval, test=True)
np.save("y_test_feat_nofir1_new.npy",y_test)
with open('X_test_feat_nofir1_new.pkl', 'wb') as f:
    pickle.dump(X_test, f)
np.save("X_test_feat_nofir1_new.npy",X_feat_test)


