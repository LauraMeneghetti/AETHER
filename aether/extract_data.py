import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import pickle

from aether.utils_data import segmentation, label_segments, extract_data, extract_apnee, save_testevent, normalize_signal, extract_resp_segments, extract_feat_segments


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
        file = os.path.join('aether/tracciati_validazione/',  file)
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




if  __name__ == "__main__":

    # Load respiratory measurements data and apnea
    # PLACE HERE THE NAME OF THE DATA FILES FOR TRAINING/TESTING
    # NEED FILES WITH ALL DATA + SCORING FILES (ONLY APNEAS)
    # csv_files_tot =  ['2.edf-with-date.csv', '3.edf-with-date.csv']

    # file_listapnee_tot = [['secondo.csv', 'secondo1.csv'], ['terzo.csv']]


    # csv_files_eval = [ '4.edf-with-date.csv' ]
    # file_listapnee_eval = [ ['quarto.csv']]

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

    X,y, X_feat, id_patients, times_tot, id_times, resp_int=prep_data(csv_files_tot,file_listapnee_tot,test=False)
    np.save("resp_labels.npy",y)
    with open('resp_events.pkl', 'wb') as f:
        pickle.dump(X, f)
    np.save('resp_feat.npy', X_feat)
    np.save('id_list.npy', id_patients)
    np.save('id_list_times.npy', id_times)

    with open('times_tot.pkl', 'wb') as f:
        pickle.dump(times_tot, f)
    with open('resp_int.pkl', 'wb') as f:
        pickle.dump(resp_int, f)

    X_test,y_test, X_feat_test, id_patients, _, _, _=prep_data(csv_files_eval,file_listapnee_eval, test=True)
    np.save("labels_test.npy",y_test)
    with open('data_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    np.save("data_test_feat.npy",X_feat_test)


