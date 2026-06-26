import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import csv

notebook_dir = os.getcwd()
parent_dir = os.path.join(notebook_dir, '..')
sys.path.append(parent_dir)
from aether.utils_data import extract_data, extract_apnee, normalize_signal
from aether.utils_data import segmentation

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

def create_labels_with_overlap(timestamps, apnea_events, overlap_thresh=0.5):
    """
    Crea un vettore di label globale (1 campione = 1 label) calcolando se il punto
    è coperto da un'apnea.
    """
    labels = np.zeros(len(timestamps), dtype=int)
    
    for apnea_start_str, apnea_end_str in apnea_events:
        apnea_start = datetime.strptime(apnea_start_str, "%H:%M:%S.%f").time()
        apnea_end = datetime.strptime(apnea_end_str, "%H:%M:%S.%f").time()
        
        # Trova gli indici temporali che cadono dentro l'apnea
        for idx, t in enumerate(timestamps):
            if apnea_start <= t <= apnea_end:
                labels[idx] = 1
                
    return labels

def create_windows_multimodal(signal, geom_features, labels, fs=24, t_window=10, overlap=0.5):
    """
    Affetta contemporaneamente i segnali scalati, le feature geometriche e applica
    la regola del 50% di maggioranza per la label della finestra.
    """
    n_points = int(t_window * fs)
    step = int(n_points * (1 - overlap))
    
    sig_windows = []
    geom_windows = []
    label_windows = []
    
    for start_idx in range(0, len(signal) - n_points, step):
        end_idx = start_idx + n_points
        
        # Estrai le finestre
        window_sig = signal[start_idx:end_idx]
        window_geom = geom_features[start_idx:end_idx]
        window_labels = labels[start_idx:end_idx]
        
        # CRITERIO MATEMATICO DEL 50%:
        # Se più del 50% dei campioni nella finestra sono marcati come apnea, la finestra è Apnea
        if np.mean(window_labels) >= 0.5:
            final_label = 1
        else:
            final_label = 0
            
        sig_windows.append(window_sig)
        geom_windows.append(window_geom)
        label_windows.append(final_label)
        
    return np.array(sig_windows), np.array(geom_windows), np.array(label_windows)

def find_asincrony(timeline, timestamps, asyn, csv_path, file_name):
        # 2. Conta il numero totale di campioni marcati come 1
        totale_campioni_1 = np.sum(timeline == 1.0)

        # 3. Conta quanti EVENTI discreti (blocchi di 1) ci sono nella notte
        # Sfruttiamo la derivata del flag: ogni volta che passa da 0 a 1 c'è un fronte di salita (+1)
        num_eventi = np.sum(np.diff(timeline) == 1.0)
        # Controllo di sicurezza se la notte iniziasse già in double trigger (rarissimo ma possibile)
        if timeline[0] == 1.0:
            num_eventi += 1

        # --- STAMPA I REQUISITI ---
        print(f"-> Numero totale di campioni con '1': {totale_campioni_1}")
        print(f"-> Durata totale in secondi affetta:  {totale_campioni_1 / 24:.2f} secondi")
        print(f"-> Numero di episodi rilevati: {num_eventi}")
        print("-" * 40)

        # --- ESTRAZIONE EVENTI CON TIMESTAMPS (STILE APNEA) ---
        cambi_stato = np.diff(timeline)
        
        # Identifichiamo gli indici sulla timeline locale
        inizi_indici = np.where(cambi_stato == 1.0)[0] + 1
        fini_indici = np.where(cambi_stato == -1.0)[0] + 1

        if timeline[0] == 1.0:
            inizi_indici = np.insert(inizi_indici, 0, 0)
        if timeline[-1] == 1.0:
            fini_indici = np.append(fini_indici, len(timeline) - 1)

        events_time = []

        
        for num_evento, (ini, fin) in enumerate(zip(inizi_indici, fini_indici), 1):
            durata_campioni = fin - ini
            durata_secondi = durata_campioni / 24
            start = 0 #1002200
            # Dal momento che la tua timeline parte dall'indice 'start' del file originale,
            # per pescare il timestamp corretto dobbiamo sommare l'offset 'start'
            idx_timestamp_inizio = start + ini
            idx_timestamp_fine = start + fin
            
            # Estraiamo le stringhe orarie direttamente dall'array dei timestamps del file
            start_time_str = timestamps[idx_timestamp_inizio]
            end_time_str = timestamps[idx_timestamp_fine]
            # Creiamo il dizionario con tutti i dati richiesti, incluso il TIPO
            evento_dict = {
                'Inizio': start_time_str,
                'Fine': end_time_str,
                'Durata_Secondi': round(durata_secondi, 2),
                'Tipo_Asincronia': asyn,
                'file_name': file_name
            }
            # Salviamo la tupla (Inizio, Fine, Durata) proprio come fai per le apnee
            events_time.append(evento_dict)
            
            print(f"  Evento {num_evento}: Indici [{ini} : {fin}] | Ora: {start_time_str} -> {end_time_str} | Durata: {durata_secondi:.2f}s")
            
        print("-" * 40)

        # --- SALVATAGGIO IN CSV (Se il percorso è fornito) ---
        if csv_path and len(events_time) > 0:
            file_esiste = os.path.exists(csv_path)
            
            # Apriamo in modalità 'a' (append) così puoi chiamare la funzione più volte 
            # per diverse asincronie e scriverle tutte nello stesso file
            with open(csv_path, mode='a', newline='', encoding='utf-8') as file_csv:
                campi = ['Inizio', 'Fine', 'Durata_Secondi', 'Tipo_Asincronia', 'file_name']
                scrittore = csv.DictWriter(file_csv, fieldnames=campi)
                
                # Scriviamo l'intestazione solo se il file non esiste ancora
                if not file_esiste:
                    scrittore.writeheader()
                    
                scrittore.writerows(events_time)
            print(f"[INFO] Salvati {len(events_time)} eventi di '{asyn}' in: {csv_path}")
            print("=" * 40)

    #return events_list

import numpy as np
from scipy.signal import savgol_filter

def detect_double_triggering_refined(aligned_mask_intervals, resp_intervals, noise_int, resp_flow, mask_pressure, fs=24):
    """
    Versione corretta e allineata: usa aligned_mask_intervals per garantire la 
    corrispondenza biunivoca degli indici con noise_int e resp_intervals.
    """
    n_campioni = len(resp_flow)
    double_trigger_timeline = np.zeros(n_campioni)
    resp_flow_np = np.asarray(resp_flow)
    mask_pressure_np = np.asarray(mask_pressure)
    
    # --- STEP 1: CALCOLO DEL TEMPO INSPIRATORIO BIOLOGICO MEDIO ---
    durate_insp_paziente = []
    espirazioni_minime_normali = []
    
    for idx in range(len(resp_intervals)):
        start_r, end_r = resp_intervals[idx]
        if start_r is None or end_r is None:
            continue
            
        ciclo_flow = resp_flow_np[start_r:end_r]
        if len(ciclo_flow) == 0:
            continue
            
        # Unpacking pulito della tupla di float
        noise_sx = float(noise_int[idx][0])
        noise_dx = float(noise_int[idx][1])
        zero_biologico_locale = (noise_sx + noise_dx) / 2.0
        
        max_f_idx = np.argmax(ciclo_flow)
        segno = np.sign(ciclo_flow[max_f_idx:] - zero_biologico_locale)
        zc = np.where(np.diff(segno))[0]
        
        if len(zc) > 0:
            end_insp_reale = zc[0] + max_f_idx
            durate_insp_paziente.append(float(end_insp_reale / fs))
            finestra_espirio = ciclo_flow[end_insp_reale:]
            if len(finestra_espirio) > 0:
                # Forza il minimo a float puro per non infettare la lista
                espirazioni_minime_normali.append(float(np.min(finestra_espirio)) - zero_biologico_locale)
        else:
            # Se è un ciclo fuso e non interseca lo zero, stimiamo l'espirazione dalla fine del ciclo
            durate_insp_paziente.append(float((max_f_idx) / fs))

    t_insp_medio_paziente = float(np.nanmean(durate_insp_paziente)) if len(durate_insp_paziente) > 0 else 0.5
    picco_espiratorio_medio = float(np.nanmean(espirazioni_minime_normali)) if len(espirazioni_minime_normali) > 0 else -1.0
    soglia_espirazione_breve = 0.3 * t_insp_medio_paziente
    print(t_insp_medio_paziente)
    print(picco_espiratorio_medio)
    print(soglia_espirazione_breve)

    # --- STEP 2: VERIFICA INCROCIATA MORFOLOGICA ---
    
    # =========================================================================
    # LOGICA A: CONTROLLO INTRA-CICLO (Usa ALIGNED_MASK_INTERVALS)
    # =========================================================================
    for idx in range(len(aligned_mask_intervals)):
        start_m, end_m = aligned_mask_intervals[idx]
        if start_m is None or end_m is None:
            continue
            
        # Togli maschere troppo brevi (meno di mezzo secondo)
        if (end_m - start_m) < int(0.5 * fs):
            continue
            
        ciclo_flow = resp_flow_np[start_m:end_m]
        ciclo_pres = mask_pressure_np[start_m:end_m]
        
        if len(ciclo_flow) == 0 or len(ciclo_pres) == 0:
            continue
        
        # 1. Baseline e Zero Biologico (Esattamente come in Logica B)
        noise_sx = float(noise_int[idx][0])
        noise_dx = float(noise_int[idx][1])
        zero_biologico_flow = (noise_sx + noise_dx) / 2.0
        
        # 2. Analisi dei picchi del FLUSSO con find_peaks
        max_flow = np.max(ciclo_flow)
        soglia_altezza_flow = zero_biologico_flow + 0.3 * max_flow #0.1 #margine per rumore
        # Usiamo find_peaks per trovare le spinte inspiratorie del paziente
        picchi_flow_validi, _ = find_peaks(ciclo_flow, height=soglia_altezza_flow, prominence=0.1, distance=10)
        
        # 3. Analisi dei picchi della PRESSIONE con find_peaks
        min_pres, max_pres = np.min(ciclo_pres), np.max(ciclo_pres)
        soglia_altezza_pres = 0.3 * (max_pres - min_pres)
        # Usiamo find_peaks per trovare le gobbe erogate dal ventilatore
        picchi_pres_validi, _ = find_peaks(ciclo_pres, height=soglia_altezza_pres, prominence=0.4, distance=10)
        
        # 4. Verifica di Double Trigger Intra-Ciclo
        if len(picchi_flow_validi) >= 2 and len(picchi_pres_validi) >= 2:
            # Prendiamo i primi due picchi identificati nel flusso
            idx_p1 = picchi_flow_validi[0]
            idx_p2 = picchi_flow_validi[1]
            
            # 1. CRITERIO DEL TEMPO (Simmetrico all'Inter-ciclo)
            # Calcoliamo la distanza temporale tra i due picchi fusi nella stessa maschera
            durata_transizione_interna = (idx_p2 - idx_p1) / fs
            
            # Se la distanza tra i due picchi è inferiore alla soglia (30% del T_insp medio),
            # significa che il secondo picco è caduto dentro la finestra di Double Trigger.
            if durata_transizione_interna < soglia_espirazione_breve:
                
                # Isoliamo la valle tra i due picchi reali
                valle_intermedia = ciclo_flow[idx_p1:idx_p2]
                if len(valle_intermedia) == 0:
                    continue

                # # Regola Il secondo picco di flusso è tipicamente più basso del primo
                # valore_p1 = ciclo_flow[idx_p1]
                # valore_p2 = ciclo_flow[idx_p2]
                # secondo_picco_minore = valore_p2 < (1.05 * valore_p1)
                
                min_flusso_transizione = float(np.min(valle_intermedia)) - zero_biologico_flow
                
                # Confronto pulito tra FLOAT: restituisce un singolo True o False
                flusso_meno_negativo = min_flusso_transizione > (0.5 * picco_espiratorio_medio)
                #cerco se ho fase piatta del segnale prima inizio atto respiratrio successivo
                #campioni_piatti = np.sum(np.abs(df_flusso_transizione) < 0.1)
                campioni_piatti = np.sum(np.abs(valle_intermedia - zero_biologico_flow) < 0.1)

                if len(valle_intermedia) <= 3:
                    manca_pausa = True
                else:
                    manca_pausa = int(campioni_piatti) <= 2
                
                # Se l'ampiezza conferma il mancato svuotamento e non c'è pausa statica a zero,
                # allora questa maschera fusa contiene un Double Trigger reale.
                if flusso_meno_negativo and manca_pausa: # and secondo_picco_minore:
                    double_trigger_timeline[start_m:end_m] = 1.0
    # =========================================================================
    # LOGICA B: CONTROLLO INTER-CICLO (Usa ALIGNED_MASK_INTERVALS)
    # =========================================================================
    for idx in range(len(aligned_mask_intervals) - 1):
        start_m1, end_m1 = aligned_mask_intervals[idx]

        start_m2, end_m2 = aligned_mask_intervals[idx+1]
        if None in [start_m1, end_m1, start_m2, end_m2]:
            continue
            
        durata_transizione_macchina = (start_m2 - end_m1) / fs
        
        if durata_transizione_macchina < soglia_espirazione_breve:
            #print('s1', start_m1)
            #print(end_m1)
            #print(start_m2)
            #print('e2', end_m2)
            flusso_transizione = resp_flow_np[end_m1:start_m2]
            #df_flusso_transizione = df_flow[end_m1:start_m2]
            
            if len(flusso_transizione) == 0:
                continue
            # # 1. Regola [3]: Il secondo ciclo (m2) deve essere più breve del primo (m1)
            # durata_m1 = (end_m1 - start_m1) / fs
            # durata_m2 = (end_m2 - start_m2) / fs
            # secondo_ciclo_piu_breve = durata_m2 <= durata_m1
            
            # # 2. Regola [4]: Il picco di flusso del secondo ciclo deve essere inferiore al primo
            # # Recuperiamo i massimi assoluti di flusso dentro le due maschere
            # picco_flow_m1 = np.max(resp_flow_np[start_m1:end_m1])
            # picco_flow_m2 = np.max(resp_flow_np[start_m2:end_m2])
            # secondo_picco_flow_minore = picco_flow_m2 <= picco_flow_m1
            
            # Nessun accrocchio col min(): idx è sincronizzato al millisecondo
            noise_sx = float(noise_int[idx][0])
            noise_dx = float(noise_int[idx][1])
            zero_biologico_coppia = (noise_sx + noise_dx) / 2.0
            
            min_flusso_transizione = float(np.min(flusso_transizione)) - zero_biologico_coppia
            
            # Confronto pulito tra FLOAT: restituisce un singolo True o False
            flusso_meno_negativo = min_flusso_transizione > (0.5 * picco_espiratorio_medio)
            #cerco se ho fase piatta del segnale prima inizio atto respiratrio successivo
            #campioni_piatti = np.sum(np.abs(df_flusso_transizione) < 0.1)
            campioni_piatti = np.sum(np.abs(flusso_transizione - zero_biologico_coppia) < 0.1)
            if len(flusso_transizione) <= 3:
                manca_pausa = True
            else:
                manca_pausa = int(campioni_piatti) <= 2

            if flusso_meno_negativo and manca_pausa: # and secondo_ciclo_piu_breve and secondo_picco_flow_minore:
                #print(manca_pausa)
                double_trigger_timeline[start_m1:end_m2] = 1.0

    return double_trigger_timeline


def detect_autotriggering(aligned_mask_intervals, resp_flow, mask_pressure, fs=24):
    """
    Rileva gli eventi di Auto-triggering basandosi sul confronto con il ritmo medio 
    del paziente e analizzando la mancanza di sforzo esclusivamente tra atti consecutivi.
    """
    n_campioni = len(resp_flow)
    autotrigger_timeline = np.zeros(n_campioni)
    
    mask_pressure_np = np.asarray(mask_pressure)
    
    # --- STEP 1: CALCOLO DEL RITMO MEDIO (FREQUENZA) DELLA REGISTRAZIONE ---
    intertempi = []
    caratteristiche_cicli = {}
    
    indici_validi = []
    for idx, (start_m, end_m) in enumerate(aligned_mask_intervals):
        if start_m is None or end_m is None or (end_m - start_m) < int(0.3 * fs):
            continue
        indici_validi.append(idx)
        
        # Salviamo i dati geometrici di base del ciclo
        ciclo_pres = mask_pressure_np[start_m:end_m]
        idx_picco_pres = np.argmax(ciclo_pres)
        
        caratteristiche_cicli[idx] = {
            'start': start_m,
            'end': end_m,
            't_tot': (end_m - start_m) / fs,
            't_salita': idx_picco_pres / fs,
            't_discesa': (len(ciclo_pres) - idx_picco_pres) / fs
        }
    
    # Calcoliamo la distanza temporale tra l'inizio di un atto e l'inizio del successivo
    for i in range(len(indici_validi) - 1):
        idx_curr = indici_validi[i]
        idx_next = indici_validi[i+1]
        
        tempo_tra_atti = (aligned_mask_intervals[idx_next][0] - aligned_mask_intervals[idx_curr][0]) / fs
        intertempi.append(tempo_tra_atti)
            
    # Usiamo la MEDIANA: estrae il ritmo dominante ignorando i buchi e le pause lunghe
    periodo_medio_paziente = np.median(intertempi) if len(intertempi) > 5 else 1.0
    
    # --- STEP 2: VERIFICA DEI BURST DI AUTO-TRIGGER ---
    for i in range(len(indici_validi) - 1):
        idx_curr = indici_validi[i]
        idx_next = indici_validi[i+1]
        
        c1 = caratteristiche_cicli[idx_curr]
        c2 = caratteristiche_cicli[idx_next]
        
        # 1. CONTROLLO DELLA FREQUENZA RELATIVA (Il tuo approccio basato sulle medie)
        periodo_corrente = (c2['start'] - c1['start']) / fs
        pausa_tra_maschere = (c2['start'] - c1['end']) / fs
        
        # La macchina è considerata più veloce del ritmo medio se il periodo attuale 
        # è significativamente più breve del ritmo medio registrato (es. < 65%)
        macchina_accelerata = periodo_corrente < (0.65 * periodo_medio_paziente)
        
        # Manteniamo anche il vincolo fisico del periodo refrattario (pausa < 150ms)
        rispetta_periodo_refrattario = pausa_tra_maschere <= 0.200
        
        if macchina_accelerata and rispetta_periodo_refrattario:
            
            # 2. CONTROLLO DELLO SFORZO SOLO NELLA TRANSIZIONE (La tua correzione chiave)
            # Guardiamo la pressione NELLO SPAZIO TRA I DUE ATTI (fine del primo, inizio del secondo)
            finestra_transizione = mask_pressure_np[c1['end']:c2['start']]
            
            manca_sforzo_tra_atti = False
            if len(finestra_transizione) > 0:
                peep_locale = mask_pressure_np[c1['end']]
                min_pressione_transizione = np.min(finestra_transizione)
                
                # Se nella pausa la pressione non scende sotto la PEEP (manca la deflessione negativa),
                # significa che il paziente non ha chiamato il secondo atto. Il trigger è falso.
                if (peep_locale - min_pressione_transizione) < 0.2:
                    manca_sforzo_tra_atti = True
            else:
                # Se la pausa è praticamente zero campioni, non c'è stato spazio fisico per lo sforzo
                manca_sforzo_tra_atti = True
                
            if manca_sforzo_tra_atti:
                
                # # 3. VERIFICA MORFOLOGICA (I due cicli erogati dalla macchina sono stereotipati?)
                # tolleranza_tempo = 0.05 
                # stesso_t_tot = abs(c1['t_tot'] - c2['t_tot']) < tolleranza_tempo
                # stessa_salita = abs(c1['t_salita'] - c2['t_salita']) < tolleranza_tempo
                # stessa_discesa = abs(c1['t_discesa'] - c2['t_discesa']) < tolleranza_tempo
                
                # if stesso_t_tot and stessa_salita and stessa_discesa:
                #     # Se la macchina ha accelerato rispetto alla media, non c'è sforzo nella transizione,
                #     # e le curve sono fotocopie: è Auto-trigger.
                    
                autotrigger_timeline[c1['start']:c1['end']] = 1.0
                autotrigger_timeline[c2['start']:c2['end']] = 1.0

    return autotrigger_timeline


def detect_reverse_triggering(aligned_mask_intervals, resp_flow, mask_pressure, fs=24):
    """
    Rileva il Reverse Triggering analizzando ogni singola coppia di atti consecutivi 
    in modo indipendente, basandosi sulla vicinanza temporale e sull'asimmetria geometrica 
    dei picchi di flusso.
    """
    n_campioni = len(resp_flow)
    reverse_trigger_timeline = np.zeros(n_campioni)
    
    resp_flow_np = np.asarray(resp_flow)
    caratteristiche_cicli = {}
    
    # --- STEP 1: ESTRAZIONE PARAMETRI DEI CICLI ---
    indici_validi = []
    for idx, (start_m, end_m) in enumerate(aligned_mask_intervals):
        if start_m is None or end_m is None or (end_m - start_m) < int(0.3 * fs):
            continue
        indici_validi.append(idx)
        
        # Troviamo il picco massimo di flusso all'interno di questa maschera
        f_max = np.max(resp_flow_np[start_m:end_m])
        
        caratteristiche_cicli[idx] = {
            'start': start_m,
            'end': end_m,
            'durata': (end_m - start_m) / fs,
            'picco_flow': f_max
        }


    # --- STEP 2: ANALISI DELLE COPPIE ISOLATE ---
    for i in range(len(indici_validi) - 1):
        idx_1 = indici_validi[i]
        idx_2 = indici_validi[i+1]
        
        c1 = caratteristiche_cicli[idx_1] # Primo ciclo (Macchina)
        c2 = caratteristiche_cicli[idx_2] # Secondo ciclo (Paziente)
        
        # 1. VINCOLO TEMPORALE: Pausa tra i due cicli estremamente ridotta
        pausa_tra_cicli = (c2['start'] - c1['end']) / fs
        
        if pausa_tra_cicli <= 0.200: # Meno di 250 ms
            
            # 2. VINCOLO MORFOLOGICO DEL FLUSSO (Le orecchie di coniglio)
            # Il secondo ciclo indotto dal riflesso è strutturalmente più breve e più basso
            secondo_ciclo_piu_breve = c2['durata'] < c1['durata']
            #secondo_picco_minore = c2['picco_flow'] < c1['picco_flow']
            
            # Se la singola coppia rispetta queste caratteristiche geometriche, 
            # viene marcata immediatamente come Reverse Trigger
            if secondo_ciclo_piu_breve: #and secondo_picco_minore:
                reverse_trigger_timeline[c1['start']:c2['end']] = 1.0

    return reverse_trigger_timeline



def detect_underassistance(aligned_mask_intervals, resp_flow, mask_pressure, noise_int, fs=24):
    """
    Rileva l'Underassistance (insufficienza di supporto ventilatorio) distinguendo tra:
    - S9a: Mancanza di flusso all'inizio/intero ciclo (Profilo a trapezio della pressione).
    - S9b: Insufficienza di flusso a metà del ciclo inspiratorio (Doppia gobba sul flusso).
    """
    n_campioni = len(resp_flow)
    underassistance_timeline = np.zeros(n_campioni)
    
    mask_pressure_np = np.asarray(mask_pressure)
    resp_flow_np = np.asarray(resp_flow)
    
    for idx, (start_m, end_m) in enumerate(aligned_mask_intervals):
        if start_m is None or end_m is None or (end_m - start_m) < int(0.3 * fs):
            continue
            
        # Ritagliamo i segnali di pressione e flusso del ciclo corrente
        ciclo_pres = mask_pressure_np[start_m:end_m]
        ciclo_flow = resp_flow_np[start_m:end_m]
        lunghezza_ciclo = len(ciclo_pres)
        
        if lunghezza_ciclo < 6: # Ciclo troppo corto per essere analizzato intraciclo
            continue
            
        # --- CALCOLO ZERO BIOLOGICO DEL FLUSSO (DALLA TUA LOGICA B) ---
        noise_sx = float(noise_int[idx][0])
        noise_dx = float(noise_int[idx][1])
        zero_biologico_flow = (noise_sx + noise_dx) / 2.0

        # Troviamo l'istante (indice) dei picchi massimi assoluti del ciclo
        #idx_picco_pres = np.argmax(ciclo_pres)
        idx_picco_flow = np.argmax(ciclo_flow)
        # --- CORREZIONE PER IL PLATEAU DI PRESSIONE ---
        pressione_massima = np.max(ciclo_pres)
        soglia_plateau = 0.95 * pressione_massima
        
        # Troviamo il PRIMO indice in cui la pressione raggiunge almeno il 95% del massimo
        idx_picco_pres = np.where(ciclo_pres >= soglia_plateau)[0][0]
        
        # Dividiamo il ciclo a metà per l'analisi della seconda parte dell'atto
        meta_ciclo = lunghezza_ciclo // 2
        seconda_meta_pres = ciclo_pres[meta_ciclo:]
        seconda_meta_flow = ciclo_flow[meta_ciclo:]

        # Estrarre i riferimenti della prima metà per i confronti di S9b
        prima_meta_flow = ciclo_flow[:meta_ciclo]
        prima_meta_pres = ciclo_pres[:meta_ciclo]
        
        # Evitiamo array vuoti se il ciclo è strano
        if len(prima_meta_flow) == 0 or len(prima_meta_pres) == 0:
            continue
            
        primo_picco_flow_locale = np.max(prima_meta_flow)
        ipap_prima_meta = np.max(prima_meta_pres)
        
        # =========================================================================
        # TARGET S9a: UNDERASSISTANCE ALL'INIZIO / INTERO CICLO
        # =========================================================================
        # 1. Il picco del flusso precede temporalmente il picco della pressione
        picco_pressione_ritardato_assoluto = False
        distanza_significativa = False
        if idx_picco_flow < idx_picco_pres:
        
            # 2. DISTANZA FISSA (NON PERCENTUALE): 
            # Calcoliamo quanti campioni passano tra il picco di flusso e il picco di pressione.
            # Se passano più di 4-5 campioni (circa 150-200 ms a 24Hz), significa che c'è un ritardo
            # patologico della pressione nel raggiungere l'IPAP rispetto alla richiesta del flusso.
            ritardo_pressione_campioni = idx_picco_pres - idx_picco_flow
            distanza_significativa = ritardo_pressione_campioni >= 6  # circa > 160 ms
            
            # 3. VERIFICA MORFOLOGIA A TRAPEZIO:
            # Invece di controllare se il picco è dopo il 40%, guardiamo se il picco di pressione
            # si trova nella seconda metà del ciclo respiratorio, il che è totalmente anomalo 
            # per una pressurizzazione controllata (che dovrebbe completarsi nei primi istanti).
            picco_pressione_ritardato_assoluto = idx_picco_pres > (lunghezza_ciclo // 2)

        under_inizio = distanza_significativa and picco_pressione_ritardato_assoluto
        # =========================================================================
        # TARGET S9b: UNDERASSISTANCE A METÀ CICLO
        # =========================================================================
        under_meta = False
        if len(seconda_meta_flow) > 4:
            picco_flusso_ciclo = np.max(ciclo_flow)
            flow_ritorna_a_salire = False
            
            # 1. FLUSSO: Cerchiamo la seconda gobba SOLO sopra lo zero biologico
            indici_flusso_attivo = np.where(seconda_meta_flow > zero_biologico_flow)[0]
            
            if len(indici_flusso_attivo) > 3:
                # Escludiamo l'ultimo punto utile per evitare artefatti di confine
                for t in range(1, len(indici_flusso_attivo) - 1):
                    idx_reale = indici_flusso_attivo[t]
                    
                    # Ricerca della valle geometrica locale (minimo locale)
                    if (seconda_meta_flow[idx_reale] < seconda_meta_flow[idx_reale - 1] and 
                        seconda_meta_flow[idx_reale] < seconda_meta_flow[idx_reale + 1]):
                        
                        valore_valle = seconda_meta_flow[idx_reale]
                        
                        # Il picco successivo deve trovarsi sempre prima che il flusso scenda sotto lo zero biologico
                        rimanenti_attivi = seconda_meta_flow[idx_reale + 1 : indici_flusso_attivo[-1] + 1]
                        
                        if len(rimanenti_attivi) > 0:
                            valore_massimo_successivo = np.max(rimanenti_attivi)
                            risalita = max(3.0, 0.15 * primo_picco_flow_locale)
                            gobba = (valore_massimo_successivo - valore_valle) > risalita

                            flusso_picco = valore_massimo_successivo <= primo_picco_flow_locale
                            # Se risale significativamente prima di sgonfiarsi, è la seconda gobba
                            if gobba and flusso_picco:
                                
                                flow_ritorna_a_salire = True
                                break
            
            # 2. PRESSIONE: La sella deve verificarsi durante la fase attiva del flusso
            pres_ha_sella = False
            if flow_ritorna_a_salire:
                #ipap_locale = np.max(ciclo_pres)
                
                # Tagliamo gli ultimi 3 campioni dal corpo del ciclo per ignorare il crollo 
                # fisiologico della pressione verso la PEEP a fine atto (cycling)
                if len(seconda_meta_pres) > 4:
                    pressione_corpo_ciclo = seconda_meta_pres[:-3] 
                    min_transitorio_pres = np.min(pressione_corpo_ciclo)

                    picco_pres_succ = np.max(pressione_corpo_ciclo)
                    sella = (ipap_prima_meta - min_transitorio_pres) >= 0.8

                    pres_picco = picco_pres_succ >= ipap_prima_meta
                    
                    # Se c'è una caduta temporanea significativa rispetto all'IPAP massimo
                    if sella and pres_picco:
                        pres_ha_sella = True
                        
            if pres_ha_sella and flow_ritorna_a_salire:
                under_meta = True
                
        # =========================================================================
        # MARCATURA FINALE DELLA TIMELINE
        # =========================================================================
        if under_inizio or under_meta:
            underassistance_timeline[start_m:end_m] = 1.0

    return underassistance_timeline

def check_consecutive_oscillations(valori, max_scostamento=0.8, min_punti=3):
    """
    Verifica se nella lista sono presenti sequenze di punti consecutivi 
    che rimangono confinati (oscillando o stallando) dentro una tolleranza ristretta.
    
    Parameters:
    -----------
    valori : list o np.array
        La sequenza numerica da analizzare (es. la coda del flusso o della pressione).
    max_scostamento : float
        La massima differenza ammessa (Max - Min) all'interno della finestra 
        per considerare i punti bloccati in un plateau/oscillazione stretta.
    min_punti : int
        Quanti valori consecutivi vicini cerchiamo (es. più di 2 -> min_punti=3).
        
    Returns:
    --------
    bool : True se viene trovata almeno una sequenza bloccata, False altrimenti.
    list : Lista di tuple con gli indici (inizio, fine) delle sequenze trovate.
    """
    valori_np = np.asarray(valori, dtype=float)
    n = len(valori_np)
    
    if n < min_punti:
        return False, []
    
    sequenze_trovate = []
    i = 0
    
    while i <= n - min_punti:
        finestra_lunga = min_punti
        ha_oscillazione = False
        
        # Estendi la finestra in avanti per catturare tutta la catena bloccata
        while i + finestra_lunga <= n:
            sotto_sequenza = valori_np[i : i + finestra_lunga]
            
            # Il cuore della modifica: misuriamo solo l'escursione totale nella finestra
            valore_max = np.max(sotto_sequenza)
            valore_min = np.min(sotto_sequenza)
            escursione = valore_max - valore_min
            
            # Se l'escursione massima rimane dentro la tolleranza, i punti sono "bloccati"
            if escursione <= max_scostamento:
                ha_oscillazione = True
                finestra_lunga += 1  # Prova ad allargare la finestra per il prossimo campione
            else:
                break  # L'escursione è troppo grande, la catena si è rotta
                
        if ha_oscillazione:
            # Salviamo gli indici esatti della sequenza trovata
            idx_fine = i + finestra_lunga - 1
            sequenze_trovate.append((i, idx_fine))
            # Spostiamo l'indice alla fine della sequenza per evitare sovrapposizioni
            i = idx_fine - 1
            
        i += 1
        
    presente = len(sequenze_trovate) > 0
    return presente, sequenze_trovate

def detect_overshoot(aligned_mask_intervals, resp_flow, mask_pressure, noise_int, fs=24):
    """
    Rileva l'Overshoot (Target S10) basandosi sulla dinamica macroscopica iniziale.
    - Pressione: Salita verticale, picco acuto iniziale con micro-crollo, plateau stabile.
    - Flusso: Singolo spike iniziale veloce, crollo rapido, e coda pulita monotonica.
              Vengono rigettati i flussi instabili con picchi multipli (es. image_2f25f9).
    """
    n_campioni = len(resp_flow)
    overshoot_timeline = np.zeros(n_campioni)
    
    mask_pressure_np = np.asarray(mask_pressure)
    resp_flow_np = np.asarray(resp_flow)
    
    for idx, (start_m, end_m) in enumerate(aligned_mask_intervals):
        if start_m is None or end_m is None or (end_m - start_m) < int(0.3 * fs):
            continue
            
        ciclo_pres = mask_pressure_np[start_m:end_m]
        ciclo_flow = resp_flow_np[start_m:end_m]
        lunghezza_ciclo = len(ciclo_pres)
        
        if lunghezza_ciclo < 8:
            continue
            
        # --- CALCOLO ZERO BIOLOGICO DEL FLUSSO ---
        noise_sx = float(noise_int[idx][0])
        noise_dx = float(noise_int[idx][1])
        zero_biologico_flow = (noise_sx + noise_dx) / 2.0
        
        finestra_inizio = lunghezza_ciclo // 2
        if finestra_inizio < 3:
            continue
            
        indici_sopra_zero = np.where(ciclo_flow > zero_biologico_flow)[0]
        if len(indici_sopra_zero) < 5: 
            continue
        idx_fine_atto_attivo = indici_sopra_zero[-1]

        # =========================================================================
        # 1. ANALISI PRESSIONE (Salita verticale -> Micro-crollo -> Plateau)
        # =========================================================================
        idx_picco_pres = np.argmax(ciclo_pres)
        idx_picco_pres_inizio = np.argmax(ciclo_pres[:finestra_inizio])
        # if idx_picco_pres_inizio != idx_picco_pres:
        #     idx_picco_pres_inizio = idx_picco_pres
        #     finestra_inizio = idx_picco_pres_inizio + 3
        picco_pressione_iniziale = ciclo_pres[idx_picco_pres_inizio]
        
        ha_overshoot_pressione = False
        
        if 0 < idx_picco_pres_inizio <= finestra_inizio:
            pendenza_salita_pres = (picco_pressione_iniziale - ciclo_pres[0]) / idx_picco_pres_inizio
            ciclo_restante = ciclo_pres[idx_picco_pres_inizio + 1 : -2]

            if np.all(np.array(ciclo_restante) < 0.95 * picco_pressione_iniziale):
                oscill, sequenze = check_consecutive_oscillations(ciclo_pres[idx_picco_pres_inizio + 1 : -2], max_scostamento=0.8, min_punti=len(ciclo_pres)//4)
                if pendenza_salita_pres and oscill:
                    ha_overshoot_pressione = True

            # if idx_picco_pres_inizio + 2 < lunghezza_ciclo:
            #     valori_post_picco_pres = ciclo_pres[idx_picco_pres_inizio + 1 : idx_picco_pres_inizio + 4]
            #     micro_crollo_pres_ok = np.any(valori_post_picco_pres < picco_pressione_iniziale)
            # else:
            #     micro_crollo_pres_ok = False
            
            # idx_fine_plateau = min(lunghezza_ciclo - 2, idx_picco_pres_inizio + 6)
            # if idx_fine_plateau > (idx_picco_pres_inizio + 1):
            #     plateau_pressione = ciclo_pres[idx_picco_pres_inizio + 1 : idx_fine_plateau]
                
            #     if len(plateau_pressione) >= 2:
            #         valore_massimo_plateau = np.max(plateau_pressione)
            #         valore_minimo_plateau = np.min(plateau_pressione)
            #         valore_medio_plateau = np.mean(plateau_pressione)
                    
            #         salita_pres_ok = pendenza_salita_pres > 1.5
            #         plateau_stabile_alto = (valore_massimo_plateau - valore_minimo_plateau) <= 1.2
            #         pressione_alta = (np.max(ciclo_pres) * 0.75) < valore_medio_plateau #< (np.max(ciclo_pres) * 0.95)
                    
            #         # --- FILTRO AGGIUNTIVO PER LA TRASLAZIONE DELLA PRESSIONE (image_3bdd3c) ---
            #         # In un vero overshoot il plateau è piatto o quasi orizzontale. 
            #         # Calcoliamo la pendenza punto-punto del plateau barico.
            #         pendenza_plateau_pres = (plateau_pressione[-1] - plateau_pressione[0]) / len(plateau_pressione)
            #         # Se la pendenza è fortemente negativa (es. < -0.15), la pressione sta crollando, non è un plateau.
            #         plateau_non_in_discesa = pendenza_plateau_pres > -0.15

                    # if salita_pres_ok and plateau_stabile_alto and pressione_alta and micro_crollo_pres_ok and plateau_non_in_discesa:
                    #     ha_overshoot_pressione = True

        # =========================================================================
        # 2. ANALISI FLUSSO (Spike unico -> Crollo -> Coda lineare pulita)
        # =========================================================================
        idx_picco_flow_assoluto = np.argmax(ciclo_flow)
        picco_flusso_iniziale = ciclo_flow[idx_picco_flow_assoluto]
        
        ha_picco_flusso_acuto = False
        
        if 0 < idx_picco_flow_assoluto <= finestra_inizio:
            idx_controllo_crollo = idx_picco_flow_assoluto + 3
            
            if idx_controllo_crollo < idx_fine_atto_attivo:
                idx_fine = min(idx_fine_atto_attivo, idx_controllo_crollo + 4)
                quota_minima_coda = max(zero_biologico_flow + 1.0, picco_flusso_iniziale * 0.35)
                valori_coda_prolungata = ciclo_flow[idx_controllo_crollo : idx_fine] #_atto_attivo]

                coda_completa = ciclo_flow[idx_controllo_crollo : idx_fine_atto_attivo]
                
                if len(valori_coda_prolungata) >= 2 and len(coda_completa) >= 4:
                    # valori plateau all'inizio devono essere sostenuti
                    coda_sostenuta_attiva = np.all(valori_coda_prolungata > quota_minima_coda)
                    
                    # --- FILTRO AGGIUNTIVO PER "TANTI PICCHI"  ---
                    # Calcoliamo quante volte il flusso risale (derivata positiva) all'interno della coda
                    differenze_coda = np.diff(coda_completa)
                    num_risalite = np.sum(differenze_coda > 0.8) # Conta i rimbalzi significativi
                    # percentuale punti di risalita
                    flusso_pulito_senza_picchi = (num_risalite / len(differenze_coda)) <= 0.25

                    # Dividiamo la coda in prima e seconda metà per la monotonicità macroscopica
                    meta_coda = len(coda_completa) // 2
                    media_inizio_coda = np.mean(coda_completa[:meta_coda])
                    media_fine_coda = np.mean(coda_completa[meta_coda:])
                    flusso_decresce_gradualmente = media_inizio_coda > (media_fine_coda + 1.0)

                    #--- FILTRO AGGIUNTIVO SUL FATTORE DI FORMA DEL FLUSSO (Fattore Cresta) ---
                    # Un vero overshoot ha un picco altissimo e isolato rispetto alla media dell'atto.
                    # Nei respiri normali larghi (come image_3bdd3c), il rapporto Picco/Media è basso.
                    rapporto_picco_media_flow = picco_flusso_iniziale / np.mean(ciclo_flow)
                    flusso_ha_cuspide_isolata = rapporto_picco_media_flow > 1.8

                    
                    if coda_sostenuta_attiva and flusso_decresce_gradualmente and flusso_pulito_senza_picchi and flusso_ha_cuspide_isolata:
                        valore_fine_crollo_flow = ciclo_flow[idx_controllo_crollo]
                        
                        pendenza_media_salita_flow = (picco_flusso_iniziale - ciclo_flow[0]) / idx_picco_flow_assoluto
                        pendenza_media_crollo_flow = (picco_flusso_iniziale - valore_fine_crollo_flow) / 3.0
                        
                        durata_coda_flow = idx_fine_atto_attivo - idx_controllo_crollo
                        valore_fine_atto_flow = ciclo_flow[idx_fine_atto_attivo]
                        pendenza_media_coda_flow = (valore_fine_crollo_flow - valore_fine_atto_flow) / durata_coda_flow
                        
                        salita_flow_ok = pendenza_media_salita_flow > 1.5
                        crollo_flow_ok = pendenza_media_crollo_flow > 0.5
                        struttura_graduale_flow = pendenza_media_crollo_flow > (1.8 * pendenza_media_coda_flow)
                        
                        if salita_flow_ok and crollo_flow_ok and struttura_graduale_flow:
                            ha_picco_flusso_acuto = True
                        
        # =========================================================================
        # 3. VERIFICA DI SIMULTANEITÀ TEMPORALE
        # =========================================================================
        coincidenza_temporale = False
        if ha_overshoot_pressione and ha_picco_flusso_acuto:
            coincidenza_temporale = abs(idx_picco_pres_inizio - idx_picco_flow_assoluto) <= 2
            
        if ha_overshoot_pressione and ha_picco_flusso_acuto and coincidenza_temporale:
            overshoot_timeline[start_m:end_m] = 1.0
            
    return overshoot_timeline

def prep_data(csv_files, file_listapnee, test=False):
    X = []
    X_geom = [] 
    y = []

    for i, file in enumerate(csv_files):
        file_name =file
        file = os.path.join('../aether/tracciati_validazione/',  file)
        print('File:', file)
        file = pd.read_csv(file)
        
        start = 30675
        end = len(file) - 2323

        file_apnee = file_listapnee[i]
        path_apnee = '../aether/Riscorati/'

        apnee_events = extract_apnee(file_apnee, path_apnee)
        print('Number apnee', len(apnee_events))

        mask_press, resp_flow, timestamps = extract_data(file, start, end)
        
        resp_flow = [float(x) for x in resp_flow]
        mask_press = [float(x) for x in mask_press]

        # 1. SEGMENTAZIONE E FEATURE GEOMETRICHE SUL SEGNALE REALE (Pre-scaling!)
        # Usiamo i cicli reali per trovare le asincronie
        resp_intervals, noise_int, aligned_mask_intervals = segmentation(mask_press, resp_flow)

        csv_path = 'Asincronie_trovate.csv'

        double_trigger_timeline = detect_double_triggering_refined(
                        aligned_mask_intervals, 
                        resp_intervals, 
                        noise_int, 
                        resp_flow, 
                        mask_press, 
                        fs=24
                    )
        print("-" * 40)
        print(f"RISULTATI ANALISI DOUBLE TRIGGER:")
        print("-" * 40)
        find_asincrony(double_trigger_timeline, timestamps, 'double trigger', csv_path, file_name)

        autotrigger_time = detect_autotriggering(aligned_mask_intervals, resp_flow, mask_press, fs=24)
        print("-" * 40)
        print(f"RISULTATI ANALISI AUTO TRIGGER:")
        print("-" * 40)
        find_asincrony(autotrigger_time, timestamps, 'auto trigger', csv_path, file_name)

        reversetrigger_time = detect_reverse_triggering(aligned_mask_intervals, resp_flow, mask_press, fs=24)
        print("-" * 40)
        print(f"RISULTATI ANALISI REVERSE TRIGGER:")
        print("-" * 40)
        find_asincrony(reversetrigger_time, timestamps, 'reverse trigger', csv_path, file_name)

        underassistance_time = detect_underassistance(aligned_mask_intervals, resp_flow, mask_press, noise_int, fs=24)
        print("-" * 40)
        print(f"RISULTATI ANALISI UNDERASSISTANCE:")
        print("-" * 40)
        find_asincrony(underassistance_time, timestamps, 'underassistance', csv_path, file_name)


        overshoot_time = detect_overshoot(aligned_mask_intervals, resp_flow, mask_press, noise_int, fs=24)
        print("-" * 40)
        print(f"RISULTATI ANALISI OVERSHOOT:")
        print("-" * 40)
        find_asincrony(overshoot_time, timestamps, 'overshoot', csv_path, file_name)

        
        # # Calcolo istantaneo delle asincronie (uso i segnali non scalati)
        # geom_features_timeline = extract_asynchrony_features(
        #     resp_flow, mask_press, resp_intervals, aligned_mask_intervals, total_length=len(timestamps), fs=24
        # )
        

        # # 2. ORA PUOI FARE LO SCALING PER IL DEEP LEARNING
        # resp_flow_scaled = normalize_signal(resp_flow)
        # mask_press_scaled = normalize_signal(mask_press)
        # signal_tot = np.column_stack((resp_flow_scaled, mask_press_scaled))

        # # 3. CREAZIONE DELLE LABEL CON SOGLIA DI OVERLAP (Es. 50%)
        # labels_tot = create_labels_with_overlap(timestamps, apnee_events, overlap_thresh=0.5)

        # # 4. WINDOWING DI FISSO (10 secondi come volevi, fs=24 -> 240 punti)
        # # Passiamo anche le feature geometriche per farle affettare insieme al segnale
        # signal_seg, geom_seg, label_seg = create_windows_multimodal(
        #     signal_tot, geom_features_timeline, labels_tot, fs=24, t_window=10, overlap=0.5
        # )

        # X.append(signal_seg)
        # X_geom_feats.append(geom_seg)
        # y.append(label_seg)


    if len(X) != 0:
        X = np.concatenate(X, axis=0)
        X_geom = np.concatenate(X_geom_feats, axis=0)
        y = np.concatenate(y, axis=0)

    return X, X_geom, y


X,y,y1,  signal_tot, pazienti =prep_data(csv_files_tot,file_listapnee_tot,test=False)
np.save('data_window_9s.npy', X)
np.save('label_window_9s.npy', y)
np.save('label_window_1_9s.npy', y1)
np.save('pazienti_id_9s.npy', pazienti)