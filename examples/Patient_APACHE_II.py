import os
#from typing import Optional, List, Dict, Tuple, Union
import pandas as pd
from pyhealth.datasets.mimic3 import MIMIC3Dataset
from pyhealth.data import Event, Visit, Patient
from datetime import date
today = date.today()

DATASET_NAME = "mimic3-demo"
ROOT = "https://storage.googleapis.com/pyhealth/mimiciii-demo/1.4/"
TABLES = ["DIAGNOSES_ICD", "LABEVENTS"]
CODE_MAPPING = {"NDC": ("ATC", {"target_kwargs": {"level": 3}})}
REFRESH_CACHE = True

dataset = MIMIC3Dataset(
    dataset_name=DATASET_NAME,
    root=ROOT,
    tables=TABLES,
    refresh_cache=True,
)

'''The following function outputs the APACHE II score of a patient for a particular visit and date.
   is a severity-of-disease classification system that is used to assess the clinical status and predict
   the mortality risk of critically ill patients. It is commonly used in intensive care units (ICUs) and
   hospital settings to provide an objective measure of a patient's health and to help guide medical 
   decision-making
   
   The following variables are used as inputs to the score: 

   Vital signs such as heart rate, blood pressure, temperature, and respiratory rate. 

   Laboratory values like arterial pH, serum sodium, serum potassium, serum creatinine, 
   and hematocrit.

   Age: The patient's age is also factored into the APACHE II score.

   Chronic health conditions: The presence of chronic health conditions can affect 
   the patient's overall health status and is considered in the scoring. '''

def Calc_APACHE_II(visit:Visit, visit_date:date):
    APACHE = 0 
    visit_object = visit
    patient = dataset.patients[visit.patient_id]
    p_dict = {visit.patient_id: patient}
    admissions_table = "ADMISSIONS"
    chart_table = "CHARTEVENTS"
    
    admissions = pd.read_csv(
            os.path.join(dataset.root, f"{admissions_table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str},
        )
        # drop records of the other patients
    admissions = admissions[admissions["SUBJECT_ID"].isin(p_dict.keys())]
        # drop rows with missing values
    admissions = admissions.dropna(subset=["SUBJECT_ID", "HADM_ID", "ADMISSION_TYPE"])
    admissions = admissions.values.tolist()
    for admit in admissions:
        if admit[1] != patient.patient_id:
            admissions.remove(admit)


    lab_events = visit_object.get_event_list("LABEVENTS")
    chart_events = visit_object.get_event_list("CHARTEVENTS")

    #combine lab and chart events for purposes of one for loop
    for event in chart_events:
        lab_events.append(event)

    #ID conditions and procedures
    conditions = visit_object.get_event_list(table="DIAGNOSES_ICD")
    procedures = visit_object.get_event_list(table="PROCEDURES_ICD")

    lab_events = [ev for ev in lab_events if ev.timestamp == visit_date]

    #Age
    age = ((visit_date - patient.birth_datetime).days)/365.0
    if age >= 45 and age <=54: 
        APACHE += 2
    if age >= 55 and age <= 64:
        APACHE += 3
    if age >= 65 and age <= 74:
        APACHE += 5
    if age >= 75:
        APACHE +=6  

    ##acute Renal Failure? If so, then double the points for creatinine
    if len(conditions) == 0:
        arf = False
    else:
        for cond in conditions:
            #(cond.code)
            if cond.code in ['5846', '5847', '5848', '5849', '586']:
                arf = True
                break


#History of Severe Organ Insufficiency?
    insuff = False
    renal_failure = False
    heart_failure = False
    resp_failure = False
    HIV = False
    chemo = False
    lymphoma = False
    leukemia = False    
    if len(conditions) == 0:
        insuff = False
    else:
        for cond in conditions:
        #Still need to add leukemia ICD9_CODE's below
        #Renal Failure
            if cond.code in ['5712', '5715', '5716', '39891', '40201', '40211', '40291', '40401', '40403', 
            '40411', '40413', '40491', '40493', 'V568']:
                renal_failure = True

            #Heart Failure
            if cond.code in ['4280', '4281', '42820', '42822' ,'42823', '42830',
            '42832', '42840', '42842' ,'42843', '4289']: 
                heart_failure = True


            #Respiratory Failure: 
            if cond.code in ['10153', '99731', '49320', '49321', '49322', '51851', 
            '51881','5180', '5181', '5182', '5183', '51852']:
                resp_failure = True
                                        
            #HIV: 
            if cond.code in ['42']:
                HIV = True

            #Chemotherapy: 
            if cond.code in ['V672', 'V0739', 'V662', 'V5811']:
                chemo = True

            #Lymphoma: 
            if cond.code in ['20076', '20200', '20202', '20201', '20203', '20204', '20205', '20206', '20207', '202028', '20270',
            '20271', '20272', '20273', '20274', '20275', '20276', '20277', '20278', '20279', '20280','20281', '20282', '20283',
            '20284', '20285', '20286' ,'20287', '20288']:
                lymphoma = True
            
            #Leukemia:
            if cond.code in ['20240', '20241', '20242', '20243', '20244', '20245', '20246', '20247', '20248', ' 20310', '20311', '20312',
                             '20400', '20401', '20402', '20410', '20411', '20412', '20420', '20421', '20422', '20480', '20481', '20482',
                             '20490', '20491', '20500', '20501', '20502', '20510', '20511', '20512', '20520', '20521', '20522', '20580',
                             '20581', '20582', '20590', '20591', '20592', '20600', '20601', '20602', '20610', '20611', '20612', '20620',
                             '20621', '20622', '20680', '20681', '20682', '20690', '20691', '20692', '20700', '20701', '20702', '20710',
                             '20711', '20712', '20720', '20721', '20722', '20780', '20781', '20782', '20800', '20801', '20802', '20810',
                             '20811', '20812', '20820', '20821', '20822', '20880', '20881', '20882', '20890', '20891', '20892', 'V1060',
                              'V1061', 'V1062', 'V1063', 'V1069']:
                leukemia = True
            
        if renal_failure or heart_failure or resp_failure or HIV or chemo or lymphoma or leukemia:
            insuff = True
            

    #For non-operative or emergency post-op patients who have organ insufficiency or 
    # immunocompromised, add 5. For elective post-op, add 2     
    if insuff == True:
        if len(procedures) == 0:
            APACHE += 5
        else:
            adm_type = 'NA'
            for event in procedures:
                HADM = event.HADM_ID
                for adm in admissions:
                    if adm.HADM_ID == HADM:
                        adm_type = adm[6]
                        break
            if adm_type == 'EMERGENCY':
                APACHE += 5
            if adm_type == 'ELECTIVE':
                APACHE += 2
    

    use_pO2 = True

    for event in lab_events:
        if event.code == '211':
            HR = float(event.attr_dict['value'])
            if HR >= 180 or HR <= 39:
                APACHE += 4
            if (140 <= HR <= 179) or (40 <= HR <= 54):
                APACHE += 3
            if (110 <= HR <= 139) or (55 <= HR <= 69):
                APACHE += 2               
        if event.code == '618':
            RR = float(event.attr_dict['value'])
            if RR >= 50 or RR <= 5:
                APACHE += 4
            if (35 <= RR <= 49):
                APACHE += 3
            if (6 <= RR <= 9):
                APACHE += 2
            if (25 <= RR <= 34) or (10 <= RR <= 11):
                APACHE += 1   
        if event.code == '50825':
            temp = float(event.attr_dict['value'])
            if temp >= 41 or temp <= 29.9:
                APACHE += 4
            if (39 <= temp <= 40.9) or (30 <= temp <= 31.9):
                APACHE += 3
            if (32 <= temp <= 33.9):
                APACHE += 2
            if (38.5 <= temp <= 38.9) or (34 <= temp <= 35.9):
                APACHE += 1
        if event.code == '50815':
            flow_found = True
            flow = float(event.attr_dict['value'])
            if flow >= 7.5:
                use_pO2 = False
                for event in lab_events:
                    if event.code == '50801':
                        A_grad = float(event.attr_dict['value'])
                if A_grad >= 500:
                    APACHE += 4
                if (350 <= A_grad <= 499):
                    APACHE += 3
                if (200 <= A_grad <= 349):
                    APACHE += 2
        if event.code == '50821':
            if use_pO2 == True:
                pO2 = float(event.attr_dict['value'])
                if (61 <= pO2 <= 70):
                    APACHE += 1
                if (55 <= pO2 <= 60):
                    APACHE += 3
                if (pO2 < 55):
                    APACHE += 4
        if event.code == '50882':
            bicarb = float(event.attr_dict['value'])
            if (bicarb >= 52 or bicarb < 15):
                    APACHE += 4
            if (41 <= bicarb <= 51.9) or (15 <= bicarb <= 17):
                APACHE += 3
            if (18 <= bicarb <= 21.9):
                APACHE += 2
            if (32 <= bicarb <= 40.9):
                APACHE += 1   
        if event.code == '52':
            MAP = float(event.attr_dict['value'])
            if (MAP >= 160) or (MAP <= 49):
                APACHE += 4
            if 130 <= MAP <= 159:
                APACHE += 3
            if (110 <= MAP <= 129) or (50 <= MAP <= 69):
                APACHE += 2         
        if event.code == '1126':
            art_pH = float(event.attr_dict['value'])
            if (art_pH >= 7.7) or (art_pH < 7.15):
                APACHE += 4
            if (7.6 <= art_pH <= 7.69) or (7.15 <= art_pH <= 7.24):
                APACHE += 3
            if (7.25 <= art_pH <= 7.32):
                APACHE += 2
            if (7.5 <= art_pH <= 7.59):
                APACHE += 1
        if event.code == '50983':
            Na = float(event.attr_dict['value'])
            if (Na >= 180) or (Na <= 110):
                APACHE += 4
            if (160 <= Na <= 179) or (111 <= Na <= 119):
                APACHE += 3
            if (155 <= Na <= 159) or (120 <= Na <= 129):
                APACHE += 2
            if (150 <= Na <= 154):
                APACHE += 1
        if event.code == '50971':
            K = float(event.attr_dict['value'])
            if (K >= 7) or (K < 2.5):
                APACHE += 4
            if (6 <= K <= 6.9):
                APACHE += 3
            if (2.5 <= K <= 2.9):
                APACHE += 2
            if (5.5 <= K <= 5.9) or (3 <= K <= 3.4):
                APACHE += 1
        if event.code == '50912':
            creatinine = float(event.attr_dict['value'])
            add = 0
            if (creatinine > 305):
                add += 4
            if (170 <= creatinine <= 304):
                add += 3
            if (130 <= creatinine <= 169) or (creatinine > 53):
                add += 2
            if arf == True:
                APACHE += 2.0*add
            else:
                APACHE += add
        if event.code == '50810':
            Hemat = float(event.attr_dict['value'])
            if (Hemat >= 60) or (Hemat < 20):
                APACHE += 4
            if (50 <= Hemat <= 59.9) or (20 <= Hemat <= 29.9):
                APACHE += 2
            if (46 <= Hemat <= 46.9):
                APACHE += 1
        if event.code == '1127':
            WBC = float(event.attr_dict['value'])
            if (WBC >= 40) or (WBC < 1):
                APACHE += 4
            if (20 <= WBC <= 39.9) or (1 <= WBC <= 2.9):
                APACHE += 2
            if (15 <= WBC <= 19.9):
                APACHE += 1
        if event.code == '198':
            GCS = float(event.attr_dict['value'])
            APACHE += (15-GCS)
    return APACHE

