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

def Calc_APACHE_II(visit:Visit):
    APACHE_List = []
    max_APACHE = 0 
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
    
    #Get dates within first 24 hrs of admissions
    dates = []
    lab_events = [ev for ev in lab_events if (ev.timestamp - visit_object.encounter_time).total_seconds() <= (3600*24)]

    for i in range(len(lab_events)):
        vis_date = lab_events[i].timestamp
        if (vis_date in dates) == False:
            dates.append(vis_date)
    
    #print(visit_date)
    #print('Looking for labs on ' + str(visit_date))
    age_APACHE = []
    cond_APACHE = []
    HR_AP = []
    RR_AP = []
    temp_AP = []
    A_grad_AP = []
    pO2_AP = []
    bicarb_AP = []
    MAP_AP = []
    creatinine_AP = []
    Na_AP = []
    K_AP = []
    Hemat_AP = []
    WBC_AP = []
    GCS_AP = []
    art_pH_AP = []
    for visit_date in dates:    
        
        l_events = [ev for ev in lab_events if ev.timestamp == visit_date]

        #Age
        
        age = ((visit_date - patient.birth_datetime).days)/365.0
        if age >= 45 and age <=54: 
            age_APACHE.append(2)
        if age >= 55 and age <= 64:
            age_APACHE.append(3)
        if age >= 65 and age <= 74:
            age_APACHE.append(5)
        if age >= 75:
            age_APACHE.append(6) 

        #print('age is ' + str(age) + ' and age APACHE is ' + str(age_APACHE))
        ##acute Renal Failure? If so, then double the points for creatinine
        if len(conditions) == 0:
            arf = False
        else:
            for cond in conditions:
                #(cond.code)
                if cond.code in ['5846', '5847', '5848', '5849', '586']:
                    arf = True
                    break


        #Hist of Severe Organ Insuff?
        insuff = False
        if len(conditions) == 0:
            insuff = False
        else:
            for cond in conditions:
                if cond.code in ['5712', '5715', '5716', '39891', '40201', '40211',
                                    '40291', '40401', '40403', '40411', '40413', '40491', 
                                    '40493', '4280', '4281', '42820', '42822' ,'42823', '42830',
                                        '42832', '42840', '42842' ,'42843', '4289', '10153', '99731',
                                        '49320', '49321', '49322', '51851', '51881','5180', '5181',
                                            '5182', '5183', '51852', 'V568', '42', 'V672', 'V0739', 
                                            'V662', 'V5811', '20076', '20200', '20202', '20201', 
                                            '20203', '20204', '20205', '20206', '20207', '202028', '20270' ,
                                            '20271', '20272', '20273', '20274', '20275', '20276', '20277', 
                                            '20278', '20279', '20280','20281', '20282', '20283', '20284', 
                                            '20285', '20286' ,'20287', '20288', '2319', '2321', '2322', '2323',
                                            '2324', '2325', '2326', '2327', '2328', '2329', '2330', '2331', '2332',
                                                '2333', '2334', '2335', '2336', '2337', '2338', '2339', '20310', '20312',
                                                '20400', '20401', '20402', '20410', '20411', '20412', '20420', '20421',
                                                    '20422', '20480', '20481', '20490', '20491', '20492', '20500', '20501',
                                                    '20502', '20510', '20511', '20512']:
                    insuff = True
                    break


            #For non-operative or emergency post-op patients who have organ insuff or 
            # immunocompromised, add 5. For elective post-op, add 2     
        if insuff == True:
            if len(procedures) == 0:
                cond_APACHE.append(5)
            else:
                adm_type = 'NA'
                for event in procedures:
                    HADM = event.HADM_ID
                    for adm in admissions:
                        if adm.HADM_ID == HADM:
                            adm_type = adm[6]
                            break
                if adm_type == 'EMERGENCY':
                    cond_APACHE.append(5)
                if adm_type == 'ELECTIVE':
                    cond_APACHE.append(2)
            
        #print('cond APACHE is ' + str(cond_APACHE))

        '''if len(procedures) == 0:
                adm_type = "NA"
            else:
                adm_type = ''
                for event in procedures:
                    HADM = event.HADM_ID
                    for adm in admissions:
                        if adm.HADM_ID == HADM:
                            adm_type = adm[6]'''
            #  print(adm_type)

        use_pO2 = True
        
        for event in l_events:
            if event.code == '211':
                HR = float(event.attr_dict['value'])
                if HR >= 180 or HR <= 39:
                    HR_AP.append(4)
                if (140 <= HR <= 179) or (40 <= HR <= 54):
                    HR_AP.append(3)
                if (110 <= HR <= 139) or (55 <= HR <= 69):
                    HR_AP.append(2)               
                #print('Heart Rate Detected at ' + str(HR))
            if event.code == '618':
                RR = float(event.attr_dict['value'])
                if RR >= 50 or RR <= 5:
                    RR_AP.append(4)
                if (35 <= RR <= 49):
                    RR_AP.append(3)
                if (6 <= RR <= 9):
                    RR_AP.append(2)
                if (25 <= RR <= 34) or (10 <= RR <= 11):
                    RR_AP.append(1) 
                    #print('Respiratory Rate Detected at ' + str(RR))
            if event.code == '50825':
                temp = float(event.attr_dict['value'])
                if temp >= 41 or temp <= 29.9:
                    temp_AP.append(4)
                if (39 <= temp <= 40.9) or (30 <= temp <= 31.9):
                    temp_AP.append(3)
                if (32 <= temp <= 33.9):
                    temp_AP.append(2)
                if (38.5 <= temp <= 38.9) or (34 <= temp <= 35.9):
                    temp_AP.append(1)
                #print('temp detected at ' + str(temp))
            if event.code == '50815':
                flow_found = True
                flow = float(event.attr_dict['value'])
                if flow >= 7.5:
                    use_pO2 = False
                    for event in lab_events:
                        if event.code == '50801':
                            A_grad = float(event.attr_dict['value'])
                    if A_grad >= 500:
                        A_grad_AP.append(4)
                    if (350 <= A_grad <= 499):
                        A_grad_AP.append(3)
                    if (200 <= A_grad <= 349):
                        A_grad_AP.append(2)
            if event.code == '50821':
                if use_pO2 == True:
                    pO2 = float(event.attr_dict['value'])
                    if (61 <= pO2 <= 70):
                        pO2_AP.append(1)
                    if (55 <= pO2 <= 60):
                        pO2_AP.append(3)
                    if (pO2 < 55):
                        pO2_AP.append(4)
                    #print('pO2 detected at ' + str(pO2))
            if event.code == '50882':
                bicarb = float(event.attr_dict['value'])
                if (bicarb >= 52 or bicarb < 15):
                    bicarb_AP.append(4)
                if (41 <= bicarb <= 51.9) or (15 <= bicarb <= 17):
                    bicarb_AP.append(3)
                if (18 <= bicarb <= 21.9):
                    bicarb_AP.append(2)
                if (32 <= bicarb <= 40.9):
                    bicarb_AP.append(1)   
                #print('bicarb detected at ' + str(bicarb))
            if event.code == '52':
                MAP = float(event.attr_dict['value'])
                if (MAP >= 160) or (MAP <= 49):
                    MAP_AP.append(4)
                if 130 <= MAP <= 159:
                    MAP_AP.append(3)
                if (110 <= MAP <= 129) or (50 <= MAP <= 69):
                    MAP_AP.append(2)         
                #print('MAP detected at ' + str(MAP))
            if event.code == '1126':
                art_pH = float(event.attr_dict['value'])
                if (art_pH >= 7.7) or (art_pH < 7.15):
                    art_pH_AP.append(4)
                if (7.6 <= art_pH <= 7.69) or (7.15 <= art_pH <= 7.24):
                    art_pH_AP.append(3)
                if (7.25 <= art_pH <= 7.32):
                    art_pH_AP.append(2)
                if (7.5 <= art_pH <= 7.59):
                    art_pH_AP.append(1)
                #print('Arterial pH detected at ' + str(art_pH))
            if event.code == '50983':
                Na = float(event.attr_dict['value'])
                if (Na >= 180) or (Na <= 110):
                    Na_AP.append(4)
                if (160 <= Na <= 179) or (111 <= Na <= 119):
                    Na_AP.append(3)
                if (155 <= Na <= 159) or (120 <= Na <= 129):
                    Na_AP.append(2)
                if (150 <= Na <= 154):
                    Na_AP.append(1)
                #print('Na detected at ' + str(Na))
            if event.code == '50971':             
                K = float(event.attr_dict['value'])
                if (K >= 7) or (K < 2.5):
                    K_AP.append(4)
                if (6 <= K <= 6.9):
                    K_AP.append(3)
                if (2.5 <= K <= 2.9):
                    K_AP.append(2)
                if (5.5 <= K <= 5.9) or (3 <= K <= 3.4):
                    K_AP.append(1)
                #print('K detected at ' + str(K) + ' at' + str(event.timestamp))
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
                    creatinine_AP.append(2.0*add)
                else:
                    creatinine_AP.append(add)
                    #print('creatinine detected at ' + str(creatinine))
            if event.code == '50810':
                Hemat = float(event.attr_dict['value'])
                if (Hemat >= 60) or (Hemat < 20):
                    Hemat_AP.append(4)
                if (50 <= Hemat <= 59.9) or (20 <= Hemat <= 29.9):
                    Hemat_AP.append(2)
                if (46 <= Hemat <= 46.9):
                    Hemat_AP.append(1)
                #print('Hematocrit detected at ' + str(Hemat))
            if event.code == '1127':
                WBC = float(event.attr_dict['value'])
                if (WBC >= 40) or (WBC < 1):
                    WBC_AP.append(4)
                if (20 <= WBC <= 39.9) or (1 <= WBC <= 2.9):
                    WBC_AP.append(2)
                if (15 <= WBC <= 19.9):
                    WBC_AP.append(1)
                #print('WBC detected at ' + str(WBC))
            if event.code == '198':
                GCS = float(event.attr_dict['value'])
                GCS_AP.append(15-GCS)
                #print('GCS detected at ' + str(GCS))
        #APACHE_List.append(APACHE)
    
    #APACHE score will be according to the worst value of each physiological measure
    values_found = [age_APACHE, cond_APACHE, HR_AP, RR_AP, temp_AP, A_grad_AP, pO2_AP, bicarb_AP, creatinine_AP, Na_AP, K_AP, Hemat_AP, WBC_AP, GCS_AP, art_pH_AP]
    values_found = [el for el in values_found if len(el) > 0]
    
    #print(str(len(values_found))+ ' APACHE values were found')
    #print(values_found) 
    for li in values_found:
        #print(max(li))
        max_APACHE+=max(li)
    return max_APACHE
    #max_APACHE = sum(max(HR_AP)  + max(RR_AP) + max(temp_AP)  +  max(A_grad_AP) + max(pO2_AP) + max(bicarb_AP) + max(creatinine_AP) + max(Na_AP) + max(K_AP) + max(Hemat_AP) + max(WBC_AP) + max(GCS_AP) + max(art_pH_AP)) 
    
patient = dataset.patients['43746']
p_dict = {'43746': patient}
visit_dict = patient.visits
visit = list(visit_dict)[len(visit_dict)-1]
visit_object = visit_dict[visit]

def Patient_APACHE_II(patient:Patient):
    samples = []
    for visit in patient:
        '''conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue'''
        samples.append({'visit_id':visit.visit_id, 'APACHE_II_Score':Calc_APACHE_II(visit)})
    return samples


print(Patient_APACHE_II(patient))
#lab_events = visit_object.get_event_list("LABEVENTS")

#visit_date = lab_events[1].timestamp

#print((Calc_APACHE_II(visit_object)))








    #events = visit.get_event_list
    #visit = patient_object.
    
    
    
'''for visit in patient_object:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")'''
        
        #procedures = visit.get_code_list(table="PROCEDURES_ICD")

       # MAP = 
'''Temp = 
        
        HR = 
        RR = 
        pO2 = 
        Bicarb = 
        pH = '''
#    print(patient.visit)
