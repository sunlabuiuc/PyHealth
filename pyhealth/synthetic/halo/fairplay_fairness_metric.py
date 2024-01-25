import os
import pickle
import numpy as np
import pandas as pd

basedir = '/home/bpt3/code/PyHealth/pyhealth/synthetic/halo/temp'
methods = [
  'baseline',
  'combined',
  'downsampling',
  'separate',
  'smote',
  'upsampling'
]
experiments = [
  'age',
  'gender'  
]

for experiment in experiments:
  demographics = pickle.load(open(os.path.join(basedir, f'{experiment}_training_demographics.pkl'), 'rb'))
  demographics = {k.split(f' ({experiment.capitalize()})')[0]: float(v.split(' +/- ')[0]) for k, v in demographics.items()}
  majority = max(demographics, key=demographics.get)
  minorityGroups = [k for k in demographics.keys() if k != majority]
  minoritySum = sum(demographics.values()) - demographics[majority]
  minorityWeights = {k: demographics[k] / minoritySum for k, v in demographics.items() if k != majority}
  
  dfDict = {
    'Method': [],
    'Performance Variance': [],
    'Theil Index': [],
    'Disparate Impact (Averaged)': [],
    'Disparate Impact (Composite)': [],
    # 'Equalized Odds (Averaged)': [],
    # 'Equalized Odds (Composite)': [],
    # 'Equality of Opportunity (Averaged)': [],
    # 'Equality of Opportunity (Composite)': []
  }
  for g in minorityGroups:
    dfDict[f'Disparate Impact ({g})'] = []
  # for g in minorityGroups:
    # dfDict[f'Equalized Odds ({g})'] = []
  # for g in minorityGroups:
    # dfDict[f'Equality of Opportunity ({g})'] = []
  
  for method in methods:
    dfDict['Method'].append(method)
    df = pd.read_csv(os.path.join(basedir, f'{method}_{experiment}_results.csv'))
    df = df[df['algorithm'] == 'Average'].iloc[0]
    scores = {k.split(' (')[-1].split(')')[0]: float(v.split(' +/- ')[0]) for k, v in df.items() if k.startswith('f1_score (') and 'Average' not in k}
    # tpr_scores = {k.split(' (')[-1].split(')')[0]: float(v.split(' +/- ')[0]) for k, v in df.items() if k.startswith('recall_score (') and 'Average' not in k}
    # fpr_scores = {k.split(' (')[-1].split(')')[0]: float(v.split(' +/- ')[0]) for k, v in df.items() if k.startswith('false_positive_rate (') and 'Average' not in k}
    
    # Performance Variance
    dfDict['Performance Variance'].append(np.std(list(scores.values())))
    
    # Theil Index (GEI with alpha = 1)
    mu = np.mean(list(scores.values()))
    n = len(scores)
    ratios = np.array(list(scores.values())) / mu
    gei = np.mean(ratios * np.log(ratios))
    dfDict['Theil Index'].append(gei)
    
    # Disparate Impact
    majorityScore = scores[majority]
    for g in minorityGroups:
      dfDict[f'Disparate Impact ({g})'].append(scores[g] / majorityScore)
      
    dfDict['Disparate Impact (Averaged)'].append(np.mean([scores[g] / majorityScore for g in minorityGroups]))
    dfDict['Disparate Impact (Composite)'].append(np.sum([minorityWeights[g] * scores[g] / majorityScore for g in minorityGroups]))

    # # Equality of Opportunity
    # majorityTPR = tpr_scores[majority]
    # for g in minorityGroups:
    #   dfDict[f'Equality of Opportunity ({g})'].append(tpr_scores[g] / majorityTPR)
      
    # dfDict['Equality of Opportunity (Averaged)'].append(np.mean([tpr_scores[g] / majorityTPR for g in minorityGroups]))
    # dfDict['Equality of Opportunity (Composite)'].append(np.sum([minorityWeights[g] * tpr_scores[g] / majorityTPR for g in minorityGroups]))
    
    # # Equalized Odds
    # majorityFPR = fpr_scores[majority]
    # for g in minorityGroups:
    #   dfDict[f'Equalized Odds ({g})'].append(min(fpr_scores[g] / majorityFPR, tpr_scores[g] / majorityTPR))
      
    # dfDict['Equalized Odds (Averaged)'].append(np.mean([min(fpr_scores[g] / majorityFPR, tpr_scores[g] / majorityTPR) for g in minorityGroups]))
    # dfDict['Equalized Odds (Composite)'].append(np.sum([minorityWeights[g] * min(fpr_scores[g] / majorityFPR, tpr_scores[g] / majorityTPR) for g in minorityGroups]))
    
  results = pd.DataFrame(dfDict)
  results.to_csv(os.path.join(basedir, f'{experiment}_fairness_metrics.csv'), index=False)