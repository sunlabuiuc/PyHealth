import os
import pickle
import xgboost
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn import ensemble, linear_model, neural_network, metrics, neighbors

basedir = '/home/bpt3/code/PyHealth/pyhealth/synthetic/halo/temp'
MIN_THRESHOLD = 50
MIN_VALUE = 1000
    
def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)
    
def reverse_full_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    age_idx = label_vec[1:4]
    gender_idx = label_vec[4:7]
    ethnicity_idx = label_vec[7:]
    return {
        'mortality': 1 if mortality_idx[0] == 1 else 0,
        'age': 'Pediatric' if age_idx[0] == 1 else 'Adult' if age_idx[1] == 1 else 'Elderly',
        'gender': 'Male' if gender_idx[0] == 1 else 'Female' if gender_idx[1] == 1 else 'Other/Unknown',
        'ethnicity': 'Caucasian' if ethnicity_idx[0] == 1 else 'African American' if ethnicity_idx[1] == 1 else 'Hispanic' if ethnicity_idx[2] == 1 else 'Asian' if ethnicity_idx[3] == 1 else 'Native American' if ethnicity_idx[4] == 1 else 'Other/Unknown',
    }

def reverse_mortality_label_fn(label_vec):
    return {
        'mortality': 1 if label_vec[0] == 1 else 0,
    }

def reverse_age_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    age_idx = label_vec[1:4]
    return {
        'mortality': 1 if mortality_idx[0] == 1 else 0,
        'age': 'Pediatric' if age_idx[0] == 1 else 'Adult' if age_idx[1] == 1 else 'Elderly'
    }   

def reverse_gender_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    gender_idx = label_vec[1:4]
    return {
        'mortality': 1 if mortality_idx[0] == 1 else 0,
        'gender': 'Male' if gender_idx[0] == 1 else 'Female' if gender_idx[1] == 1 else 'Other/Unknown'
    } 

def reverse_ethnicity_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    ethnicity_idx = label_vec[1:]
    return {
        'mortality': 1 if mortality_idx[0] == 1 else 0,
        'ethnicity': 'Caucasian' if ethnicity_idx[0] == 1 else 'African American' if ethnicity_idx[1] == 1 else 'Hispanic' if ethnicity_idx[2] == 1 else 'Asian' if ethnicity_idx[3] == 1 else 'Native American' if ethnicity_idx[4] == 1 else 'Other/Unknown',
    }
    
def reverse_genderAndAge_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    age_idx = label_vec[1:4]
    gender_idx = label_vec[4:7]
    return {
        'mortality': 1 if mortality_idx[0] == 1 else 0,
        'age': 'Pediatric' if age_idx[0] == 1 else 'Adult' if age_idx[1] == 1 else 'Elderly',
        'gender': 'Male' if gender_idx[0] == 1 else 'Female' if gender_idx[1] == 1 else 'Other/Unknown',
    }

reverse_label_fn = reverse_genderAndAge_label_fn
experiment_name = 'genderAndAge'
num_folds = 5

def run_experiments(train_data, test_data, codeToIndex, groups):

    def getInput(data, groups):
        X = np.zeros((len([v for p in data for v in p['visits']]), len(codeToIndex)))
        y = np.zeros((len([v for p in data for v in p['visits']])))
        demographics = {g[0]: np.array([p['label'][g[0]] for p in data for v in p['visits']]) for g in groups}
        counter = 0
        for p in data:
            for i, v in enumerate(p['visits']):
                X[counter, [codeToIndex[c] for c in v]] = 1
                if i == len(p['visits']) - 1:
                    y[counter] = p['label']['mortality']
                counter += 1
                
        return X, y, demographics

    candidate_algorithms = [
        linear_model.LogisticRegression(solver='lbfgs', max_iter=1000),
        ensemble.RandomForestClassifier(n_estimators=100),
        neural_network.MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=1000),
        neighbors.KNeighborsClassifier(),
        xgboost.XGBClassifier(),
    ]

    algorithms = {algorithm.__class__.__name__: algorithm for algorithm in candidate_algorithms}

    def evaluate(competing_metrics, groups, algorithm_name, algorithm, x, y, demographics):
        """evaluate how an algorithm does on the provided dataset & generate a pd row"""
        preds = algorithm.predict_proba(x)[:, 1]
        rounded_preds = algorithm.predict(x)
        results = []
        
        for metric_func, binary_metric in competing_metrics:
            if binary_metric:
                metric_preds = rounded_preds
            else:
                metric_preds = preds
            results.append(metric_func(y, metric_preds))
        
        for metric_func, binary_metric in competing_metrics:
            if binary_metric:
                metric_preds = rounded_preds
            else:
                metric_preds = preds
            for category, group_names in groups:
                category_results = []
                for g in group_names:
                    y_g, preds_g = y[demographics[category] == g], metric_preds[demographics[category] == g]
                    res = metric_func(y_g, preds_g)
                    print(f'{algorithm_name} {metric_func.__name__} {category} {g}: {res}')
                    category_results.append(res)
                    results.append(res)
                results.append(np.mean(category_results))

        if len(groups) > 1:
            for metric_func, pred_type in competing_metrics:
                if pred_type == 0:
                    metric_preds = preds
                else:
                    metric_preds = rounded_preds
                for (category1, group_names1), (category2, group_names2) in itertools.combinations(groups, 2):
                    for g1 in group_names1:
                        for g2 in group_names2:
                            if len([p for p in x[(demographics[category1] == g1) & (demographics[category2] == g2)]]) < MIN_THRESHOLD:
                                continue
                            
                            y_cross = y[(demographics[category1] == g1) & (demographics[category2] == g2)]
                            preds_cross = metric_preds[(demographics[category1] == g1) & (demographics[category2] == g2)]
                            res = metric_func(y_cross, preds_cross)
                            print(f'{algorithm_name} {metric_func.__name__} ({category1}, {category2}) ({g1}, {g2}): {res}')
                            results.append(res)
        
        row = [algorithm_name] + results
        return row

    def process_results(column_names, results):
        df = pd.DataFrame(data=results, columns=column_names, index=None)
        algs = df.pop('algorithm')
        df.loc[len(df)] = df.mean()
        algs[len(algs)] = 'Average'
        df['algorithm'] = algs
        df = df[column_names]
        return df

    def compete(algorithms, x_train, y_train, x_test, y_test, demographics_test, groups):
        """Compete the algorithms"""
        classification_metrics = [(metrics.f1_score, True), (metrics.recall_score, True), (false_positive_rate, True), (metrics.roc_auc_score, False)]

        column_names = ["algorithm"]
        column_names += [f'{metric.__name__} Overall' for metric, _ in classification_metrics]
        column_names += [f'{metric.__name__} ({group})' for metric, _ in classification_metrics for group in [g for (category, populations) in groups for g in [pop for pop in populations + [f'Average {category.capitalize()}']]]]
        if len(groups) > 1:
            column_names += [f'{metric.__name__} ({group1}, {group2})' for metric, _ in classification_metrics for (category1, populations1), (category2, populations2) in itertools.combinations(groups, 2) for group1 in populations1 for group2 in populations2 if len([p for p in x_test[(demographics_test[category1] == group1) & (demographics_test[category2] == group2)]]) >= MIN_THRESHOLD]
        
        results = []
        for algorithm_name, algorithm in tqdm(algorithms.items(), desc="Competing Algorithms", total=len(algorithms)):
            algorithm.fit(x_train, y_train)
            row = evaluate(classification_metrics, groups, algorithm_name, algorithm, x_test, y_test, demographics_test)
            results.append(row)

        res = process_results(column_names, results)
        return res
    
    x_train, y_train, _ = getInput(train_data, groups)
    x_test, y_test, demographics_test = getInput(test_data, groups)
    
    results = compete(algorithms, x_train, y_train, x_test, y_test, demographics_test, groups)
    print(results)
    return results

def getSMOTE(data, allCombos, codeToIndex):
    comboMap = {tuple(sorted(c)): i for i, c in enumerate(allCombos)}
    X = np.zeros((len([v for p in data for v in p['visits']]), len(codeToIndex)))  
    y = np.zeros((len([v for p in data for v in p['visits']])))
    counter = 0
    for p in data:
        for i, v in enumerate(p['visits']):
            labels = p['label']
            labels['mortality'] = 1 if i == len(p['visits']) - 1 and p['label']['mortality'] == 1 else 0
            X[counter, [codeToIndex[c] for c in v]] = 1
            y[counter] = comboMap[tuple(sorted(labels.items()))]
            counter += 1
            
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X, y)
    indexToCode = {v: k for k, v in codeToIndex.items()}
    upsampled_data = []
    for i in range(len(X_res)):
        p = {}
        p['label'] = {g: v for (g,v) in allCombos[int(y_res[i])]}
        p['visits'] = [[indexToCode[c] for c in X_res[i].nonzero()[0]]]
        upsampled_data.append(p)

    return upsampled_data
        
        

if __name__ == "__main__":
    upsampling_results = []
    downsampling_results = []
    smote_results = []
    for fold in tqdm(range(num_folds), desc='Evaluation Folds'):
        real_train = pickle.load(open(f'{basedir}/train_{experiment_name}_data_{fold}.pkl', 'rb')) + pickle.load(open(f'{basedir}/eval_{experiment_name}_data_{fold}.pkl', 'rb'))
        real_test = pickle.load(open(f'{basedir}/test_{experiment_name}_data_{fold}.pkl', 'rb'))

        for p in real_train + real_test:
            p['label'] = reverse_label_fn(p['label'])

        groups = [(g, list(set([p['label'][g] for p in real_train + real_test]))) for g in real_train[0]['label'].keys() if g != 'mortality']
        combos = [c for c in list(itertools.combinations([(g, v) for (g, l) in groups for v in l], len(groups))) if len(set([g for (g, _) in c])) == len(groups)]
        codeToIndex = {c: i for i, c in enumerate(set([c for p in real_train + real_test for v in p['visits'] for c in v]))}

        allGroups = [(g, list(set([p['label'][g] for p in real_train + real_test]))) for g in real_train[0]['label'].keys()]
        allCombos = [c for c in list(itertools.combinations([(g, v) for (g, l) in allGroups for v in l], len(allGroups))) if len(set([g for (g, _) in c])) == len(allGroups)]

        maxComboSize = max([len([p for p in real_train if all([p['label'][g] == v for (g, v) in c])]) for c in allCombos])
        minComboSize = min([len([p for p in real_train if all([p['label'][g] == v for (g, v) in c])]) for c in allCombos if len([p for p in real_train if all([p['label'][g] == v for (g, v) in c])]) > 0])
        if minComboSize < MIN_VALUE:
            minComboSize = MIN_VALUE
            
        upsampled_train = []
        for c in allCombos:
            comboData = [p for p in real_train if all([p['label'][g] == v for (g, v) in c])]
            upsampled_train += comboData
            if 0 < len(comboData) < maxComboSize:
                upsampled_train += np.random.choice(comboData, maxComboSize-len(comboData), replace=True).tolist()
                
        downsampled_train = []
        for c in allCombos:
            comboData = [p for p in real_train if all([p['label'][g] == v for (g, v) in c])]
            if len(comboData) > 0:
                downsampled_train += comboData if len(comboData) < minComboSize else np.random.choice(comboData, minComboSize, replace=False).tolist()
                
        smote_train = getSMOTE(real_train, allCombos, codeToIndex)
        
        if os.path.exists(f'{basedir}/upsampling_{experiment_name}_results_{fold}.csv'):
            upsampling_result = pd.read_csv(f'{basedir}/upsampling_{experiment_name}_results_{fold}.csv')
        else:
            upsampling_result = run_experiments(upsampled_train, real_test, codeToIndex, groups)    
            upsampling_result.to_csv(f'{basedir}/upsampling_{experiment_name}_results_{fold}.csv')
            
        if os.path.exists(f'{basedir}/downsampling_{experiment_name}_results_{fold}.csv'):
            downsampling_result = pd.read_csv(f'{basedir}/downsampling_{experiment_name}_results_{fold}.csv')
        else:
            downsampling_result = run_experiments(downsampled_train, real_test, codeToIndex, groups)    
            downsampling_result.to_csv(f'{basedir}/downsampling_{experiment_name}_results_{fold}.csv')
            
        if os.path.exists(f'{basedir}/smote_{experiment_name}_results_{fold}.csv'):
            smote_result = pd.read_csv(f'{basedir}/smote_{experiment_name}_results_{fold}.csv')
        else:
            smote_result = run_experiments(smote_train, real_test, codeToIndex, groups)    
            smote_result.to_csv(f'{basedir}/smote_{experiment_name}_results_{fold}.csv')
            
        upsampling_results.append(upsampling_result)
        downsampling_results.append(downsampling_result)
        smote_results.append(smote_result)
        
        
        
    upsampling_concat = pd.concat(upsampling_results)
    upsampling_grouped = upsampling_concat.groupby('algorithm')
    upsampling_df = upsampling_grouped.mean()
    upsampling_stderr = upsampling_grouped.sem()
    for col in upsampling_df.columns:
        upsampling_df[col] = upsampling_df[col].apply(lambda x: '{:.3f}'.format(x)) + " +/- " + upsampling_stderr[col].apply(lambda x: '{:.3f}'.format(x))

    print('Upsampled Results')
    print(upsampling_df)
    upsampling_df.to_csv(f'{basedir}/upsampling_{experiment_name}_results.csv')
    
    
    downsampling_concat = pd.concat(downsampling_results)
    downsampling_grouped = downsampling_concat.groupby('algorithm')
    downsampling_df = downsampling_grouped.mean()
    downsampling_stderr = downsampling_grouped.sem()
    for col in downsampling_df.columns:
        downsampling_df[col] = downsampling_df[col].apply(lambda x: '{:.3f}'.format(x)) + " +/- " + downsampling_stderr[col].apply(lambda x: '{:.3f}'.format(x))

    print('Downsampled Results')
    print(downsampling_df)
    downsampling_df.to_csv(f'{basedir}/downsampling_{experiment_name}_results.csv')
    
    
    smote_concat = pd.concat(smote_results)
    smote_grouped = smote_concat.groupby('algorithm')
    smote_df = smote_grouped.mean()
    smote_stderr = smote_grouped.sem()
    for col in smote_df.columns:
        smote_df[col] = smote_df[col].apply(lambda x: '{:.3f}'.format(x)) + " +/- " + smote_stderr[col].apply(lambda x: '{:.3f}'.format(x))

    print('SMOTE Results')
    print(smote_df)
    smote_df.to_csv(f'{basedir}/smote_{experiment_name}_results.csv')