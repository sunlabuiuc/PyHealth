import os
import pickle
import xgboost
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from sklearn import ensemble, linear_model, neural_network, metrics, neighbors

basedir = '/home/bpt3/code/PyHealth/pyhealth/synthetic/halo/temp'
MIN_THRESHOLD = 50
MIN_VALUE = 1000

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)

class SeparateClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        self.classifiers = {}
        
    def fit(self, combo, X, y):
        cls = deepcopy(self.classifier)
        cls.fit(X, y)
        self.classifiers[combo] = cls
        
    def predict(self, combo, X):
        return self.classifiers[combo].predict(X)
    
    def predict_proba(self, combo, X):
        return self.classifiers[combo].predict_proba(X)
    
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
synthetic_data_name = 'synthetic_genderAndAge_data'
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
        full_y = np.zeros(sum([len(v) for v in y.values() if len(v) > MIN_THRESHOLD]))
        preds = np.zeros(len(full_y))
        rounded_preds = np.zeros(len(full_y))
        demographics = {g: np.concatenate([demographics[c][g] for c in demographics if len(demographics[c][g]) > MIN_THRESHOLD]) for g in list(demographics.values())[0]}
        
        counter = 0
        for c in x:
            if len(y[c]) < MIN_THRESHOLD:
                continue
            
            cls_size = len(y[c])
            full_y[counter:counter+cls_size] = y[c]
            preds[counter:counter+cls_size] = algorithm.predict_proba(c, x[c])[:, 1]
            rounded_preds[counter:counter+cls_size] = algorithm.predict(c, x[c])
            counter += cls_size
        
        results = []
        for metric_func, binary_metric in competing_metrics:
            if binary_metric:
                metric_preds = rounded_preds
            else:
                metric_preds = preds
            results.append(metric_func(full_y, metric_preds))
        
        for metric_func, binary_metric in competing_metrics:
            if binary_metric:
                metric_preds = rounded_preds
            else:
                metric_preds = preds
            for category, group_names in groups:
                category_results = []
                for g in group_names:
                    y_g, preds_g = full_y[demographics[category] == g], metric_preds[demographics[category] == g]
                    if len(y_g) > 0:
                        res = metric_func(y_g, preds_g)
                        print(f'{algorithm_name} {metric_func.__name__} {category} {g}: {res}')
                        category_results.append(res)
                        results.append(res)
                    else:
                        print(f'{algorithm_name} {metric_func.__name__} {category} {g}: Unevaluated')
                        category_results.append(0)
                        results.append(0)
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
                            y_cross = full_y[(demographics[category1] == g1) & (demographics[category2] == g2)]
                            if len(y_cross) < MIN_THRESHOLD:
                                continue
                            
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
            column_names += [f'{metric.__name__} ({group1}, {group2})' for metric, _ in classification_metrics for (category1, populations1), (category2, populations2) in itertools.combinations(groups, 2) for group1 in populations1 for group2 in populations2 if tuple(sorted(((category1, group1), (category2, group2)))) in demographics_test and len(x_test[tuple(sorted(((category1, group1), (category2, group2))))] >= MIN_THRESHOLD)]
        
        results = []
        for algorithm_name, algorithm in tqdm(algorithms.items(), desc="Competing Algorithms", total=len(algorithms)):
            classifier = SeparateClassifier(algorithm)
            for c in x_train:
                classifier.fit(c, x_train[c], y_train[c])
            row = evaluate(classification_metrics, groups, algorithm_name, classifier, x_test, y_test, demographics_test)
            results.append(row)

        res = process_results(column_names, results)
        return res
    
    x_train = {}
    y_train = {}
    x_test = {}
    y_test = {}
    demographics_test = {}
    for c in train_data.keys():
        x_train_group, y_train_group, _ = getInput(train_data[c], groups)
        x_test_group, y_test_group, demographics_test_group = getInput(test_data[c], groups)
        if len(x_test_group) < MIN_VALUE:
            continue
        
        x_train[c] = x_train_group
        y_train[c] = y_train_group
        x_test[c] = x_test_group
        y_test[c] = y_test_group
        demographics_test[c] = demographics_test_group
    
    results = compete(algorithms, x_train, y_train, x_test, y_test, demographics_test, groups)
    print(results)
    return results



if __name__ == "__main__":
    separate_results = []
    for fold in tqdm(range(num_folds), desc='Evaluation Folds'):
        real_train = pickle.load(open(f'{basedir}/train_{experiment_name}_data_{fold}.pkl', 'rb')) + pickle.load(open(f'{basedir}/eval_{experiment_name}_data_{fold}.pkl', 'rb'))
        real_test = pickle.load(open(f'{basedir}/test_{experiment_name}_data_{fold}.pkl', 'rb'))

        for p in real_train + real_test:
            p['label'] = reverse_label_fn(p['label'])

        groups = [(g, list(set([p['label'][g] for p in real_train + real_test]))) for g in real_train[0]['label'].keys() if g != 'mortality']
        combos = [c for c in list(itertools.combinations([(g, v) for (g, l) in groups for v in l], len(groups))) if len(set([g for (g, _) in c])) == len(groups)]
        codeToIndex = {c: i for i, c in enumerate(set([c for p in real_train + real_test for v in p['visits'] for c in v]))}

        separate_train = {}
        for c in combos:
            comboData = [p for p in real_train if all([p['label'][g] == v for (g, v) in c])]
            if len(comboData) > 0:
                separate_train[tuple(sorted(c))] = comboData
                
        separate_test = {}
        for c in combos:
            comboData = [p for p in real_test if all([p['label'][g] == v for (g, v) in c])]
            if len(comboData) > 0:
                separate_test[tuple(sorted(c))] = comboData

        if os.path.exists(f'{basedir}/separate_{experiment_name}_results_{fold}.csv'):
            separate_result = pd.read_csv(f'{basedir}/separate_{experiment_name}_results_{fold}.csv')
        else:
            separate_result = run_experiments(separate_train, separate_test, codeToIndex, groups)    
            separate_result.to_csv(f'{basedir}/separate_{experiment_name}_results_{fold}.csv')
            
        separate_results.append(separate_result)

    separate_concat = pd.concat(separate_results)
    separate_grouped = separate_concat.groupby('algorithm')
    separate_df = separate_grouped.mean()
    separate_stderr = separate_grouped.sem()
    for col in separate_df.columns:
        separate_df[col] = separate_df[col].apply(lambda x: '{:.3f}'.format(x)) + " +/- " + separate_stderr[col].apply(lambda x: '{:.3f}'.format(x))

    print('Separate Results')
    print(separate_df)
    separate_df.to_csv(f'{basedir}/separate_{experiment_name}_results.csv')