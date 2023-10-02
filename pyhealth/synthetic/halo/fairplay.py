import os
import itertools
import pickle
from collections import Counter
import pandas as pd
import numpy as np
import xgboost
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import discriminant_analysis, ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm

class DeepNetwork(nn.Module):
    def __init__(self, input_dim):
        super(DeepNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x
    
    def fit(self, X, y, epochs=10, batch_size=32, lr=0.001):
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create DataLoader for mini-batch processing
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Training loop
        for _ in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()  # Reset gradients
                y_pred = self(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
            
    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred = self(X_tensor)
            return y_pred.numpy().squeeze() > 0.5
        
    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred = self(X_tensor)
            probs = torch.cat((1 - y_pred, y_pred), dim=1)
            return probs.numpy()

basedir = '/home/bpt3/code/PyHealth/pyhealth/synthetic/halo/temp'
MIN_THRESHOLD = 50
    
def reverse_full_label_fn(label_vec):
    mortality_idx = label_vec[:1]
    age_idx = label_vec[1:4]
    gender_idx = label_vec[4:7]
    ethnicity_idx = label_vec[7:]
    return {
        'mortality': 1 if mortality_idx[0] == 1 else 0,
        'age': 'Pediatric' if age_idx[0] == 1 else 'Adult' if age_idx[1] == 1 else 'Geriatric',
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
        'age': 'Pediatric' if age_idx[0] == 1 else 'Adult' if age_idx[1] == 1 else 'Geriatric'
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
    
reverse_label_fn = reverse_mortality_label_fn
synthetic_data_name = 'synthetic_mortality_data'
experiment_name = 'mortality'
num_folds = 5

def run_experiments(train_data, test_data, codeToIndex, groups):

    def getInput(data, groups):
        X = np.zeros((len([v for p in data for v in p['visits']]), len(codeToIndex)))
        y = np.zeros((len([v for p in data for v in p['visits']])))
        demographics = {g: np.array([p['label'][g] for p in data for v in p['visits']]) for g in groups}
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
        DeepNetwork(len(codeToIndex)),
        neural_network.MLPClassifier(),
        neighbors.KNeighborsClassifier(),
        xgboost.XGBClassifier(),
        xgboost.XGBRFClassifier()
        # ensemble.AdaBoostClassifier(),
        # ensemble.GradientBoostingClassifier(),
        # linear_model.SGDClassifier(loss='hinge', learning_rate='optimal', early_stopping=True, fit_intercept=1),
        # svm.SVC(probability=True, gamma='scale'),
        # discriminant_analysis.LinearDiscriminantAnalysis(),
        # discriminant_analysis.QuadraticDiscriminantAnalysis(),
        # ensemble.BaggingClassifier(),
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
        classification_metrics = [(metrics.f1_score, True), (metrics.roc_auc_score, False)]

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
    
    # separate predictors
    x_train, y_train, _ = getInput(train_data, groups)
    x_test, y_test, demographics_test = getInput(test_data, groups)
    
    results = compete(algorithms, x_train, y_train, x_test, y_test, demographics_test, groups)
    print(results)
    return results

if __name__ == "__main__":
    real_results = []
    synthetic_results = []
    training_demographics = []
    for fold in tqdm(range(num_folds), desc='Evaluation Folds'):
        real_train = pickle.load(open(f'{basedir}/train_data_{fold}.pkl', 'rb')) + pickle.load(open(f'{basedir}/eval_data_{fold}.pkl', 'rb'))
        real_test = pickle.load(open(f'{basedir}/test_data_{fold}.pkl', 'rb'))
        synthetic_data = pickle.load(open(f'{basedir}/{synthetic_data_name}.pkl', 'rb'))

        for p in real_train + real_test + synthetic_data:
            p['label'] = reverse_label_fn(p['label'])

        groups = [(g, list(set([p['label'][g] for p in real_train + real_test + synthetic_data]))) for g in real_train[0]['label'].keys()]
        codeToIndex = {c: i for i, c in enumerate(set([c for p in real_train + real_test + synthetic_data for v in p['visits'] for c in v]))}

        demographics = {}
        overall_population = len(real_train)
        for category, populations in groups:
            category_proportions = {}
            for pop in populations:
                demographics[f'{pop.capitalize()} ({category.capitalize()})'] = len([p for p in real_train if p['label'][category] == pop]) / overall_population
                
        if len(groups) > 1:
            for category1, populations1 in groups:
                for pop1 in populations1:
                    pop_proportions = {}
                    overall_group_population = len([p for p in real_train if p['label'][category1] == pop1])
                    for category2, populations2 in groups:
                        category_proportions = {}
                        if category1 == category2:
                            continue

                        for pop2 in populations2:
                            category_proportions[pop2.capitalize()] = len([p for p in real_train if p['label'][category1] == pop1 and p['label'][category2] == pop2]) / overall_group_population
                        
                        pop_proportions[category2.capitalize()] = category_proportions
                    demographics[f'{pop1.capitalize()} ({category1.capitalize()}) Detailed'] = pop_proportions
                    
        training_demographics.append(demographics)

        if os.path.exists(f'{basedir}/baseline_{experiment_name}_results_{fold}.csv'):
            baseline_results = pd.read_csv(f'{basedir}/baseline_{experiment_name}_results_{fold}.csv')
        else:
            baseline_results = run_experiments(real_train, real_test, codeToIndex, groups)    
            baseline_results.to_csv(f'{basedir}/baseline_{experiment_name}_results_{fold}.csv')
        
        if os.path.exists(f'{basedir}/combined_{experiment_name}_results_{fold}.csv'):
            combined_results = pd.read_csv(f'{basedir}/combined_{experiment_name}_results_{fold}.csv')
        else:
            combined_data = real_train + synthetic_data
            combined_results = run_experiments(combined_data, real_test, codeToIndex, groups)
            combined_results.to_csv(f'{basedir}/combined_{experiment_name}_results_{fold}.csv')
            
        real_results.append(baseline_results)
        synthetic_results.append(combined_results)
        
    baseline_concat = pd.concat(real_results)
    baseline_grouped = baseline_concat.groupby('algorithm')
    baseline_df = baseline_grouped.mean()
    baseline_stderr = baseline_grouped.sem()
    for col in baseline_df.columns:
        baseline_df[col] = baseline_df[col].apply(lambda x: '{:.3f}'.format(x)) + " +/- " + baseline_stderr[col].apply(lambda x: '{:.3f}'.format(x))

    print('Baseline Results')
    print(baseline_df)
    baseline_df.to_csv(f'{basedir}/baseline_{experiment_name}_results.csv')
    
    combined_concat = pd.concat(synthetic_results)
    combined_grouped = combined_concat.groupby('algorithm')
    combined_df = combined_grouped.mean()
    combined_stderr = combined_grouped.sem()
    for col in combined_df.columns:
        combined_df[col] = combined_df[col].apply(lambda x: '{:.3f}'.format(x)) + " +/- " + combined_stderr[col].apply(lambda x: '{:.3f}'.format(x))
        
    print('Combined Results')
    print(combined_df)
    combined_df.to_csv(f'{basedir}/combined_{experiment_name}_results.csv')
    
    
    combined_training_demographics = {}
    for key in training_demographics[0].keys():
        if isinstance(training_demographics[0][key], dict):
            combined_training_demographics[key] = {}
            for middlekey in training_demographics[0][key].keys():
                combined_training_demographics[key][middlekey] = {}
                for subkey in training_demographics[0][key][middlekey].keys():
                    values = [fold[key][middlekey][subkey] for fold in training_demographics]
                    combined_training_demographics[key][middlekey][subkey] = f"{np.mean(values):.4f} +/- {np.std(values) / np.sqrt(len(values)):.4f}"
        else:
            values = [fold[key] for fold in training_demographics]
            combined_training_demographics[key] = f"{np.mean(values):.4f} +/- {np.std(values) / np.sqrt(len(values)):.4f}"

    print('Training Demographics')
    print(combined_training_demographics)
    pickle.dump(combined_training_demographics, open(f'{basedir}/{experiment_name}_training_demographics.pkl', 'wb'))