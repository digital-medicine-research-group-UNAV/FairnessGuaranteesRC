import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import pickle
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


from core.rcpredictors import MonotoneThresholdOptimizer, MonotoneIndividualFlipper

dataset = "coverage"  # "acs", "coverage", "compas"
sen = "SEX" 
n_cal = 1000
delta = 0.1
fairness_criterion = "predictive_equality"  # "demographic_parity", "equal_opportunity", "predictive_equality"
mitigation = "if"


if dataset == "acs":
    y = "PINCP"
    
if dataset == "coverage":
    y = "label"

if dataset == "aof":
    y = "fraud_bool"

if dataset == "compas":
    y = "two_year_recid"
    
if dataset == "icu_diabetes":
    y = "diabetes_mellitus"
    
    
data = pd.read_csv(f"data/clean/{dataset}_val.csv")

# Change categorical features to categorical dtypes
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype("category")

with open(f'blackboxes/xgboost_{dataset}.pkl', 'rb') as f:
    rfc = pickle.load(f)
    
risk_tolrs = [0.025, 0.01]

report = {
    "alpha": [],
    "crc_unfairness": [],
    "crc_acc": [],
    "crc_precision": [],
    "crc_recall": [],
    "crc_specificity": [],
    "crc_f1": [],
    "crc_lambda": [],
    "ucb_unfairness": [],
    "ucb_acc": [],
    "ucb_precision": [],
    "ucb_recall": [],
    "ucb_specificity": [],
    "ucb_f1": [],
    "ucb_lambda": [],
}

fairness_dict = {
    "demographic_parity": "dp",
    "equal_opportunity": "eo",
    "predictive_equality": "pe",
}

for alpha in risk_tolrs:
    print(f"alpha = {alpha}")
    for i in tqdm(range(500)):
        
        report["alpha"].append(alpha)
        
        if fairness_criterion == "demographic_parity":
            g0_cal = data[data[sen]==0].sample(n=n_cal//2)
            g1_cal = data[data[sen]==1].sample(n=n_cal//2)
        elif fairness_criterion == "equal_opportunity":
            g0_cal = data[(data[sen] == 0) & (data[y] == 1)].sample(n=n_cal//2)
            g1_cal = data[(data[sen] == 1) & (data[y] == 1)].sample(n=n_cal//2)
        else:
            g0_cal = data[(data[sen] == 0) & (data[y] == 0)].sample(n=n_cal//2)
            g1_cal = data[(data[sen] == 1) & (data[y] == 0)].sample(n=n_cal//2)
    
        
        data_cal = pd.concat([g0_cal, g1_cal]).sample(frac=1)
        data_test = data.drop(data_cal.index)
        
        X_cal = data_cal.drop([y], axis=1)
        s_cal = data_cal[sen]
        y_cal = data_cal[y]
        
        X_test = data_test.drop([y], axis=1)
        s_test = data_test[sen]
        y_test = data_test[y]
        
        if dataset == "aof":
            threshold = 0.0058916043
        else :
            threshold = 0.5
        
        if mitigation == "to":
            rc = MonotoneThresholdOptimizer(rfc, theta=threshold) #0.005984779
        if mitigation == "if":
            rc = MonotoneIndividualFlipper(rfc, theta=threshold) #0.0058916043
        
        rc.find_crc_lambda(X=X_cal, s=s_cal, y=y_cal, utility_metric="accuracy", fairness_notion=fairness_criterion, alpha=alpha)
        acc, unfairness, precision, recall, specificity, f1 = rc.evaluate(X_test, s_test, y_test)
        report["crc_acc"].append(acc)
        report["crc_unfairness"].append(unfairness)
        report["crc_precision"].append(precision)
        report["crc_recall"].append(recall)
        report["crc_specificity"].append(specificity)
        report["crc_f1"].append(f1)
        report["crc_lambda"].append(rc.lambda_)

        rc.find_ucb_lambda(X=X_cal, s=s_cal, y=y_cal, utility_metric="accuracy", fairness_notion=fairness_criterion, alpha=alpha, delta=delta)
        acc, unfairness, precision, recall, specificity, f1 = rc.evaluate(X_test, s_test, y_test)
        report["ucb_acc"].append(acc)
        report["ucb_unfairness"].append(unfairness)
        report["ucb_precision"].append(precision)
        report["ucb_recall"].append(recall)
        report["ucb_specificity"].append(specificity)
        report["ucb_f1"].append(f1)
        report["ucb_lambda"].append(rc.lambda_)
        

        
        delta = str(delta).replace(".", "")    
        pd.DataFrame(report).to_csv(f"reports_targets/{dataset}_{sen}_{fairness_dict[fairness_criterion]}_{mitigation}_n{n_cal}_d{delta}.csv", index=False)
        delta = 0.1