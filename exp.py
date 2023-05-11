import numpy as np
import pandas as pd
from linear_ce import LinearActionExtractor
from mlp_ce import MLPActionExtractor
from forest_ce import ForestActionExtractor


def exp_sens(N=50, dataset='h', K=4, time_limit=600):
    np.random.seed(0)

    print('# Sensitivity')
    from utils import DatasetHelper
    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    print('* Dataset: ', D.dataset_name)
    print('* Feature: ', X_tr.shape[1])

    from sklearn.linear_model import LogisticRegression
    mdl = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    mdl = mdl.fit(X_tr, y_tr)
    ce = LinearActionExtractor(mdl, X_tr, Y=y_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                    feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    print('* Model: LR')
    print('* Test Score: ', mdl.score(X_ts, y_ts))
    print()
    denied = X_ts[mdl.predict(X_ts)==1]

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    dict_sens = {'Mahalanobis':{0.001:[], 0.01:[], 0.1:[], 1.0:[], 10.0:[], 100.0:[]}, '10-LOF':{0.001:[], 0.01:[], 0.1:[], 1.0:[], 10.0:[], 100.0:[]}, 'Time':{0.001:[], 0.01:[], 0.1:[], 1.0:[], 10.0:[], 100.0:[]}}
    for n,x in enumerate(denied[:N]):
        print('## {}-th Denied Individual '.format(n+1))
        print('### Cost: DACE')
        for alpha in alphas:
            print('#### alpha = {}'.format(alpha))
            a = ce.extract(x, max_change_num=K, cost_type='DACE', alpha=alpha, time_limit=time_limit)
            if(a!=-1): 
                print(a)
                for key in dict_sens.keys(): dict_sens[key][alpha].append(a.scores_[key])
            else:
                for key in dict_sens.keys(): dict_sens[key][alpha].append(-1)

    for key in dict_sens.keys():
        pd.DataFrame(dict_sens[key]).to_csv('./res/sens/{}_{}.csv'.format(D.dataset_name, key), index=False)


def exp_compare(N=1, dataset='h', model='LR', K=4, time_limit=600):
    np.random.seed(0)

    print('# Comparison')
    from utils import DatasetHelper
    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    print('* Dataset: ', D.dataset_name)
    print('* Feature: ', X_tr.shape[1])

    print('* Model: ', model)
    if(model=='LR'):
        from sklearn.linear_model import LogisticRegression
        mdl = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
        print('* C: ', mdl.C)
        mdl = mdl.fit(X_tr, y_tr)
        ce = LinearActionExtractor(mdl, X_tr, Y=y_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                        feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    elif(model=='MLP'):
        from sklearn.neural_network import MLPClassifier
        mdl = MLPClassifier(hidden_layer_sizes=(200,), max_iter=500, activation='relu', alpha=0.0001)
        print('* T: ', mdl.hidden_layer_sizes)
        mdl = mdl.fit(X_tr, y_tr)
        ce = MLPActionExtractor(mdl, X_tr, Y=y_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                       feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    elif(model=='RF'):
        from sklearn.ensemble import RandomForestClassifier
        mdl = RandomForestClassifier(n_estimators=100, max_depth=8 if dataset=='o' else 4)
        print('* T: ', mdl.n_estimators)
        print('* depth: ', mdl.max_depth)
        mdl = mdl.fit(X_tr, y_tr)
        ce = ForestActionExtractor(mdl, X_tr, Y=y_tr, feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                       feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    print('* Test Score: ', mdl.score(X_ts, y_ts))
    print()
    denied = X_ts[mdl.predict(X_ts)==1]

    dict_comp = {'TLPS':{'Mahalanobis':[], '10-LOF':[], 'Time':[]}, 'MAD':{'Mahalanobis':[], '10-LOF':[], 'Time':[]}, 'PCC':{'Mahalanobis':[], '10-LOF':[], 'Time':[]}}
    alphas = [0.01, 0.1, 1.0]
    dict_dace = {0.01:{'Mahalanobis':[], '10-LOF':[], 'Time':[]}, 0.1:{'Mahalanobis':[], '10-LOF':[], 'Time':[]}, 1.0:{'Mahalanobis':[], '10-LOF':[], 'Time':[]}}

    for n,x in enumerate(denied[:N]):
        print('## {}-th Denied Individual '.format(n+1))
        for cost in ['TLPS', 'MAD', 'PCC']:
            print('### Cost: ', cost)
            a = ce.extract(x, max_change_num=K, cost_type=cost, time_limit=time_limit)
            if(a!=-1): 
                print(a)
                for key in dict_comp[cost].keys(): dict_comp[cost][key].append(a.scores_[key])
            else:
                for key in dict_comp[cost].keys(): dict_comp[cost][key].append(-1)

        print('### Cost: DACE')
        for alpha in alphas:
            print('#### alpha = {}'.format(alpha))
            a = ce.extract(x, max_change_num=K, cost_type='DACE', alpha=alpha, time_limit=time_limit)
            if(a!=-1): 
                print(a)
                for key in dict_dace[alpha].keys(): dict_dace[alpha][key].append(a.scores_[key])
            else:
                for key in dict_dace[alpha].keys(): dict_dace[alpha][key].append(-1)

    for key in dict_comp.keys():
        pd.DataFrame(dict_comp[key]).to_csv('./res/{}/{}_{}.csv'.format(model, D.dataset_name, key), index=False)
    for key in dict_dace.keys():
        pd.DataFrame(dict_dace[key]).to_csv('./res/{}/{}_DACE_{}.csv'.format(model, D.dataset_name, key), index=False)
    

if(__name__ == '__main__'):
    N = 50
    time_limit = 600

    for model in ['LR', 'MLP', 'RF']:
        for dataset in ['g', 'h', 'w', 'd']:
            exp_compare(N=N, dataset=dataset, model=model, K=4, time_limit=time_limit)

    for dataset in ['g', 'h', 'w', 'd']:
        exp_sens(N=50, dataset=dataset, K=4, time_limit=time_limit)

