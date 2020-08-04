import numpy as np
import pandas as pd
np.random.seed(0)


def example(clf='rf', N=3):
    from LinearCE import LinearCEExtractor
    from EnsembleCE import TreeEnsembleCEExtractor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    print('# Loading German-credit dataset ...')
    df = pd.read_csv('./data/german_credit.csv', dtype='float')
    X = df.drop(['Default'], axis=1).values
    y = df['Default'].values
    features = ['duration_in_month', 'credit_amount', 'installment_as_income_perc', 'present_res_since', 'age', 'credits_this_bank', 'people_under_maintenance', 
                'account_check_status:0_<=_..._<_200_DM', 'account_check_status:<_0_DM', 'account_check_status:>=_200_DM_/_salary_assignments_for_at_least_1_year', 
                'account_check_status:no_checking_account', 'credit_history:all_credits_at_this_bank_paid_back_duly', 
                'credit_history:critical_account/_other_credits_existing_(not_at_this_bank)', 'credit_history:delay_in_paying_off_in_the_past', 
                'credit_history:existing_credits_paid_back_duly_till_now', 'credit_history:no_credits_taken/_all_credits_paid_back_duly', 
                'purpose:(vacation_-_does_not_exist?)', 'purpose:business', 'purpose:car_(new)', 'purpose:car_(used)', 'purpose:domestic_appliances', 
                'purpose:education', 'purpose:furniture/equipment', 'purpose:radio/television', 'purpose:repairs', 'purpose:retraining', 
                'savings:.._>=_1000_DM_', 'savings:..._<_100_DM', 'savings:100_<=_..._<_500_DM', 'savings:500_<=_..._<_1000_DM_', 'savings:unknown/_no_savings_account', 
                'present_emp_since:.._>=_7_years', 'present_emp_since:..._<_1_year_', 'present_emp_since:1_<=_..._<_4_years', 'present_emp_since:4_<=_..._<_7_years', 
                'present_emp_since:unemployed', 'personal_status_sex:female_:_divorced/separated/married', 'personal_status_sex:male_:_divorced/separated', 
                'personal_status_sex:male_:_married/widowed', 'personal_status_sex:male_:_single', 'other_debtors:co-applicant', 'other_debtors:guarantor', 'other_debtors:none', 
                'property:if_not_A121_:_building_society_savings_agreement/_life_insurance', 'property:if_not_A121/A122_:_car_or_other,_not_in_attribute_6', 
                'property:real_estate', 'property:unknown_/_no_property', 'other_installment_plans:bank', 'other_installment_plans:none', 'other_installment_plans:stores', 
                'housing:for_free', 'housing:own', 'housing:rent', 'job:management/_self-employed/_highly_qualified_employee/_officer', 'job:skilled_employee_/_official', 
                'job:unemployed/_unskilled_-_non-resident', 'job:unskilled_-_resident', 'telephone:none', 'telephone:yes,_registered_under_the_customers_name_', 
                'foreign_worker:no', 'foreign_worker:yes']
    types = ['N', 'N', 'N', 'N', 'N:FIX', 'N', 'N', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B:FIX', 'B:FIX', 'B:FIX', 'B', 
             'B', 'B', 'B', 'B', 'B', 'B:FIX', 'B:FIX']
    categories = [[7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25], [26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39], 
                  [40, 41, 42], [43, 44, 45, 46], [47, 48, 49], [50, 51, 52], [53, 54, 55, 56], [57, 58], [59, 60]]

    print('# Fitting classifier ({}) ...'.format('Logistic Regression' if clf=='lr' else 'Random Forest'))
    X_tr, X_ts, y_tr, y_ts = train_test_split(X,y)
    mdl = LogisticRegression(C=1.0, penalty='l2', solver='liblinear') if clf=='lr' else RandomForestClassifier(n_estimators=50, max_depth=6)
    mdl = mdl.fit(X_tr, y_tr)
    print('# Test score: {}\n'.format(mdl.score(X_ts, y_ts)))
    denied_individual = [i for i in np.where(mdl.predict(X_ts)==1)[0] if y_ts[i]==1]
    K = 6

    for i in range(N):
        x = X_ts[denied_individual[i]]
        print('---------- DACE for {}-th denied individual ----------'.format(denied_individual[i]))
        cee = LinearCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories) if clf=='lr' else TreeEnsembleCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=True,alpha=0.1)
        if(action!=None): print(cee.explanation_)


def compare_linear(name='g', N=1):
    from LinearCE import LinearCEExtractor
    from sklearn.linear_model import LogisticRegression
    from utils import Database

    S = Database(filename=name)
    X, X_ts, y, y_ts, features, types, categories = S.get_dataset(split=True)
    mdl = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=0)
    mdl = mdl.fit(X, y)
    denied_individual = [i for i in np.where(mdl.predict(X_ts)==1)[0] if y_ts[i]==1]

    N = np.min([N, len(denied_individual)])
    K = 6
    for i in range(N):
        x = X_ts[denied_individual[i]]
        print('---------- PCC ----------')
        cee = LinearCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=False,alpha=-1,weight_type='PCC')
        if(action!=None): print(cee.explanation_)

        print('---------- TLPS ----------')
        cee = LinearCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=False,alpha=-1,weight_type='TLPS')
        if(action!=None): print(cee.explanation_)

        print('---------- MAD ----------')
        cee = LinearCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=False,alpha=-1,weight_type='MAD')
        if(action!=None): print(cee.explanation_)

        print('---------- Proposed (alp. = 0.1) ----------')
        cee = LinearCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=True,alpha=0.1)
        if(action!=None): print(cee.explanation_)

        print('-----------------------------------------------------------------')


def compare_ensemble(name='g', N=1, T=50, H=4):
    from EnsembleCE import TreeEnsembleCEExtractor
    from sklearn.ensemble import RandomForestClassifier
    from utils import Database

    S = Database(filename=name)
    X, X_ts, y, y_ts, features, types, categories = S.get_dataset(split=True)
    mdl = RandomForestClassifier(n_estimators=T, max_depth=H, random_state=0)
    mdl = mdl.fit(X, y)
    denied_individual = [i for i in np.where(mdl.predict(X_ts)==1)[0] if y_ts[i]==1]

    K = 6
    N = np.min([N,len(denied_individual)])
    for i in range(N):
        x = X_ts[denied_individual[i]]
        print('---------- PCC ----------')
        cee = TreeEnsembleCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=False,alpha=-1,weight_type='PCC')
        if(action!=None): print(cee.explanation_)

        print('---------- TLPS ----------')
        cee = TreeEnsembleCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=False,alpha=-1,weight_type='TLPS')
        if(action!=None): print(cee.explanation_)

        print('---------- MAD ----------')
        cee = TreeEnsembleCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=False,alpha=-1,weight_type='MAD')
        if(action!=None): print(cee.explanation_)

        print('---------- Proposed (alp. = 0.01) ----------')
        cee = TreeEnsembleCEExtractor(mdl,X=X,Y=y,FeatureNames=features,FeatureTypes=types,Categories=categories)
        action = cee.extract(x,max_mod=K,interaction=True,alpha=0.01)
        if(action!=None): print(cee.explanation_)

        print('-----------------------------------------------------------------')


if(__name__ == '__main__'):
    from sys import argv
    if(len(argv)<2):
        print('[Ex] python example.py lr')
        print('[Ex] python example.py rf')
    elif(argv[1] not in ['rf', 'lr']):
        print('[Ex] python example.py lr')
        print('[Ex] python example.py rf')
    else:
        example(clf=argv[1])

