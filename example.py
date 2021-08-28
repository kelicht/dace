import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from linear_ce import LinearActionExtractor
np.random.seed(0)


# Prepare Dataset and Feature Constraints (NOTE: These operations are provided by "DatasetHelper" class of utils.py)
df = pd.read_csv('./data/german_credit.csv')
feature_names = ['duration_in_month', 'credit_amount', 'installment_as_income_perc',
                 'present_res_since', 'age', 'credits_this_bank',
                 'people_under_maintenance', 'account_check_status:0<=...<200DM',
                 'account_check_status:<0DM', 'account_check_status:>=200DM/salary_assignments_for_at_least_1year',
                 'account_check_status:no_checking_account',
                 'credit_history:all_credits_at_this_bank_paid_back_duly',
                 'credit_history:critical_account/other_credits_existing_(not_at_this_bank)',
                 'credit_history:delay_in_paying_off_in_the_past',
                 'credit_history:existing_credits_paid_back_duly_till_now',
                 'credit_history:no_credits_taken/all_credits_paid_back_duly',
                 'purpose:(vacation-does_not_exist?)', 'purpose:business',
                 'purpose:car(new)', 'purpose:car(used)', 'purpose:domestic_appliances',
                 'purpose:education', 'purpose:furniture/equipment',
                 'purpose:radio/television', 'purpose:repairs', 'purpose:retraining',
                 'savings:..>=1000DM', 'savings:...<100DM', 'savings:100<=...<500DM',
                 'savings:500<=...<1000DM', 'savings:unknown/no_savings_account',
                 'present_emp_since:..>=7years', 'present_emp_since:...<1year',
                 'present_emp_since:1<=...<4years', 'present_emp_since:4<=...<7years',
                 'present_emp_since:unemployed',
                 'personal_status_sex:female_divorced/separated/married',
                 'personal_status_sex:male_divorced/separated',
                 'personal_status_sex:male_married/widowed',
                 'personal_status_sex:male_single', 'other_debtors:co-applicant',
                 'other_debtors:guarantor', 'other_debtors:none',
                 'property:if_not_A121_building_society_savings_agreement/life_insurance',
                 'property:if_not_A121/A122_car_or_other_not_in_attribute_6',
                 'property:real_estate', 'property:unknown/no_property',
                 'other_installment_plans:bank', 'other_installment_plans:none',
                 'other_installment_plans:stores', 'housing:for_free', 'housing:own', 'housing:rent',
                 'job:management/self-employed/highly_qualified_employee/officer',
                 'job:skilled_employee/official',
                 'job:unemployed/unskilled-non-resident', 'job:unskilled-resident',
                 'telephone:none', 'telephone:yes_registered_under_the_customers_name',
                 'foreign_worker:no', 'foreign_worker:yes']
feature_types = ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'B', 'B', 'B',
                 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
                 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
                 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
                 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 
                 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
feature_categories = [[7, 8, 9, 10], [11, 12, 13, 14, 15], 
                      [16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 
                      [26, 27, 28, 29, 30], [31, 32, 33, 34, 35], 
                      [36, 37, 38, 39], [40, 41, 42], 
                      [43, 44, 45, 46], [47, 48, 49], [50, 51, 52], 
                      [53, 54, 55, 56], [57, 58], [59]]
feature_constraints = ['', '', '', '', '', '', '', '', '', '', '', 
                       '', '', '', '', '', '', '', '', '', '', '', 
                       '', '', '', '', '', '', '', '', '', '', '', 
                       '', '', '', '', '', '', '', '', '', '', '', 
                       '', '', '', '', '', '', 'FIX', 'FIX', 'FIX', 
                       '', '', '', '', '', '', 'FIX', 'FIX']
target_name = 'Default'
target_labels = ['No', 'Yes']


# Train Classifier
y = df.pop(target_name).values
X = df.values
mdl = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
mdl = mdl.fit(X, y)
X_undesirable = X[mdl.predict(X)==1]
x = X_undesirable[np.random.choice(X_undesirable.shape[0])]


# Extract Action
ce = LinearActionExtractor(mdl, X, Y=y, 
                           feature_names=feature_names, 
                           feature_types=feature_types, 
                           feature_categories=feature_categories, 
                           feature_constraints=feature_constraints, 
                           target_name=target_name, 
                           target_labels=target_labels)

print('='*20 + ' TLPS ' + '='*20)
a = ce.extract(x, max_change_num=3, cost_type='TLPS', alpha=0.0)
if(a!=-1): print(a)

print('='*20 + ' MAD ' + '='*20)
a = ce.extract(x, max_change_num=3, cost_type='MAD', alpha=0.0)
if(a!=-1): print(a)

print('='*20 + ' PCC ' + '='*20)
a = ce.extract(x, max_change_num=3, cost_type='PCC', alpha=0.0)
if(a!=-1): print(a)

print('='*20 + ' DACE ' + '='*20)
a = ce.extract(x, max_change_num=3, cost_type='DACE', alpha=0.01)
if(a!=-1): print(a)
