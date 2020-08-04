# DACE: Distribution-Aware Counterfactual Explanation

Temporary repository for our paper: ["*DACE: Distribution-Aware Counterfactual Explanation by Mixed-Integer Linear Optimization*,"](https://www.ijcai.org/Proceedings/2020/395) IJCAI2020.

DACE optimizes the following objective function: C(a | x) = d_M(x, x+a) + alpha * q_1(x+a), where
* d_M(x, x+a): Surrogate Mahalanobis' distance (SMD)
* q_1(x+a): 1-Local Outlier Factor (1-LOF)
* alpha:  Trade-off parameter between SMD and 1-LOF
* a: action for the given input instance x

If you want to use DACE, please set the following parameters:
* interaction = True  (Surrogate Mahalanobis' distance)
* alpha > 0  (Trade-off parameter between SMD and 1-LOF)

Running examples are provided in `example.py`.
```
$ python example.py rf
# Loading German-credit dataset ...
# Fitting classifier (Random Forest) ...
# Test score: 0.764

---------- DACE for 10-th denied individual ----------
Predicted Class: 1 => 0 (Obj.: 6.178, Time: 16.39, Mahal.: 1.273, 10-LOF: 1.033)
	 * 'duration_in_month': 36.0 => 22.0 (-14.0)
	 * 'credit_amount': 9034.0 => 7694.0 (-1340.0)

---------- DACE for 11-th denied individual ----------
Predicted Class: 1 => 0 (Obj.: 4.779, Time: 10.8, Mahal.: 1.163, 10-LOF: 0.9169)
	 * 'duration_in_month': 24.0 => 27.0 (+3.0)
	 * 'credit_amount': 3552.0 => 3616.0 (+64.0)
	 * 'present_res_since': 4.0 => 3.0 (-1.0)

---------- DACE for 25-th denied individual ----------
Predicted Class: 1 => 0 (Obj.: 2.152, Time: 4.7, Mahal.: 0.4535, 10-LOF: 1.129)
	 * 'duration_in_month': 48.0 => 43.0 (-5.0)
	 * 'credit_amount': 6224.0 => 5697.0 (-527.0)
```
```
$ python example.py lr
# Loading German-credit dataset ...
# Fitting classifier (Logistic Regression) ...
# Test score: 0.74

---------- DACE for 9-th denied individual ----------
Predicted Class: 1.0 => 0.0 (Obj.: 8.182, Deci.Func.: -0.0005178, Time: 1.034, Mahal.: 1.776, 10-LOF: 0.9751)
	 * 'duration_in_month': 48.0 => 29.0 (-19.0)
	 * 'credit_amount': 6560.0 => 3587.0 (-2973.0)

---------- DACE for 10-th denied individual ----------
Predicted Class: 1.0 => 0.0 (Obj.: 24.63, Deci.Func.: -0.01258, Time: 1.105, Mahal.: 4.502, 10-LOF: 1.035)
	 * 'duration_in_month': 36.0 => 3.0 (-33.0)
	 * 'credit_amount': 9034.0 => 6184.0 (-2850.0)
	 * 'installment_as_income_perc': 4.0 => 3.0 (-1.0)
	 * 'account_check_status': '0_<=_..._<_200_DM' => 'no_checking_account'

---------- DACE for 11-th denied individual ----------
Predicted Class: 1.0 => 0.0 (Obj.: 7.919, Deci.Func.: -0.0015, Time: 0.7499, Mahal.: 1.65, 10-LOF: 1.1)
	 * 'duration_in_month': 24.0 => 10.0 (-14.0)
	 * 'credit_amount': 3552.0 => 2103.0 (-1449.0)
	 * 'installment_as_income_perc': 3.0 => 2.0 (-1.0)
```