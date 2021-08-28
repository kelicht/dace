# DACE: Distribution-Aware Counterfactual Explanation

Temporary repository for our paper: ["*DACE: Distribution-Aware Counterfactual Explanation by Mixed-Integer Linear Optimization*,"](https://www.ijcai.org/Proceedings/2020/395) IJCAI-20.

DACE is a framework of Counterfactual Explanation (CE) that optimizes the following objective function: 
C(a | x) = d_M(x, x+a) + alpha * q_1(x+a), 
where
* a: action for the given input instance x
* d_M(x, x+a): l1-Mahalanobis distance (l1-MD)
* q_1(x+a): 1-Local Outlier Factor (1-LOF)
* alpha:  Trade-off parameter between l1-MD and 1-LOF

If you want to use DACE, please set the following parameters:
* cost_type = 'DACE' (utilize l1-MD as the basic cost)
* alpha > 0  (Trade-off parameter between l1-MD and 1-LOF)



## Usage

Basic running examples are provided in `example.py`:
```
$ python example.py

==================== TLPS ====================
* Action (Default: Yes -> No):
	* duration_in_month: 12 -> 4 (-8)
	* age: 23 -> 45 (+22)
	* job: skilled_employee/official -> unemployed/unskilled-non-resident
* Scores: 
	* Time: 0.06543192
	* alpha: 0.00000000
	* Objective: 81.17015841
	* Cost (TLPS): 81.17015841
	* Mahalanobis: 8.34391339
	* 10-LOF: 1.19636343

==================== MAD ====================
* Action (Default: Yes -> No):
	* purpose: (vacation-does_not_exist?) -> retraining
* Scores: 
	* Time: 0.03359125
	* alpha: 0.00000000
	* Objective: 0.46314260
	* Cost (MAD): 0.46314260
	* Mahalanobis: 18.88119622
	* 10-LOF: 1.13466846

==================== PCC ====================
* Action (Default: Yes -> No):
	* other_debtors: none -> guarantor
	* job: skilled_employee/official -> unemployed/unskilled-non-resident
* Scores: 
	* Time: 0.06525796
	* alpha: 0.00000000
	* Objective: 0.07530073
	* Cost (PCC): 0.07530073
	* Mahalanobis: 9.98328166
	* 10-LOF: 1.13634779

==================== DACE ====================
* Action (Default: Yes -> No):
	* duration_in_month: 12 -> 5 (-7)
	* installment_as_income_perc: 4 -> 2 (-2)
	* age: 23 -> 29 (+6)
* Scores: 
	* Time: 2.85304871
	* alpha: 0.01000000
	* Objective: 8.97257815
	* Cost (l1-Mahal.): 8.96435471
	* Cost (LOF): 0.82234381
	* Mahalanobis: 2.27827220
	* 10-LOF: 1.09590389
```


Quick examples for each classifier can be ran as follows:

### LogisticRegression classifier (`linear_ce.py`)
```
$ python linear_ce.py 3

# Demonstration: linear_ce.py
- Dataset: FICO HELOC Dataset
- Classifier: l2-Regularized Logistic Regression Classifier

# 1st Individual with high risk of default
* Action (RiskPerformance: Bad -> Good):
	* ExternalRiskEstimate: 70 -> 80 (+10)
	* NetFractionRevolvingBurden: 70 -> 4 (-66)
	* NumBank2NatlTradesWHighUtilization: 2 -> 0 (-2)
* Scores: 
	* Time: 9.92181595
	* alpha: 0.01000000
	* Objective: 8.11828936
	* Cost (l1-Mahal.): 8.10626566
	* Cost (LOF): 1.20237024
	* Mahalanobis: 2.76675010
	* 10-LOF: 1.19098847

# 2nd Individual with high risk of default
* Action (RiskPerformance: Bad -> Good):
	* ExternalRiskEstimate: 69 -> 78 (+9)
	* MSinceOldestTradeOpen: 155 -> 292 (+137)
	* AverageMInFile: 81 -> 134 (+53)
* Scores: 
	* Time: 14.56293435
	* alpha: 0.01000000
	* Objective: 7.25777212
	* Cost (l1-Mahal.): 7.21427372
	* Cost (LOF): 4.34983913
	* Mahalanobis: 2.26445112
	* 10-LOF: 1.35521965

# 3rd Individual with high risk of default
* Action (RiskPerformance: Bad -> Good):
	* ExternalRiskEstimate: 59 -> 84 (+25)
	* MaxDelq2PublicRecLast12M: 4 -> 8 (+4)
	* MaxDelqEver: 6 -> 8 (+2)
* Scores: 
	* Time: 18.00154517
	* alpha: 0.01000000
	* Objective: 11.94607759
	* Cost (l1-Mahal.): 11.85384015
	* Cost (LOF): 9.22374429
	* Mahalanobis: 4.39399600
	* 10-LOF: 1.39338479
```

### MLP classifier (`mlp_ce.py`)
```
$ python mlp_ce.py 3

# Demonstration: mlp_ce.py
- Dataset: Diabetes Dataset
- Classifier: Multi-Layer Perceptron Classifier

# 1st Individual with high risk of diabetes
* Action (DiseaseProgression: Bad -> Good):
	* BMI: 35.0000 -> 29.8580 (-5.1420)
	* Lamotrigine: 4.0431 -> 4.0273 (-0.0158)
	* BloodSugarLevel: 91 -> 92 (+1)
* Scores: 
	* Time: 18.92564264
	* alpha: 0.01000000
	* Objective: 1.79501897
	* Cost (l1-Mahal.): 1.78394779
	* Cost (LOF): 1.10711775
	* Mahalanobis: 1.42359425
	* 10-LOF: 1.06446166

# 2nd Individual with high risk of diabetes
* Action (DiseaseProgression: Bad -> Good):
	* TCells: 194 -> 202 (+8)
	* LowDensityLipoproteins: 126.6000 -> 131.9600 (+5.3600)
	* HighDensityLipoproteins: 43.0000 -> 44.8000 (+1.8000)
* Scores: 
	* Time: 19.74549178
	* alpha: 0.01000000
	* Objective: 0.86038674
	* Cost (l1-Mahal.): 0.84653713
	* Cost (LOF): 1.38496115
	* Mahalanobis: 0.35309675
	* 10-LOF: 1.06654302

# 3rd Individual with high risk of diabetes
* Action (DiseaseProgression: Bad -> Good):
	* BMI: 28.2000 -> 25.0180 (-3.1820)
	* BloodPressure: 112.0000 -> 112.7000 (+0.7000)
	* Lamotrigine: 4.9836 -> 4.9674 (-0.0162)
* Scores: 
	* Time: 21.07739703
	* alpha: 0.01000000
	* Objective: 1.07244689
	* Cost (l1-Mahal.): 1.06348294
	* Cost (LOF): 0.89639446
	* Mahalanobis: 0.88916807
	* 10-LOF: 1.04295307
```

### RandomForest classifier (`forest_ce.py`)
```
$ python forest_ce.py

# Demonstration: linear_ce.py
- Dataset: German Dataset
- Classifier: Random Forest Classifier

# 1st Individual with high risk of default
* Action (Default: Yes -> No):
	* duration_in_month: 36 -> 14 (-22)
	* credit_amount: 15857 -> 4558 (-11299)
	* installment_as_income_perc: 2 -> 3 (+1)
* Scores: 
	* Time: 10.37286215
	* alpha: 0.01000000
	* Objective: 22.33710194
	* Cost (l1-Mahal.): 21.51082580
	* Cost (LOF): 82.62761388
	* Mahalanobis: 4.82378028
	* 10-LOF: 0.98032141

# 2nd Individual with high risk of default
* Action (Default: Yes -> No):
	* duration_in_month: 48 -> 33 (-15)
	* installment_as_income_perc: 3 -> 2 (-1)
	* age: 24 -> 30 (+6)
* Scores: 
	* Time: 99.48086417
	* alpha: 0.01000000
	* Objective: 8.96938964
	* Cost (l1-Mahal.): 8.04373529
	* Cost (LOF): 92.56543557
	* Mahalanobis: 1.96893252
	* 10-LOF: 0.81689791

# 3rd Individual with high risk of default
* Action (Default: Yes -> No):
	* credit_amount: 1842 -> 1373 (-469)
	* age: 34 -> 37 (+3)
* Scores: 
	* Time: 44.72032961
	* alpha: 0.01000000
	* Objective: 2.22400200
	* Cost (l1-Mahal.): 2.20602798
	* Cost (LOF): 1.79740260
	* Mahalanobis: 0.46215243
	* 10-LOF: 1.54871136
```


## Citation
```
@inproceedings{ijcai2020-395,
  title     = {DACE: Distribution-Aware Counterfactual Explanation by Mixed-Integer Linear Optimization},
  author    = {Kanamori, Kentaro and Takagi, Takuya and Kobayashi, Ken and Arimura, Hiroki},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Christian Bessiere},
  pages     = {2855--2862},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/395},
  url       = {https://doi.org/10.24963/ijcai.2020/395},
}
```
