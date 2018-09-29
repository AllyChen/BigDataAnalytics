# Big Data Analytics
1. 請以XGBoost與Gradient Boosting對於Microsoft Malware 2015 Dataset進行分析，透過參數組合分析，選擇最適合的參數組合。結果請以markdown撰寫分析，程式碼可以ipython notebook格式撰寫，請於5/20前繳交至系統。
* 請針對各個參數逐一進行分析，建議先固定某個參數，並逐步選擇最佳評估參數
* 可輔以heatmap來呈現grid search的分析結果
* 請留意binary與multi-class classification在評估上的差異 (Precision/Recall僅適合binary)
2. 加分題：
* 比較standalone與分散式運算的效能
* 應用於其他資料上(matrix須大於 10K * 100)
3. 參考資料：
* scipy-2016-sklearn-tutorial, 13. cross-validation: https://github.com/amueller/scipy-2016-sklearn/tree/master/notebooks 
* Complete Guide to Parameter Tuning in XGBoost: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 
* Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

## Homework 2 (M10515053 陳品陵)
### Implementation
- Development environment: Ipython2.7
- Dataset: https://github.com/ManSoSec/Microsoft-Malware-Challenge
- Reference : 
	- Complete Guide to Parameter Tuning in XGBoost: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 

### My Code
```  python
# Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV   #Perforing grid search.

# Load the data
train_df = pd.read_csv('./LargeTrain.csv', header=0)
target = 'Class'
# Choose all predictors except target
predictors = [x for x in train_df.columns if x not in [target]]

# initialize model
xgb_model = XGBClassifier( 
    learning_rate = 0.1,
    n_estimators = 140, 
    max_depth = 3, 
    min_child_weight = 1,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.8,         
    objective = "multi:softmax",
    nthread = 4, 
    scale_pos_weight = 1, 
    seed = 27)
``` 
#### max_depth 和 min_weight 參數調整
```  python
# Test the parameters - max_depth & min_child_weight
# Because those two parameters make much impact in the result
param_test1 = {
    'max_depth':range(3,11,2),
    'min_child_weight':list(range(3,11,2))
}

gsearch1 = GridSearchCV(
    estimator = xgb_model,
    param_grid = param_test1,
    scoring = 'accuracy',
    n_jobs = 4,
    iid = False,
    cv = 5)

gsearch1.fit(train_df[predictors],train_df[target])

# Print the output
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```
**>>Output**
([mean: 0.99678, std: 0.00096, params: {'max_depth': 3, 'min_child_weight': 3},
  mean: 0.99650, std: 0.00111, params: {'max_depth': 3, 'min_child_weight': 5},
  mean: 0.99650, std: 0.00138, params: {'max_depth': 3, 'min_child_weight': 7},
  mean: 0.99604, std: 0.00125, params: {'max_depth': 3, 'min_child_weight': 9},
  mean: 0.99669, std: 0.00089, params: {'max_depth': 5, 'min_child_weight': 3},
  mean: 0.99632, std: 0.00130, params: {'max_depth': 5, 'min_child_weight': 5},
  mean: 0.99623, std: 0.00134, params: {'max_depth': 5, 'min_child_weight': 7},
  mean: 0.99632, std: 0.00151, params: {'max_depth': 5, 'min_child_weight': 9},
  mean: 0.99678, std: 0.00071, params: {'max_depth': 7, 'min_child_weight': 3},
  mean: 0.99696, std: 0.00099, params: {'max_depth': 7, 'min_child_weight': 5},
  mean: 0.99660, std: 0.00129, params: {'max_depth': 7, 'min_child_weight': 7},
  mean: 0.99632, std: 0.00151, params: {'max_depth': 7, 'min_child_weight': 9},
  mean: 0.99660, std: 0.00080, params: {'max_depth': 9, 'min_child_weight': 3},
  mean: 0.99687, std: 0.00114, params: {'max_depth': 9, 'min_child_weight': 5},
  mean: 0.99669, std: 0.00125, params: {'max_depth': 9, 'min_child_weight': 7},
  mean: 0.99632, std: 0.00133, params: {'max_depth': 9, 'min_child_weight': 9}],
 {'max_depth': 7, 'min_child_weight': 5},
 0.99696428540599558)

**理想的 max_depth 值為 7，理想的 min_child_weight 值為 5。**

#### gamma 參數調整
```  python
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

gsearch2 = GridSearchCV(
    estimator = xgb_model, 
    param_grid = param_test2, 
    scoring='accuracy',
    n_jobs=4,
    iid=False, 
    cv=5)

gsearch2.fit(train_df[predictors],train_df[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
```
**>>Output**
([mean: 0.99696, std: 0.00099, params: {'gamma': 0.0},
  mean: 0.99669, std: 0.00118, params: {'gamma': 0.1},
  mean: 0.99650, std: 0.00090, params: {'gamma': 0.2},
  mean: 0.99669, std: 0.00102, params: {'gamma': 0.3},
  mean: 0.99641, std: 0.00114, params: {'gamma': 0.4}],
 {'gamma': 0.0},
 0.99696428540599558)
 
**理想的 gamma 值為 0。**

#### subsample 和 colsample_bytree 參數調整
```  python
param_test3 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch3 = GridSearchCV(
    estimator = xgb_model, 
    param_grid = param_test3, 
    scoring='accuracy',
    n_jobs=4,
    iid=False, 
    cv=5)
                        
gsearch3.fit(train_df[predictors],train_df[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```
**>>Output**
([mean: 0.99632, std: 0.00113, params: {'subsample': 0.6, 'colsample_bytree': 0.6},
  mean: 0.99678, std: 0.00167, params: {'subsample': 0.7, 'colsample_bytree': 0.6},
  mean: 0.99669, std: 0.00128, params: {'subsample': 0.8, 'colsample_bytree': 0.6},
  mean: 0.99669, std: 0.00121, params: {'subsample': 0.9, 'colsample_bytree': 0.6},
  mean: 0.99632, std: 0.00112, params: {'subsample': 0.6, 'colsample_bytree': 0.7},
  mean: 0.99660, std: 0.00118, params: {'subsample': 0.7, 'colsample_bytree': 0.7},
  mean: 0.99678, std: 0.00082, params: {'subsample': 0.8, 'colsample_bytree': 0.7},
  mean: 0.99650, std: 0.00122, params: {'subsample': 0.9, 'colsample_bytree': 0.7},
  mean: 0.99604, std: 0.00138, params: {'subsample': 0.6, 'colsample_bytree': 0.8},
  mean: 0.99669, std: 0.00079, params: {'subsample': 0.7, 'colsample_bytree': 0.8},
  mean: 0.99696, std: 0.00099, params: {'subsample': 0.8, 'colsample_bytree': 0.8},
  mean: 0.99650, std: 0.00090, params: {'subsample': 0.9, 'colsample_bytree': 0.8},
  mean: 0.99604, std: 0.00150, params: {'subsample': 0.6, 'colsample_bytree': 0.9},
  mean: 0.99660, std: 0.00138, params: {'subsample': 0.7, 'colsample_bytree': 0.9},
  mean: 0.99669, std: 0.00121, params: {'subsample': 0.8, 'colsample_bytree': 0.9},
  mean: 0.99650, std: 0.00103, params: {'subsample': 0.9, 'colsample_bytree': 0.9}],
 {'colsample_bytree': 0.8, 'subsample': 0.8},
 0.99696428540599558)
 
**理想的 colsample_bytree 值為 0.8，理想的 subsample 值為 0.8。**

#### reg_alpha 參數調整
```  python
param_test4 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

gsearch4 = GridSearchCV(
    estimator = xgb_model, 
    param_grid = param_test4, 
    scoring='accuracy',
    n_jobs=4,
    iid=False, 
    cv=5)

gsearch4.fit(train_df[predictors],train_df[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```
**>>Output**
([mean: 0.99696, std: 0.00099, params: {'reg_alpha': 1e-05},
  mean: 0.99687, std: 0.00125, params: {'reg_alpha': 0.01},
  mean: 0.99669, std: 0.00089, params: {'reg_alpha': 0.1},
  mean: 0.99660, std: 0.00125, params: {'reg_alpha': 1},
  mean: 0.98749, std: 0.00262, params: {'reg_alpha': 100}],
 {'reg_alpha': 1e-05},
 0.99696428540599558)
 
**理想的 reg_alpha 值為 1e-05。**
