from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
scale_pos_weight = 2810.0/48025.0
import numpy as np
rf_parameters = {
    'min_samples_split' : hp.uniform('min_samples_split', 1e-8, 0.5),
    'min_samples_leaf' : hp.uniform('min_samples_leaf', 1e-8, 0.5),
    'max_depth': hp.choice('max_depth', [None]+list(range(1,40)) ),
    'max_features': hp.choice('max_features', range(1,7)),
    'n_estimators': hp.choice('n_estimators', [100,200,400]),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1]),
    'class_weight':hp.choice('class_weight', ['balanced', 'balanced_subsample'])
    ,'oob_score': hp.choice('oob_score', [False, True])
}

xgb_reg_params = {
	'scale_pos_weight'	 :     hp.choice('scale_pos_weight',    np.arange(0.10, 0.16, 0.01)),
    'learning_rate'			 :     hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
	'colsample_bytree'	 : 	   hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
	'subsample'				 :     hp.uniform('subsample', 0.8, 1),
	'n_estimators'			:     hp.choice('n_estimators', [100,200,400,800,1600]),
	'reg_alpha'				  :     hp.uniform('reg_alpha', 0.2,0.4),
    'max_depth'			 	:     hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight' :      hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
}
model_zoo = {
	RandomForestClassifier: rf_parameters
    # ,
    # XGBClassifier:xgb_reg_params
}
