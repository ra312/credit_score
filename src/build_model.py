#TODO: hyperopt with AUC as objective using cross-validation and performing oversampling only on partitions
#TODO: imb-xgboost 
import argparse
import os
import numpy as np
import pandas as pd
import joblib
random_state = 42
 # import scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# import feature_processing code
from feature_processing import check_data_balanced
from feature_processing import extract_transform_load
from feature_processing import global_train_parameters
from scoring_methods import  normalized_gini, new_gini_scorer
from feature_processing import evaluate
from feature_processing import gini_scorer
from feature_processing import parent_dir

from model_zoo import model_zoo
from tune_models import pick_animal_from_the_zoo
from sklearn import datasets
from sklearn.datasets import make_classification
from collections import Counter

from feature_processing import process_data

def build_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
    print(Counter(y))
    # X_train, y_train = check_data_balanced(X_train,y_train)
    
    # Parameters
    # ----------
    random_forest_paramaters = {
    'random_state':0,
    'n_estimators' : 200,
    'criterion' : 'entropy',
    'max_depth' : None,
    'min_samples_split' : 25,
    'min_samples_leaf' : 3,
    'max_features' : 5,
    'bootstrap' : True,
    'oob_score' : True,
    'verbose' : 0,
    'max_samples' : 0.012,
    'class_weight': 'balanced_subsample'
    }
    
    base_model = RandomForestClassifier(**random_forest_paramaters)
    #print('the train features are {}'.format(set(X_train.columns)))
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test,  model_name='base_model')
    baseline_model_path = os.path.join(parent_dir,'models','sahulat_forest_baseline.joblib')
    joblib.dump(base_model, open(baseline_model_path,'wb'))
    random_grid = {
        'min_samples_split' : [20,25,30],
        'min_samples_leaf' : [2,3,5],
        'max_depth': [None],
        'max_features' : [3,5,7],
        'bootstrap' : [True],
        'oob_score' :        [True],
        'random_state' :    [0],
        'verbose' :             [3],
        'max_samples' :   [0.008, 0.012, 0.02],
        'class_weight':     ['balanced','balanced_subsample']
    }

    
    best_model_params, tuned_model = pick_animal_from_the_zoo(zoopark = model_zoo)
    best_model = tuned_model(**best_model_params)
    best_model.fit(X_train, Y_train)
    
    train_features = set(X_train.columns)
    # num_train_features = len(train_features)                            
    print('there are {} train features: {}'.format(num_train_features, train_features))
    # rf_random.fit(X_train, y_train)
    # # retrieving the best model
    best_model = rf_random.best_estimator_
    best_params = rf_random.best_params_
    print('best params are {}'.format(best_params))
    best_accuracy = evaluate(best_model, X_test, y_test, model_name='best_model')
    print('Improvement of {:0.2f}%.'.format( 100 * (best_accuracy - base_accuracy) / base_accuracy))
    model_path = os.path.join(parent_dir,'models','sahulat_forest.joblib')
    joblib.dump(best_model, open(model_path,'wb'))


if __name__ == '__main__':
	raw_data_path = global_train_parameters['raw_data_path']
	X, y = process_data(raw_data_path)
    #X,y = make_classification(n_samples=50000, n_features=8, n_redundant=0,
	#n_clusters_per_class=2, weights=[0.94], flip_y=0, random_state=random_state)
	build_model(X, y)
