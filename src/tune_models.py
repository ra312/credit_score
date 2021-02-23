# -*- coding: utf-8 -*-
"""
We test hyperopt parameter tuning on artificial highly imbalanced data
"""
from sklearn.metrics import make_scorer, confusion_matrix, roc_auc_score
import os
import joblib
import pprint
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter
#from build_model import process_data
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as imb_pipeline
# from sklearn import datasets
# from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import RepeatedStratifiedKFold
from scoring_methods import gini_scorer, custom_scorer, new_gini_scorer, custom_precision_scorer
from model_zoo import model_zoo

def dropping_nan_features(data, nan_columns_to_drop):

    data.drop(nan_columns_to_drop, axis = 1, inplace=True)
    print(f'the columns {nan_columns_to_drop} have been dropped')
    num_dropped = len(nan_columns_to_drop)
    print(f'we dropped {num_dropped}')
    num_cols=len(data.columns)
    print(f'there are {num_cols} in total')
    return data
    
def processing_nan_targets(data, target_column):
    nan_targets = data[data[target_column].isnull()]
    nan_target_rows = data[data[target_column].isnull()].index
    num_of_target_missing = len(nan_targets)
    num_of_target_total = len(data[target_column])
    fraction_target_missing = num_of_target_missing/num_of_target_total
    message = f'I found out that {fraction_target_missing} target column values are missing'
    print(message)
    data.drop(nan_target_rows, inplace=True)
    # we save unlabelled data to test in deployment

    verify_in_deployment_data_path = global_train_parameters['verify_in_deployment_data_path']

    nan_targets.drop(columns=[target_column], axis=1, inplace=False).to_csv(verify_in_deployment_data_path, index=False)
    print('dropping these values')
    
    return data

def parsing_object_type_columns(data):
    """[summary]
    parsing string numbers in the format '012,12'
    Arguments:
        data {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if 'No_of_Deposits_Transactions' in set(data.columns):
        data['No_of_Deposits_Transactions']=data['No_of_Deposits_Transactions'].apply(lambda x: str(x).replace(',',''))
        data['No_of_Deposits_Transactions']=data['No_of_Deposits_Transactions'].astype('float64')
    if 'Days_Closed_In' in set(data.columns):
        data['Days_Closed_In']=data['Days_Closed_In'].apply(lambda x: str(x).replace(',',''))
        data['Days_Closed_In']=data['Days_Closed_In'].astype('float64')
    return data

def missing(data):
    print('checking if we have NaN values ...')
    return bool(data.isnull().any().sum())

def encoding_data(data, encoder):
    string_data = data.select_dtypes(include=['object'], exclude=['int','float'])
    string_cols = list(string_data.columns)
    index_to_overwrite = string_data.index
    
    # encoder.fit(string_data.as_matrix())
    encoder.fit(string_data.values)

    # encoded_string_data = encoder.transform(string_data.as_matrix())
    encoded_string_data = encoder.transform(string_data.values)

    encoded_string_df = pd.DataFrame(data=encoded_string_data, columns=string_cols, index=index_to_overwrite)
    processed_data = data.copy(deep=True)
    processed_data.drop(columns=string_cols, axis=1, inplace=True)
    processed_data = pd.concat([processed_data, encoded_string_df], axis=1)
    
    return processed_data, encoder

def process_features(data):
    data = dropping_a_posteriori_columns(data)
    return data


def extract_transform_load(raw_data_path):
    """
    load input data, transform and save it
    Return: fitted encoder 
    """
    data = pd.read_csv(raw_data_path, sep=';', encoding='UTF-8', skipinitialspace=True)
    if missing(data):
        num_of_values = data.shape[0]*data.shape[1]
        num_of_missing = data.isnull().sum().sum()
        fraction_missing = num_of_missing/num_of_values
        message = f'There are {num_of_missing} values missing out of {num_of_values}'
        message = f'I found out that {fraction_missing} of values are missing'
        print(message)

    encoder = OrdinalEncoder()
    
    nan_columns_to_drop = ['Gender','Age_Years',
                        'Occupation', 
                        'Terminate_Date','Required_Installments',
                        'No_of_Installments_Paid','Loan_Close_Date',
                        'Loan_Disburse_Date',
                        'Loan_Maturity_Date'] 
   # data = dropping_nan_features(data, nan_columns_to_drop)
    #select_features = global_train_parameters['select_features']
    select_features=list(data.columns)
    if select_features:
        print('select features {}'.format(select_features))
        data = process_features(data)
    #target_column = global_train_parameters['target_column']
    target_column='TARGET'
    #data = processing_nan_targets(data, target_column)
    # num_of_nan_values = data.isnull().sum()
    # print(f'There are {num_of_nan_values} nan values left in the data!')
    if missing(data):
        print('Warning: there is still data missing!\n')
    else:
        print('No missing data!\n')
    data = parsing_object_type_columns(data)
    processed_data, encoder  = encoding_data(data, encoder=encoder)
    #processed_data_path = global_train_parameters['processed_data_path']
    processed_data_path='train_processed.csv'
    processed_data.to_csv(processed_data_path, index=False)
    #encoder_path = global_train_parameters['encoder_path']
    encoder_path='encoder.pkl'
    with open(encoder_path,'wb') as encoder_file:
        joblib.dump(encoder, encoder_file)
    return encoder




def dropping_a_posteriori_columns(data):
    '''
    we have to drop the columns with possible a posteriori information, i.e.
    the information not available at the moment of decision-making
    '''
    a_posteriori_features = [
    'Days_Closed_In', 'Join_Date',
    'Service_Charge_Amount',
    # 'Loan_Product_Name',
    'Loan_Amount',
    'Deposit_DB_Total_Amount',
    'Foreclose_Status', 'No_of_Installments_Paid', 'Loan_Close_Date', 
    'Loan_Disburse_Date', 'Loan_Maturity_Date',
    'Required_Installments'
    ]
    
    a_posteriori_cols_found = list(set(a_posteriori_features).intersection(set(data.columns)))
    print('dropping possibly a posteriori columns ....\n')
    data.drop(columns=a_posteriori_cols_found, axis=1, inplace=True)
    print('columns {} have been dropped ...'.format(a_posteriori_cols_found))

    return data


def check_data_balanced(X,y):
    oversampled = global_train_parameters['oversampled']
    oversample = SMOTE()
    print('the class distribution is imbalanced {}'.format(Counter(y)))
    print('The dataset target column is imbalanced {}'.format(Counter(y)))
    if oversampled:
        print('oversampling the features and the target ...')
        X, y = oversample.fit_resample(X, y)
        print('achieved the ratio of {}'.format(Counter(y)))
    processed_features_path = global_train_parameters['processed_features_path']
    processed_labels_path = global_train_parameters['processed_labels_path']
    X.to_csv(processed_features_path, index=False)
    y.to_csv(processed_labels_path, index=False)
    return X,y


def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)/len(df)

def normalized_gini(solution, submission, gini=gini):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini


def gini_normalized_new(y_actual, y_pred):
    """Simple normalized Gini based on Scikit-Learn's roc_auc_score"""
    
    # If the predictions y_pred are binary class probabilities
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
    return gini(y_actual, y_pred) / gini(y_actual, y_actual)
gini_scorer = make_scorer(normalized_gini, greater_is_better = True)

def process_data(raw_data_path):
    encoder = extract_transform_load(raw_data_path)
    #processed_data_path = global_train_parameters['processed_data_path']
    processed_data_path = 'train_processed.csv'
    X = pd.read_csv(processed_data_path)
    target_column = 'TARGET'
    y = X.pop(target_column)
    #encoder_path = os.path.join(parent_dir, 'models','sahulat_encoder.joblib')
    encoder_path='encoder.pkl'
    with open(encoder_path,'wb') as encoder_file:
        joblib.dump(encoder, encoder_file)
    return X, y
#from get_sahulat_data import load_sahulat_data

pp = pprint.PrettyPrinter(indent=4)

global random_state 
random_state = 42

#raw_data_path = 'train.csv'
#X, Y = process_data(raw_data_path)
#X, Y = load_sahulat_data()
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, shuffle=True,  random_state=random_state, stratify=Y)
#scale = MinMaxScaler()
#counter = Counter(Y)
#print(counter)
raw_data_path = 'train.csv'
X, y = process_data(raw_data_path)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
def clean_best(params):
    del params['scale']
    del params['normalize']
    # del params['scorer']
    # del params['model']
    return params



from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             log_loss, make_scorer, precision_recall_curve,
                             r2_score, roc_auc_score)
def testing_the_hyperparameters():
	print('Testing best params on the unseen data ...\n' )
	cleaned_best = clean_best(best)
	rf = RandomForestClassifier(**cleaned_best)
	rf.fit(X_train, Y_train)
	Y_pred = rf.predict(X_test)
	Y_pred_prba = rf.predict_proba(X_test)
	accuracy_score_value = accuracy_score(y_true=Y_test, y_pred=Y_pred)
	r2_score_value = r2_score(y_true=Y_test, y_pred=Y_pred)
	f1_score_value = f1_score(y_true=Y_test, y_pred=Y_pred)
	roc_auc_value = roc_auc_score(y_true=Y_test, y_score=Y_pred)
	g_v = normalized_gini(solution=Y_test, submission=Y_pred)
	g_v_proba = normalized_gini_new(solution=Y_test, submission=Y_pred_proba)
	conf_matrix = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
	tn, fp, fn, tp = conf_matrix.ravel()
	print(f'tn={tn} ')
	print(f'fp={fp} ')
	print(f'fn={fn} ')
	print(f'tp={tp}\n')
	recall = tp / (tp + fn)
	precision = tp / (tp + fp)
	F1 = 2 * recall * precision / (recall + precision)
	model_name = 'best_hyperopted_model'
	print('Model Performance:{}'.format(model_name))
	print('accuracy_score is: {:0.4f} '.format(accuracy_score_value))
	print('roc-auc value is {:0.4f}.'.format(roc_auc_value))
	print('gini score is {:0.4f}.'.format(g_v))
	print('new gini score is {:0.4f}.'.format(g_v_proba))
	print(f'recall = {recall}')
	print(f'precision = {precision}\n')
	print(f'F1 = {F1}\n')
	print('conf_matrix = \n')
	print(conf_matrix)

# global best_cross_val_score
# best_cross_val_score = 0
def pick_animal_from_the_zoo(zoopark = model_zoo): 
	'''''
		we select best model from the zoopark

	"'''
	trials = Trials()
	# best_cross_val_score = 0
	best_model = None
	for model, params in model_zoo.items():
		cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
		# cv = 5
		def hyperopt_cross_val_custom_score(params):
			X_ = X_train[:]
			# if 'normalize' in params:
			# 	if params['normalize'] == 1:
			# 		X_ = normalize(X_)
			del params['normalize']
			# if 'scale' in params:
			# 	if params['scale'] == 1:
			# 		X_ = scale.fit_transform(X_)
			del params['scale']
			# scoring = gini_scorer
			scoring = custom_precision_scorer
            
			adasyn = SMOTE()
			clf = model(**params)
			pipeline = imb_pipeline([('sampling', adasyn), ('class', clf)])
			# pipeline = clf
			score = cross_val_score(estimator=pipeline,
													X=X_, 
													y=Y_train, 
													scoring=scoring, 
													cv=cv).mean()
			return score
		
		def loss_function(params):
			cross_val_score = hyperopt_cross_val_custom_score(params=params)	
			loss_value = 1-cross_val_score
			return {'loss': loss_value, 'status': STATUS_OK}
		
		print(f'model = {model}')
		print(f'params={params}')
		best = fmin(
					fn = loss_function, 
					space=params, 
					algo=tpe.suggest, 
					max_evals=50, 
					trials=trials,
					return_argmin=False
					)
		print("best:\n")
		pp.pprint(best)
		best_model = model
	cleaned_best = clean_best(best)
	return cleaned_best, best_model
	

if  __name__ == '__main__':
	_, _ = pick_animal_from_the_zoo(zoopark=model_zoo)
