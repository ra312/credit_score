from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, confusion_matrix, r2_score, f1_score


from collections import Counter
import pandas as pd
import argparse
import joblib

import os
parent_dir = os.path.realpath(__file__)
    # going up to parent directory of the current directory
for _ in range(2):
    parent_dir = os.path.dirname(parent_dir)
from azureml.core.run import Run
import azureml.core
from azureml.core import Workspace, Datastore

# Get PipelineData argument
parser = argparse.ArgumentParser()
parser.add_argument('--encoder_path', type=str, dest='encoder_path')
parser.add_argument('--processed_data_path', type=str, dest='processed_data_path')

args = parser.parse_args()
encoder_path = args.encoder_path
processed_data_path = args.processed_data_path

encoder_path = os.path.join(parent_dir,'models','sahulat_encoder.joblib')
global_train_parameters ={
    'select_features':True,
    'oversampled': True,
    'target_column':'Loan_Good_Or_Bad',
    'encoder_path': encoder_path,
    'raw_data_path': '~/projects/scoring_product/data/raw/sahulat_data.csv',
    'processed_data_path':r'~/projects/scoring_product/data/processed/test_sahulat_data.csv',
    # this dataset below is formed after balancing
    'processed_features_path':r'~/projects/scoring_product/data/processed/test_sahulat_features.csv',
    # this dataset below is formed after balancing
    'processed_labels_path':r'~/projects/scoring_product/data/processed/test_sahulat_labels.csv',
    'processed_train_path':r'~/projects/scoring_product/data/processed/test_sahulat_train_features.csv',
    'processed_test_path':r'~/projects/scoring_product/data/processed/test_sahulat_test_features.csv',
    'verify_in_deployment_data_path' :r'~/projects/scoring_product/models/test_test_when_deployed.csv'
}


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
    data = pd.read_csv(raw_data_path, sep=',', encoding='UTF-8', skipinitialspace=True)
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
    data = dropping_nan_features(data, nan_columns_to_drop)
    select_features = global_train_parameters['select_features']
    if select_features:
        print('select features {}'.format(select_features))
        data = process_features(data)
    target_column = global_train_parameters['target_column']

    data = processing_nan_targets(data, target_column)
    # num_of_nan_values = data.isnull().sum()
    # print(f'There are {num_of_nan_values} nan values left in the data!')
    if missing(data):
        print('Warning: there is still data missing!\n')
    else:
        print('No missing data!\n')
    data = parsing_object_type_columns(data)
    processed_data, encoder  = encoding_data(data, encoder=encoder)
    processed_data_path = global_train_parameters['processed_data_path']
    processed_data.to_csv(processed_data_path, index=False)
    encoder_path = global_train_parameters['encoder_path']
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
    processed_data_path = global_train_parameters['processed_data_path']
    X = pd.read_csv(processed_data_path)
    target_column = global_train_parameters['target_column']
    y = X.pop(target_column)
    encoder_path = os.path.join(parent_dir, 'models','sahulat_encoder.joblib')
    with open(encoder_path,'wb') as encoder_file:
        joblib.dump(encoder, encoder_file)
    return X, y

def evaluate(model, X_test, y_test, model_name):
    # r2_score_value = r2_score(y_true=y_test, y_pred=y_pred)
    # f1_score_value = f1_score(y_true=y_test, y_pred=y_pred)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    accuracy_score_value = accuracy_score(y_true=y_test, y_pred=y_pred)
    r2_score_value = r2_score(y_true=y_test, y_pred=y_pred)
    f1_score_value = f1_score(y_true=y_test, y_pred=y_pred)
    roc_auc_value = roc_auc_score(y_true=y_test, y_score=y_pred)
    # difference = (y_test-y_pred).values
    # gini_value = gini(y_test, y_pred)
    # gini_value = gini(solution=y_test, submission=y_pred)
    gini_normalized_value = normalized_gini(solution=y_test, submission = y_pred)
    gini_normalized_proba = gini_normalized_new(y_actual=y_test, y_pred = y_pred_proba)
    # gini_index_value = gini_index(model, X_test, y_test)
    # errors = abs(y_pred - y_test)
    # mape = np.mean(errors)
    # accuracy = 100 - mape
    
    print('Model Performance:{}'.format(model_name))
    print('accuracy_score is: {:0.4f} '.format(accuracy_score_value))
    print('roc-auc value is {:0.4f}.'.format(roc_auc_value))
    # print('gini score  is {:0.4f}.'.format(gini_value))
    print('gini score is {:0.4f}.'.format(gini_normalized_value))
    print('new gini score is {:0.4f}.'.format(gini_normalized_proba))
    # print('gini_index is {:0.4f}.'.format(gini_index_value))
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f'fp={fp}\n ')
    print(f'fn={fn} \n')
    print(f'tp={tp} \n')
    print(f'tn={tn}\n ')
    print(conf_matrix)
    # run.log('accuracy_score', np.float(accuracy_score_value))
    # # print(f'specificity = {specificity}\n')
    # run.log('r2_score_value', np.float(r2_score_value))
    # run.log('f1_score_value', np.float(f1_score_value))
    # run.log('roc_auc value', np.float(roc_auc_value))
    # run.log('gini score', np.float(gini_normalized_value))

    # run.log('gini normalized value', np.float(gini_normalized_value))
    feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_test.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
    print('Feature importance scores of {} model \n'.format(model_name))
    print(feature_importances)

    return accuracy_score_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', dest='encoder_path', required=True)
    parser.add_argument('--processed_data_path', dest='processed_data_path', required=True)
    args = parser.parse_args()
    encoder_path = args.encoder_path
    processed_data_path = args.processed_data_path
    # run = Run.get_context()
    raw_data = run.input_datasets['sahulat_ds']
    sahulat_ds = Dataset.File.from_files([(datastore, 'sahulat-data')])
    extract_transform_load(raw_data)
    datastore.upload(src_dir=encoder_path,
                target_path='sahulat-data',
                overwrite=True,
                show_progress=True)
    datastore.upload(src_dir=processed_data_path,
                target_path='sahulat-data',
                overwrite=True,
                show_progress=True) 
    print('upload completed')

if __name__ == '__main__':
    main()


    

    
