import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

import joblib

from azureml.core.run import Run


target_column = 'Loan_Good_Or_Bad'

run = Run.get_context()

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
    nan_targets.drop(axis=1, columns=target_column).to_csv('loan_test.csv', index=False)
    data.drop(nan_target_rows, inplace=True)
    nan_targets.to_csv(verify_in_deployment_data_path, index=False)
    print('dropping these values')
    
    return data

def parsing_object_type_columns(data):
    data['No_of_Deposits_Transactions']=data['No_of_Deposits_Transactions'].apply(lambda x: str(x).replace(',',''))
    data['No_of_Deposits_Transactions']=data['No_of_Deposits_Transactions'].astype('float64')
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
    encoder.fit(string_data.as_matrix())
    encoded_string_data = encoder.transform(string_data.as_matrix())
    encoded_string_df = pd.DataFrame(data=encoded_string_data, columns=string_cols, index=index_to_overwrite)
    processed_data = data.copy(deep=True)
    processed_data.drop(columns=string_cols, axis=1, inplace=True)
    processed_data = pd.concat([processed_data, encoded_string_df], axis=1)
    
    return processed_data

def main():
    parser = argparse.ArgumentParser()
    # "--input", input_named.as_download(), "--output", processed_data],

    parser.add_argument('--input', type=str, default='indian_data/joined_data_v2.csv',
                        help='raw data to be processed ')
    parser.add_argument('--output', type=str, default='indian_data/processed_data_v2.csv',
                        help='Penalty parameter of the error term')

    args = parser.parse_args()
    run.log('raw data to be processed', np.str(args.input))
    run.log('Penalty', np.str(args.output))