import os
import sys
package_directory = os.path.dirname(os.path.realpath(__file__))
for _ in range(1):
	parent_dir = os.path.dirname(package_directory)
sys.path.append(os.path.join(parent_dir, 'utils'))
sys.path.append(os.path.join(parent_dir, 'src'))

import requests
import json
import pandas as pd
import joblib
from sklearn import datasets
from sklearn.datasets import make_classification
from collections import Counter

from build_model import build_model
from build_model import random_state
def get_synthetic_imbalanced_data():
	columns =  ['Loan_Product_Name', 'Branch', 'Member_Since_Days', 
						'No_of_Loans_Taken', 'No_of_Deposits_Transactions',
 						'Account_No', 'No_of_Deposit_Accounts', 'Is_Prompt_Depositer']
	target_column = ['Loan_Good_Or_Bad']
	X,y = make_classification(n_samples=50000, n_features=8, n_redundant=0,
		n_clusters_per_class=2, weights=[0.94], flip_y=0, random_state=random_state)
	X = pd.DataFrame(data=X, columns=columns)
	y = pd.DataFrame(data=y, columns=target_column)
	return X, y
	
if __name__ == '__main__':
	# raw_data_path = global_train_parameters['raw_data_path']
    # X, y = process_data(raw_data_path)
	# X,y = make_classification(n_samples=50000, n_features=9, n_redundant=0,
	# 	n_clusters_per_class=2, weights=[0.94], flip_y=0, random_state=random_state)
	X, y = get_synthetic_imbalanced_data()
	build_model(X,y)



