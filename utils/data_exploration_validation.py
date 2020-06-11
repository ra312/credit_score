# %%
import pixiedust
import pandas as pd

# %%
'''
## loading data
'''

# %%
data_foler = "../../sample_data"
data_file  = "loan_5k.csv"
train_data_file = "loan_train_5k.csv"
test_data_file = "loan_test_5k.csv"
from os.path import join

tf_records_file= "loan.tfrecord"

# %%
def load_and_trim(URL='data/loan_5k.csv'):

   pd.options.mode.use_inf_as_na = True
   dataframe = pd.read_csv(URL, sep=',', encoding='UTF-8', skipinitialspace=True)
   all_null_columns = dataframe.isnull().all()
   all_null_cols = list(all_null_columns[all_null_columns].index)
   dataframe.drop(columns=all_null_cols, axis=1, inplace=True)
   df_dtypes = pd.DataFrame(data=dataframe.dtypes, columns=['dtype'])
   df_dtypes.reset_index(inplace=True)
   df_dtypes.rename(columns={'index': 'col_name'}, inplace=True)
   value_types = dict(int64=0, float64=0.0, object='')
   df_dtypes['values_to_fill'] = df_dtypes['dtype'].apply(lambda x: value_types[str(x)])

   values_to_fill = dict(zip(df_dtypes['col_name'], df_dtypes['values_to_fill']))

   dataframe.fillna(value=values_to_fill, inplace=True)

   return dataframe


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
full_path_to_data = join(data_foler, data_file)
# data = pd.read_csv(full_path_to_data, sep=',', encoding='UTF-8', skipinitialspace=True)
data = load_and_trim(full_path_to_data)
train, test = train_test_split(data,test_size=0.2, train_size=0.8)
full_path_to_train = join(data_foler, train_data_file)
full_path_to_test = join(data_foler, test_data_file)
train.to_csv(path_or_buf=full_path_to_train,index=False)
test.to_csv(path_or_buf=full_path_to_test,index=False)

# %%
'''
## checking no NaN values
'''

# %%
bool(data.isnull().any().sum())

# %%
train.shape

# %%
test.shape

# %%
'''
we are using Tensorflow Data Validation since Azure Data validation seems to reside in a separate service
'''

# %%
import pyarrow
import apache_beam as beam
import apache_beam.io.iobase
import tensorflow
import tensorflow_data_validation as tfdv


# %%
#train_stats = tfdv.generate_statistics_from_csv(data_location=full_path_to_train)

# %%
train_stats = tfdv.generate_statistics_from_dataframe(train)

# %%


# %%
'''
## visualize statistics of train data
'''

# %%
tfdv.visualize_statistics(train_stats)

# %%


# %%
schema = tfdv.infer_schema(statistics=train_stats)

# %%


# %%
tfdv.display_schema(schema=schema)

# %%
'''
## check that training data complies with the schema inferred from train data !
'''

# %%
# Compute stats for evaluation data
test_stats = tfdv.generate_statistics_from_csv(data_location=full_path_to_test)

# Compare evaluation data with training data
tfdv.visualize_statistics(lhs_statistics=test_stats, rhs_statistics=train_stats,
                          lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')

# %%
'''
## evaluate for anomalies related with schema
'''

# %%
anomalies = tfdv.validate_statistics(statistics=train_stats, schema=schema)
tfdv.display_anomalies(anomalies)

# %%
## checking the data is compliant with the schema of the train part

# %%
data_stats = tfdv.generate_statistics_from_dataframe(data)
serving_anomalies = tfdv.validate_statistics(data_stats, schema)
tfdv.display_anomalies(serving_anomalies)

# %%


# %%


# %%

%matplotlib inline  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

# # Compute the correlation matrix
corr = train.corr('pearson')
# # Generate a mask for the upper triangle
# 

# %%
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# # Set up_ the matplotlib figure
f, ax = plt.subplots(figsize=(80, 80))

# # Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# plt.figure(figsize=(20,15))
# plt.subplots(figsize=(20,15))
# # Draw the heatmap with the mask and correct aspect ratio
# heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=2.3, center=0,annot=True,
# square=True, linewidths=1.5, cbar_kws={"shrink": .3})
heatmap = sns.heatmap(corr, mask=mask, annot=True, vmin=-1, vmax=1, center= 0,cmap='coolwarm')

# %%


# %%
nan_col_series = corr.isnull().sum()!=96

# %%
nan_cols = list(nan_col_series[nan_col_series==False].index)

# %%


# %%


# %%
nan_cols

# %%
train.drop(columns=nan_cols, axis=1, inplace=True)
reduced_corr = train.corr()
corr = reduced_corr

# %%
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# # Set up_ the matplotlib figure
f, ax = plt.subplots(figsize=(80, 80))

# # Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# plt.figure(figsize=(20,15))
# plt.subplots(figsize=(20,15))
# # Draw the heatmap with the mask and correct aspect ratio
# heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=2.3, center=0,annot=True,
# square=True, linewidths=1.5, cbar_kws={"shrink": .3})
heatmap = sns.heatmap(corr, mask=mask, annot=True, vmin=-1, vmax=1, center= 0,cmap='coolwarm')

# %%
threshold = 0.3
corr = corr.applymap(lambda x: 0.0 if (x < threshold and x>0.0) else x)

# %%
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# # Set up_ the matplotlib figure
f, ax = plt.subplots(figsize=(80, 80))

# # Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# plt.figure(figsize=(20,15))
# plt.subplots(figsize=(20,15))
# # Draw the heatmap with the mask and correct aspect ratio
# heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=2.3, center=0,annot=True,
# square=True, linewidths=1.5, cbar_kws={"shrink": .3})
heatmap = sns.heatmap(corr, mask=mask, annot=True, vmin=-1, vmax=1, center= 0,cmap='coolwarm')

# %%
'''
## drop columns with zero corr coefficients
'''

# %%
non_significant = corr.sum(axis=1)<1.3

# %%
non_significant_cols = list(corr[non_significant].index)

# %%
train.drop(columns=non_significant_cols, axis=1, inplace=True)
corr = train.corr()

# %%
corr

# %%
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# # Set up_ the matplotlib figure
f, ax = plt.subplots(figsize=(80, 80))

# # Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# plt.figure(figsize=(20,15))
# plt.subplots(figsize=(20,15))
# # Draw the heatmap with the mask and correct aspect ratio
# heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=2.3, center=0,annot=True,
# square=True, linewidths=1.5, cbar_kws={"shrink": .3})
heatmap = sns.heatmap(corr, mask=mask, annot=True, vmin=-1, vmax=1, center= 0,cmap='coolwarm')

# %%
'loan_status' in list(corr.columns)

# %%
