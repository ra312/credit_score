import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from tpot import TPOTClassifier
def preprocess_data(filename):
   """
   read csv data and preprocess the data using scikit-learn utilities
   :param filename: csv file
   :param label_column: label name
   :return:scipy.sparse coo
   """

   data = pd.read_csv(filename)
   labels = data['loan_status']
   data.drop(columns=['loan_status'], axis=1, inplace=True)

   infer_type = lambda x: pd.api.types.infer_dtype(x, skipna=True)
   data_types = data.apply(infer_type, axis=0)
   empty_col_index = [i for i, type in enumerate(data_types) if type == 'empty']
   cols = data.columns
   columns_to_drop = cols[empty_col_index]
   data.drop(columns=columns_to_drop, axis=1, inplace=True)

   float_columns = [cols[i] for i, type in enumerate(data_types) if type == 'floating']
   integer_columns = [cols[i] for i, type in enumerate(data_types) if type == 'integer']
   string_columns = [cols[i] for i, type in enumerate(data_types) if type == 'string']
   numeric_features = float_columns + integer_columns
   numeric_transformer = Pipeline(steps=[
   ('imputer', SimpleImputer(strategy='median')),
   ('scaler', StandardScaler())])

   categorical_features = string_columns
   categorical_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
      ('onehot', OneHotEncoder(handle_unknown='ignore'))])

   preprocessor = ColumnTransformer(
      transformers=[
      ('num', numeric_transformer, numeric_features),
      ('cat', categorical_transformer, categorical_features)
      ]
   )

   data_preprocessor= Pipeline(steps=[
                                       ('preprocessor', preprocessor)
                                      ])

   data = data_preprocessor.fit_transform(data)
   label_encoder = LabelEncoder()
   encoded_labels = label_encoder.fit_transform(labels)

   return data, encoded_labels




# if __name__ == '__main__':
#    data, labels = preprocess_data('data/loan_5k.csv')
#    print(data.shape)
