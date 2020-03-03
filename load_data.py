import tensorflow as tf
import csv
import numpy as np
from tensorflow.python.lib.io import file_io
from tensorflow.keras.wrappers import scikit_learn as sklearn_tf_wrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

from tensorflow.python.framework import dtypes
# import tensorflow_transform as tft
from sklearn.preprocessing import MinMaxScaler,StandardScaler

train_hyperparameters = {
   'batch_size': 5,
   'split_ratio':0.75,
   'num_epochs': 1
}
rf_regr_parameters = {
   'n_estimators': 500,
   'criterion': 'mse',
   'bootstrap': True
}

_ACCEPTABLE_CSV_TYPES = (dtypes.float32, dtypes.float64, dtypes.int32,
                         dtypes.int64, dtypes.string)


def _is_valid_int32(str_val):
   try:
      # Checks equality to prevent int32 overflow
      return dtypes.int32.as_numpy_dtype(str_val) == dtypes.int64.as_numpy_dtype(
         str_val)
   except (ValueError, OverflowError):
      return False


def _is_valid_int64(str_val):
   try:
      dtypes.int64.as_numpy_dtype(str_val)
      return True
   except (ValueError, OverflowError):
      return False


def _is_valid_float(str_val, float_dtype):
   try:
      return float_dtype.as_numpy_dtype(str_val) < np.inf
   except ValueError:
      return False


def _infer_type(str_val, na_value, prev_type):

   if str_val in ("", na_value):
      # If the field is null, it gives no extra information about its type
      return prev_type

   type_list = [
      dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.string
   ]  # list of types to try, ordered from least permissive to most

   type_functions = [
      _is_valid_int32,
      _is_valid_int64,
      lambda str_val: _is_valid_float(str_val, dtypes.float32),
      lambda str_val: _is_valid_float(str_val, dtypes.float64),
      lambda str_val: True,
   ]  # Corresponding list of validation functions

   for i in range(len(type_list)):
      validation_fn = type_functions[i]
      if validation_fn(str_val) and (prev_type is None or
                                     prev_type in type_list[:i + 1]):
         return type_list[i]


def infer_column_names_and_types(filename, num_rows_for_inference=150):
   field_delim = ','
   use_quote_delim = True
   header = True
   na_value =""
   file_io_fn= lambda filename: file_io.FileIO(filename, "r")
   column_names = []
   num_cols = 0
   with file_io_fn(filename) as f:
      rdr = csv.reader(
         f,
         delimiter=field_delim,
         quoting=csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE)
      csv_row = next(rdr)
      if header:
         column_names = csv_row
         num_cols = len(column_names)
         if num_cols == 0:
            raise ValueError("Empty header!")
      inferred_types = [None] * len(csv_row)
      for i, csv_row in enumerate(rdr):
         if len(csv_row) != num_cols:
            raise ValueError(
               "Problem inferring types: CSV row has different number of fields "
               "than expected.")
         for j, col_index in enumerate(range(num_cols)):
            inferred_types[j] = _infer_type(csv_row[col_index], na_value,
                                            inferred_types[j])
         if num_rows_for_inference is not None and i >= num_rows_for_inference:
            break

   inferred_types = [t or dtypes.string for t in inferred_types]
   # keys = columns
   # values = column_types
   metadata = dict(zip(column_names, inferred_types))
   return metadata


def load_data(filename):
   batch_size = train_hyperparameters['batch_size']
   num_epochs = train_hyperparameters['num_epochs']
   metadata = infer_column_names_and_types(filename=filename)
   column_names = metadata.keys()
   label_name = 'loan_status'
   if label_name not in column_names:
      print(f'The label {label_name} not found in {column_names}')
      exit(-1)
   df = tf.data.experimental.make_csv_dataset(filename, batch_size=batch_size,
                                              column_names=column_names,
                                              label_name=label_name,
                                              num_epochs=num_epochs, ignore_errors=True)


   return df, metadata

#

def preprocessing_fn(data, label_column, metadata):
   # categorical_columns = [n for n in metadata if metadata[n] == tf.string]
   # numeric_columns = [n for n in metadata if metadata[n] != tf.string]
   column_names = metadata.keys()
   types = set(metadata.values())


   float_columns = [column_name for column_name in metadata.keys() if metadata[column_name].name.startswith('float')]
   int_columns = [column_name for column_name in metadata.keys() if metadata[column_name].name.startswith('int')]
   byte_columns = [column_name for column_name in metadata.keys() if metadata[column_name].name.startswith('string')]
   for line in data.as_numpy_iterator():
      print(type(line))
      print(line[0])
      print(line[0].keys())
      feature = {}
      raw_features = line[0]
      raw_label = line[1]
      for column_name in raw_features.keys():
         values = raw_features[column_name]
         if column_name in float_columns:
            feature[column_name]=tf.train.Feature(float_list=tf.train.FloatList(value=values))
         elif column_name in int_columns:
            feature[column_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
         elif column_name in byte_columns:
            feature[column_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
      label_values = raw_label[label_column]
      feature['label']=tf.train.Feature(bytes_list=tf.train.BytesList(value=label_values))


   # X = data[[label_column]]
   # Y =
#   """Preprocess input columns into transformed columns."""
#   # Since we are modifying some features and leaving others unchanged, we
#   # start by setting `outputs` to a copy of `inputs.
#   outputs = inputs.copy()
#
#   # Scale numeric columns to have range [0, 1].
#   for key in NUMERIC_FEATURE_KEYS:
#     outputs[key] = tft.scale_to_0_1(outputs[key])
#
#
#   # For all categorical columns except the label column, we generate a
#   # vocabulary but do not modify the feature.  This vocabulary is instead
#   # used in the trainer, by means of a feature column, to convert the feature
#   # from a string to an integer id.
#   for key in CATEGORICAL_FEATURE_KEYS:
#     tft.vocabulary(inputs[key], vocab_filename=key)
#
#   # For the label column we provide the mapping from string to index.
#   initializer = tf.lookup.KeyValueTensorInitializer(
#       keys=['>50K', '<=50K'],
#       values=tf.cast(tf.range(2), tf.int64),
#       key_dtype=tf.string,
#       value_dtype=tf.int64)
#   table = tf.lookup.StaticHashTable(initializer, default_value=-1)
#
#   outputs[LABEL_KEY] = table.lookup(outputs[LABEL_KEY])
#
#   return outputs
   X_train  = data
   X_test = data
   Y_train = data
   Y_test = data
   return X_train, X_test, Y_train, Y_test

def build_model():
   rf = RandomForestRegressor(**rf_regr_parameters)
   return rf


if __name__ == '__main__':
   filename = 'loan_10.csv'
   data, metadata = load_data(filename=filename)
   X_train, X_test, Y_train, Y_test = preprocessing_fn(data=data, label_column='loan_status',metadata=metadata)
   rf = build_model()


# def infer_csv_columns(filename):
#     file_io_fn = lambda filename: file_io.FileIO(filename, "r")
#     csv_kwargs = {
#         "delimiter": field_delim,
#         "quoting": csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE
#     }
#     with file_io_fn(filename as f:
#     try:
#         column_names = next(csv.reader(f, **csv_kwargs))
#     except StopIteration:
#         raise ValueError("Received StopIteration when reading the header line "
#                         "of %s.  Empty file?") % filename])
#     return column_names
