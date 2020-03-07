import os
import csv
import numpy as np

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import dtypes

from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential






data_folder = 'data'

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
   full_filename = os.path.join(data_folder,filename)
   batch_size = train_hyperparameters['batch_size']
   num_epochs = train_hyperparameters['num_epochs']
   metadata = infer_column_names_and_types(filename=full_filename)
   column_names = metadata.keys()
   label_name = 'loan_status'
   if label_name not in column_names:
      print(f'The label {label_name} not found in {column_names}')
      exit(-1)
   df = tf.data.experimental.make_csv_dataset(full_filename, batch_size=batch_size,
                                              column_names=column_names,
                                              label_name=label_name,
                                              num_epochs=num_epochs, ignore_errors=True)


   return df, metadata

#

def process_data_to_tfrecord(data, metadata):
   encoder = LabelEncoder()
   float_columns = [column_name for column_name in metadata.keys() if metadata[column_name].name.startswith('float')]
   int_columns = [column_name for column_name in metadata.keys() if metadata[column_name].name.startswith('int')]
   string_columns = [column_name for column_name in metadata.keys() if metadata[column_name].name.startswith('string')]


   tfrecord_file = 'loan'+'.tfrecord'
   full_tfrecord_file = os.path.join(data_folder,tfrecord_file)
   records_writer = tf.io.TFRecordWriter(full_tfrecord_file)
   for line in data.as_numpy_iterator():
      # print(type(line))
      # print(line[0])
      # print(line[0].keys())
      feature = {}
      raw_features = line[0]
      raw_label = line[1]
      for column_name in raw_features.keys():
         values = list(raw_features[column_name])
         if column_name in float_columns:
            feature[column_name]=tf.train.Feature(float_list=tf.train.FloatList(value=values))
         elif column_name in int_columns:
            feature[column_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
         elif column_name in string_columns:
            values = encoder.fit_transform(values)
            feature[column_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
      label_values = list(raw_label)
      label_values = encoder.fit_transform(label_values)
      feature['label']=tf.train.Feature(int64_list=tf.train.Int64List(value=label_values))
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      records_writer.write(example.SerializeToString())
   records_writer.close()
   return full_tfrecord_file
def load_process_tf_records(filename, metadata):

   full_filename = os.path.join(data_folder,filename)
   DATASET_SIZE = 10000 #dummy value
   train_size = int(0.7 * DATASET_SIZE)
   val_size = int(0.15 * DATASET_SIZE)
   test_size = int(0.15 * DATASET_SIZE)

   full_dataset = tf.data.TFRecordDataset(full_filename)
   feature_description = {}
   for column in metadata.keys():
      if metadata[column].name.startswith('int'):
         feature_description[column]=tf.io.VarLenFeature(tf.int64)
      elif metadata[column].name.startswith('float'):
         feature_description[column] = tf.io.VarLenFeature(tf.float64)
   feature_description['label']=tf.io.FixedLenFeature([], tf.int64, default_value=0)
   def parse_protobuf(protobuf):
      example = tf.io.parse_example(protobuf, feature_description)
   full_dataset = full_dataset.shuffle(buffer_size=1024)

   train_dataset = full_dataset.take(train_size)
   test_dataset = full_dataset.skip(train_size)
   test_dataset = test_dataset.take(test_size)
   val_dataset = test_dataset.skip(val_size)


   return train_dataset, test_dataset, val_dataset

def compile_model(input_dimension, batch_size):
   # input_dim = 8*n, n - the depth of the network
   model = Sequential()
   model.add(Flatten(input_shape=[batch_size, input_dimension]))
   model.add(Dense(units=input_dimension, activation='relu'))
   n = int(input_dimension/8.0)
   model.add(Dense(units=n, activation='relu'))
   model.add(Dense(units=5, activation='softmax'))
   model.compile(loss="sparse_categorical_crossentropy",
                 optimizer="sgd",
                 metrics=["accuracy"])

   return model


if __name__ == '__main__':
   filename = 'loan_10k.csv'
   data, metadata = load_data(filename=filename)
   tfrecord_file = process_data_to_tfrecord(data=data, metadata=metadata)
   train_data, test_data, val_data = load_process_tf_records(filename=tfrecord_file)
   input_dimension = len(metadata.keys())
   batch_size = train_hyperparameters['batch_size']
   the_model = compile_model(input_dimension=input_dimension, batch_size=batch_size)
   