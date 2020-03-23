train_hyperparameters = {
   'batch_size': 32,
   'train_size': 0.7,
   'test_size': 0.15,
   'validate_size': 0.15,
   'steps_per_epoch': 1,
   'split_ratio': 0.75,
   'num_epochs': 5,
   'buffer_size': 32,
   'label_column': 'loan_status',
   'shuffle': True,
   'my_log_dir':'my_log_dir'
}
classification_parameters = dict()

from os import system as zsh
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.backend import set_floatx
set_floatx('float64')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

def split_data(dataframe):
   # cols = list(dataframe.columns)
   # dataframe = dataframe.head(train_hyperparameters['batch_size'] * 300)
   # dataframe.drop(columns=cols[0:75], axis=1, inplace=True)
   string_columns = list(dataframe.select_dtypes(exclude=['int', 'float']).columns)

   int_columns = list(dataframe.select_dtypes(include=['int'], exclude=['float', 'object']).columns)
   # int_columns = []
   float_columns = list(dataframe.select_dtypes(include=['float'], exclude=['int', 'object']).columns)

   cat_vocabulary = {}
   for col in string_columns:
      cat_vocabulary[col] = list(set(dataframe[col].values))
   numerical_columns = int_columns + float_columns
   train, test = train_test_split(dataframe, test_size=0.2)
   train, val = train_test_split(train, test_size=0.2)
   print(len(train), 'train examples')
   print(len(val), 'validation examples')
   print(len(test), 'test examples')
   return train, test, val, cat_vocabulary, numerical_columns


def load_and_trim(URL='data/loan_5k.csv'):

   pd.options.mode.use_inf_as_na = True
   dataframe = pd.read_csv(URL, sep=',', encoding='UTF-8', skipinitialspace=True)
   train_hyperparameters['batch_size'] = 1
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


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
   dataframe = dataframe.copy()
   label_column = train_hyperparameters['label_column']
   labels = dataframe.pop(label_column)

   encoded_labels=label_encoder.fit_transform(labels)
   classification_parameters['classes'] = list(set(encoded_labels))
   encoded_labels = np.asarray(encoded_labels).astype('float32').reshape((-1, 1))

   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), encoded_labels))

   if shuffle:
      ds = ds.shuffle(buffer_size=len(dataframe))
   ds = ds.batch(batch_size)
   return ds




def compile_feature_layer(cat_vocabulary, numerical_columns):
   feature_columns = []
   label_column = train_hyperparameters['label_column']
   if label_column in numerical_columns:
      numerical_columns.remove(label_column)
   elif label_column in list(cat_vocabulary.keys()):
      cat_vocabulary.pop(label_column)

   for header in numerical_columns:
      feature_columns.append(feature_column.numeric_column(header))

   for col in cat_vocabulary.keys():
      try:
         vocab = cat_vocabulary[col]
      except:
         print(col)
      column = feature_column.categorical_column_with_vocabulary_list(col, vocab)
      column_one_hot = feature_column.indicator_column(column)
      feature_columns.append(column_one_hot)

   feature_layer = tf.keras.layers.DenseFeatures(feature_columns, autocast=True)
   return feature_layer


def main():
   # cat_voc, num_cols
   tf.keras.backend.set_floatx('float64')
   dataframe = load_and_trim()
   train, test, val, cat_voc, num_cols = split_data(dataframe=dataframe)

   feature_layer = compile_feature_layer(cat_vocabulary=cat_voc, numerical_columns=num_cols)

   batch_size = train_hyperparameters['batch_size']  # A small batch sized is used for demonstration purposes
   num_epochs = train_hyperparameters['num_epochs']

   train_ds = df_to_dataset(train, batch_size=batch_size)
   val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
   test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

   tf.keras.backend.set_floatx('float64')
   num_of_distinct_classes = len(classification_parameters['classes'])
   # labels = classification_parameters['classes']
   model = tf.keras.Sequential([
      feature_layer,
      layers.Dense(units=128, activation='relu'),
      layers.Dense(units=128, activation='relu'),
      layers.Dense(units=num_of_distinct_classes, activation='sigmoid')
   ])
   the_optimizer = RMSprop()
   # target = classification_parameters['classes']
   # output = target
   # the_loss_function = binary_crossentropy(target, output, from_logits=True)
   model.compile(optimizer=the_optimizer,
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])
   callbacks = [
      TensorBoard(
         log_dir='my_log_dir',
         histogram_freq=1,
         embeddings_freq=1,
      ),
      EarlyStopping(
         monitor='accuracy',
         patience=1,
      ),
      ModelCheckpoint(
         filepath='my_model.h5',
         monitor='val_loss',
         save_best_only=True,
      ),
      ReduceLROnPlateau(
         monitor='val_loss',
         factor = 0.1,
         patience = 10
      )
   ]

   history=model.fit(train_ds,
             epochs=num_epochs,
             callbacks=callbacks,
             validation_data=val_ds
              )
   model.summary()
   loss, accuracy = model.evaluate(test_ds)
   model.save('saved_models/dnn_model')
   print("Accuracy", accuracy)



if __name__ == '__main__':
   main()
