!pip install -q sklearn
%tensorflow_version 2.x
  #importing important libraries
from __future__ import  absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
#loading data files  testing and training 
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
#output label data 
y_train=dftrain.pop('survived')
y_eval=dfeval.pop('survived')
#grouping into categorical features 
NUMERICAL_COLUMNS=["age","fare"]
CATEGORICAL_COLUMNS=["sex","n_siblings_spouses","parch","class","deck","embark_town","alone"]
feature_columns=[]
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary=dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))
for feature_name in NUMERICAL_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))
#function to create suitable input data with proper batch size and  sequence (epochs)
def make_input_fn(data_df,label_df,num_epochs=10,shuffle=True,batch_size=32):
  def input_function():
    ds=tf.data.Dataset.fromt_tensor_slices(dict(data_df),label_df)
    if shuffle:
     ds=ds.shuffle(1000)
    ds=ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function
