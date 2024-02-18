import pandas as pd
import tensorflow as tf
from __future__ import absolute_import, division , print_function, unicode_literals
CSV_COLUMN_NAMES=['sepallength','sepalwidth','petallength','petalwidth','species']
SPECIES=['setosa','versicolor','virginica']
train_path=tf.keras.utils.get_file('iris_training.csv',"https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path=tf.keras.utils.get_file('iris_testing.csv',"https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train=pd.read_csv(train_path,names=CSV_COLUMN_NAMES,header=0)
test=pd.read_csv(test_path,names=CSV_COLUMN_NAMES,header=0)
train.head()
train_y=train.pop('species')
test_y=test.pop('species')
def input_fn(features,labels,training=True,batch_size=256):
  ds=tf .data.Dataset.from_tensor_slices((dict(features),labels))
  if training:
    ds=ds.shuffle(1000).repeat()
  return ds.batch(batch_size)
my_features=[]
for key in train.keys():
  my_features.append(tf.feature_column.numeric_column(key=key))
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_features,
    hidden_units=[30,10],
    n_classes=3
)
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)
eval_result=classifier.evaluate(input_fn=lambda: input_fn(test,test_y,training=False))
print('{accuracy:0.3f}'.format(**eval_result))
def new_input_fn(features,batch_size=256):
    ds=tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
    return ds



features=['sepallength','sepalwidth','petallength','petalwidth']
predict={}
print("enter features here ")
for feature in features:
  valid =True
  while valid:
     val=input(feature+': ')
     if not val.isdigit(): valid =False

  predict[feature]=[float(val)]

predictions=classifier.predict(lambda: new_input_fn(predict))
for pred in predictions:
    class_id=pred['class_ids'][0]
    prob=pred['probabilities'][class_id]

    print('prediction is"{}" ({:.1f}%)'.format(SPECIES[class_id],prob*100))
