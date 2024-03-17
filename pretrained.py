import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
keras=tf.keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
(raw_train,raw_validation,raw_test), metadata=tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]','train[80%:90%]','train[90%:]'],
    with_info=True,
    as_supervised=True
)
get_label_name=metadata.features['label'].int2str

for image,label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
#we need to resize the images so that they are of same size
img_size=160
def resizeimg(image,label):
  image=tf.cast(image,tf.float32)
  image=(image/127.5) -1
  image=tf.image.resize(image,(img_size,img_size))
  return image,label
train=raw_train.map(resizeimg)
validation=raw_validation.map(resizeimg)
test=raw_test.map(resizeimg)
BATCH_SIZE=32
SHUFFLE_BUFFER_SIZE=1000
train_batches=train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches=validation.batch(BATCH_SIZE)
test_batches=test.batch(BATCH_SIZE)
for image,label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
  #comparing shape of old image and new image
for image,label in train.take(2):
    print(image.shape)
for image,label in raw_train.take(2):
    print(image.shape)
  input_image_shape=(img_size,img_size,3)
base_model=tf.keras.applications.MobileNetV2(input_shape=input_image_shape,
                                             include_top=False,
                                             weights='imagenet')
base_model.summary()
base_model.trainable=False#we don't want to train millions of parameters 
#we want to add our classifying layer on top 
global_average_layers=tf.keras.layers.GlobalAveragePooling2D()
prediction_layer=tf.keras.layers.Dense(1)
model=tf.keras.Sequential([base_model,
                          global_average_layers,
                          prediction_layer])
base_learning_rate=0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
          metrics=['Accuracy']    )

for image,_ in train_batches.take(1):
  pass

feature_batch=base_model(image)
print(feature_batch.shape)


validation_steps=20
epochs=3
loss1,accuracy1=model.evaluate(validation_batches,steps=validation_steps)
history =model.fit(train_batches,
                   epochs=epochs,
                   validation_data=validation_batches
                   )
accu=history.history['Accuracy']
print(accu)
model.save('catsVdogsmark1')
new_model=tf.keras.models.load_model('catsVdogsmark1')
