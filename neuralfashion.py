import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow import keras 
fashion_mnist=keras.datasets.fashion_mnist
(dftrain,y_train),(dftest,y_test)=fashion_mnist.load_data()
articles=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
dftrain=dftrain/255.0
dftest=dftest/255.0
model =keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(dftrain,y_train,epochs=9)
test_loss,test_acc=model.evaluate(dftest,y_test,verbose=1)
print('accuracy',test_acc)
