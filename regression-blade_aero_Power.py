# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:58:06 2021

@author: Chao Zhang, Ph.D.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns

# Make numpy printouts easier to read.
#np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Define Hyperparameters
Num_neurons_layer1 = 20
Num_neurons_layer2 = 40
Num_neurons_layer3 = 20
Num_neurons_layer4 = 5
Activation_func_layer1 = 'relu'
Activation_func_layer2 = 'relu'
Activation_func_layer3 = 'relu'
Activation_func_layer4 = 'relu'
Learning_rate = 0.001
Num_epochs = 200


print(tf.__version__)

raw_dataset = pd.read_csv('D:/ChaoFiles/MyProjects/2021-06-ANN_Guide_3D_Blade_Design/tensorflow_model_building/HEEDS0_data.res')
dataset = raw_dataset.copy()
dataset.columns = dataset.columns.str.strip()
dataset.info()

train_dataset = dataset.sample(frac=0.95, random_state=None)
test_dataset = dataset.drop(train_dataset.index)
#sns.pairplot(train_dataset[['Isentropic_Efficiency', 'Total_Enthalpy_Change']], diag_kind='kde')


train_features = train_dataset.iloc[:,18:38]
test_features = test_dataset.iloc[:,18:38]
train_labels = train_dataset[['Isentropic_Efficiency', 'Total_Enthalpy_Change']]
test_labels = test_dataset[['Isentropic_Efficiency', 'Total_Enthalpy_Change']]

normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

def build_and_compile_Power(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(int(Num_neurons_layer1), activation=Activation_func_layer1),
      layers.Dense(int(Num_neurons_layer2), activation=Activation_func_layer2),
      layers.Dense(int(Num_neurons_layer3), activation=Activation_func_layer3),
      layers.Dense(int(Num_neurons_layer4), activation=Activation_func_layer4),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(Learning_rate))
  return model

dnn_Power = build_and_compile_Power(normalizer)
dnn_Power.summary()

history_Power = dnn_Power.fit(
    train_features, train_labels['Total_Enthalpy_Change'],
    validation_split=0.2,
    verbose=1, epochs=int(Num_epochs))

print(dnn_Power.evaluate(test_features, test_labels['Total_Enthalpy_Change'], verbose=0))

plt.figure(figsize=(15,8))
plt.plot(history_Power.history['loss'], label='loss')
plt.plot(history_Power.history['val_loss'], label='validation_loss')
plt.ylim([10, 10000])
plt.xlabel('Epoch',fontsize=24)
plt.yscale("log")
plt.ylabel('Error',fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig('history_Power.png')

test_predictions = dnn_Power.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.figure(figsize=(10,10))
plt.scatter(test_labels['Total_Enthalpy_Change'].to_numpy(), test_predictions)
plt.xlabel('True Values - Power',fontsize=20)
plt.ylabel('Predictions - Power',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
lims = [1E6, 1.6E6]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('Predicted vs. True.png')

x = range(1,len(test_predictions)+1)
plt.figure(figsize=(20,10))
plt.scatter(x,test_labels['Total_Enthalpy_Change'].to_numpy(),marker='o')
plt.scatter(x,test_predictions,marker='^')
plt.xlabel('Design ID',fontsize=20)
plt.ylabel('Power',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
lims = [1E6, 1.6E6]
plt.ylim(lims)
plt.savefig('Predicted vs. True 2.png')

error = test_predictions - test_labels['Total_Enthalpy_Change'].to_numpy()
plt.figure(figsize=(15,10))
plt.hist(error, bins=25)
plt.xlabel('Prediction Error - Power',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('Error histogram.png')

file = open("EvalError.txt","w+")
file.write(str(dnn_Power.evaluate(test_features, test_labels['Total_Enthalpy_Change'], verbose=0)))
file.close()
print("Complete!")
