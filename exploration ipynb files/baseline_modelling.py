import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


parent_path = r'C:\Users\a_gadari\OneDrive - UNCG\Documents\bdml'
os.chdir(parent_path)


df = pd.read_csv('./dataframes\8xblocks_labelled_dataset_mak_eroded.csv')
labels = np.array(df['labels'])
data = np.array(df.drop(columns='labels'))

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify = labels, test_size=0.33, random_state=21)

'''
x_label = ['Total','train','test']
y1 = np.array([sum(labels)/df.shape[0],sum(y_train)/X_train.shape[0],sum(y_test)/X_test.shape[0]])
y2 = np.array([100]*3) - y1
print(y1, y2)
X_axis = np.arange(len(x_label))

plt.bar(X_axis - 0.2, y2,0.4, label = 'label 0')
plt.bar(X_axis + 0.2, y1,0.4, label = 'label 1')
plt.show()
print(sum(y_train),sum(y_test))
'''

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['Recall'])

print(model.summary())

