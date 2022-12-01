import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2


parent_path = r'C:\Users\a_gadari\OneDrive - UNCG\Documents\bdml'
os.chdir(parent_path)


df = pd.read_csv('./dataframes\8xblocks_labelled_dataset_mak_eroded.csv')
res =pd.read_csv('./dataframes\cnn_shuffling.csv')


for i in range(10):
    df0 = df[df.labels==0]
    df1 = df[df.labels==1]
    r = df1.shape[0]
    df0 = df0.sample(r)


    df = pd.concat([df1,df0],axis=0)
    df = df.sample(frac=1)
    labels = np.array(df['labels'])
    data = np.array(df.drop(columns='labels'))



    X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify = labels, test_size=0.33, random_state=21)

    x_label = ['Total','train','test']
    y1 = np.array([sum(labels)/df.shape[0],sum(y_train)/X_train.shape[0],sum(y_test)/X_test.shape[0]])
    y2 = np.array([100]*3) - y1
    print(y1, y2)
    X_axis = np.arange(len(x_label))

    model = models.Sequential()
    model.add(layers.Conv2D(4, (3, 3), input_shape=(8, 8, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['Recall'])

    print(model.summary())

    train_images = np.array([np.array(i).reshape(8,8) for i in list(X_train)])
    test_images = np.array([np.array(i).reshape(8,8) for i in list(X_test)])

    history = model.fit(train_images, np.array(y_train), epochs=10, validation_data=(test_images, np.array(y_test)))

    pred = model.predict(test_images)
    pred = [1 if x>0.5 else 0 for x in pred]
    tn,fp,fn,tp = confusion_matrix(y_test,pred).ravel()
    res.loc[i,'TN'] = tn
    res.loc[i,'FP'] = fp
    res.loc[i,'FN'] = fn
    res.loc[i,'TP'] = tp

    print(tn,fp,fn,tp)
#res.to_csv('./dataframes\cnn_shuffling.csv', index=False)
df = pd.read_csv('./dataframes\8xblocks_labelled_dataset_mak_eroded.csv')
pred = model.predict(np.array([np.array(i).reshape(8,8) for i in np.array(df.drop(columns='labels'))]))

nim= np.zeros((512,512),dtype=np.uint8)
k=0
for i in range(int(512/8)):
    for j in range(int(512/8)):
        if pred[k]>0.5:
            nim[i:i+8,j:j+8] = 255
        k+=1

cv2.imwrite('result_im.png',nim)
    

