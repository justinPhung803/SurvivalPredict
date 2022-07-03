from typing import Counter
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import keras
from keras.utils import to_categorical
from keras import Sequential
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

data=pd.read_csv("heart_failure.csv")
print(data.info())
print(Counter(data["death_event"]))

y=data["death_event"]
x=data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
x=pd.get_dummies(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.25,random_state=42)

ct=ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

x_train=ct.fit_transform(x_train)
x_test=ct.transform(x_test)

le=LabelEncoder()

y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

def model_design(model_feature):
    model=Sequential()
    model.add(layers.InputLayer(input_shape=(model_feature.shape[1], )))
    model.add(layers.Dense(12,activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(2,activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

model=model_design(x_train)
model.fit(x_train,y_train,epochs=100, batch_size=16)

loss, acc=model.evaluate(x_test, y_test)

y_estimate=model.predict(x_test)

y_estimate=np.argmax(y_estimate, axis=1)
y_true=np.argmax(y_test, axis=1)

print(classification_report(y_true, y_estimate))