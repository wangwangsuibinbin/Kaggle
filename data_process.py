import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

data = pd.read_csv('train.csv')
dataset = data.iloc[:,1:]

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

y_train , x_train= train[:,0:1],train[:,1:]
y_test ,x_test = test[:,0:1],test[:,1:]
x_train,x_test = np.array(x_train).reshape(-1,4991,1),np.array(x_test).reshape(-1,4991,1)
seq_length = x_train.shape[1]

model = Sequential()
model.add(Conv1D(128, 3, activation='relu', input_shape=(seq_length, 1)))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=32)
