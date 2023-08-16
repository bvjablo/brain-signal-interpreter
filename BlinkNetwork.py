#Convolutional Neural Network Test
# Note: This does not load dataset for the code to run. 
import numpy as np
import pickle
import tensorflow as tf
import tensorboard
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from datetime import datetime

x = pickle.load(open("X_Train.pickle", "rb"))
y = pickle.load(open("Y_Train.pickle", "rb"))

num_classes = 2
batch_size = 30
num_epochs = 3

print(np.shape(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True)

input_shape = np.shape(x_train)

graph = tf.Graph()

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape[1],input_shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu')) #Rectified Linear-> Linear when > 0 (simple computation)
model.add(Dense(1, activation='sigmoid')) #Sigmoid-> between 0 and 1 

#Binary crossentropy for either 'blink' or 'no blink'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

writer = tf.summary.create_file_writer(logdir)

# Here we define the writer and when we access tensorboard we will be able to visualize the graph
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

#This is data not included in the training set to test the model in a real scenario
x_check = pickle.load(open("X_Data.pickle", "rb"))
y_check = pickle.load(open("Y_Data.pickle", "rb"))
predictions = model.predict(x_check)

#Note: The training data is randomized before being fed to the model. The order of blinks and non-blinks does not affect the model's outputs
print('\nPrediction | Actual')
for i in range(len(y_check)):
    predict = 0 if (predictions[i][0] < 0.5) else 1
    actual = 0 if (y_check[i] < 0.5) else 1
    accuracy = "correct" if predict == actual else "incorrect"
    print(f"{predict}|{actual}   {accuracy}")


