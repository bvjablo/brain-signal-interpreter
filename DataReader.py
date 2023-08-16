import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

#This function reads in the data from the "blink" and "no_blink" files and combines them into an array that can be processed by TensorFlow
def dataReader(path_blink, path_no_blink, min_rows, max_rows):
    #Lists for storing the names of the .txt files
    txt_list_blink = []
    txt_list_no_blink = []
    numFiles = len(txt_list_blink + txt_list_no_blink)

    #Reads in the names of the files and stores them in their respecive list
    for iter in os.listdir(path_blink):
        if (iter.endswith(".txt")):
            txt_list_blink.append(path_blink+"/"+iter)
    for iter in os.listdir(path_no_blink):
        if (iter.endswith(".txt")):
            txt_list_no_blink.append(path_no_blink+"/"+iter)

    max_length = 0
    list_len = len(txt_list_blink)
    i = 0
    while i < list_len:
        currFile = np.loadtxt(txt_list_blink[i], delimiter=",", skiprows=5, usecols=range(0))
        if (int(np.shape(currFile)[0]) < min_rows) or (int(np.shape(currFile)[0]) > max_rows):
            #(blink) file removed and variables adjusted accordingly
            txt_list_blink.remove(txt_list_blink[i])
            i -= 1
            list_len -= 1
        else:
            if int(np.shape(currFile)[0]) > max_length:
                max_length = int(np.shape(currFile)[0])
        i += 1

    list_len = len(txt_list_no_blink)
    i = 0
    while i < list_len:
        currFile = np.loadtxt(txt_list_no_blink[i], delimiter=",", skiprows=5, usecols=range(0))
        if (int(np.shape(currFile)[0]) < min_rows) or (int(np.shape(currFile)[0]) > max_rows):
            txt_list_no_blink.remove(txt_list_no_blink[i])
            #(no_blink) file removed and variables adjusted accordingly
            i -= 1
            list_len -= 1
        else:
            if int(np.shape(currFile)[0]) > max_length:
                max_length = int(np.shape(currFile)[0])
        i += 1

    allFiles = txt_list_blink + txt_list_no_blink
    numFiles = len(txt_list_blink) + len(txt_list_no_blink)

    #allData will be what holds all of the EEG data
    #allData_Output will hold a correspending '1' for blinks and '0' for non blinks
    allData = np.empty(shape=(numFiles, max_rows, 15))
    allData_Output = np.empty(shape=numFiles)

    #Now we can read in all the data and pad the smaller inputs with zeros
    for i in range(numFiles):

        #Adjust usecols to modify which channels will be read in
        newData = np.loadtxt(allFiles[i], delimiter=",", skiprows=105, usecols=range(1,16))
        padAmount = (max_rows - (int(np.shape(newData)[0])))
        newData = np.pad(newData, ((0, padAmount),(0,0)), 'constant', constant_values=(0))
        allData[i] = newData
        if ((i+1)%100) == 0:
            print("%d Files Processed" % (i+1))

    for i in range(numFiles):
        if i < len(txt_list_blink):
            allData_Output[i] = 1
        else:
            allData_Output[i] = 0

    #Used to test y output
    print(allData_Output[(len(txt_list_blink)-1)])
    print(allData_Output[len(txt_list_blink)])

    #Shape: allData[file][timestep][channel]
    return allData, allData_Output

x, y = dataReader("path to blink data", "path to no blink data", 400, 800) #400-800 timestep files included 

print("\nTrain data shape: ")
print(np.shape(x))

print("\nTest data shape: ")
print(np.shape(y))

#Be sure to change pickle names accordingly
pickle_out = open("X_Train.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("Y_Train.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
