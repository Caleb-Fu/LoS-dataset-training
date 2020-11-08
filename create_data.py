
from urllib.request import urlopen
from io import StringIO
import csv
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dropout
from keras.layers.core import Dense
from keras.optimizers import Nadam
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


Epoch = 50
Input = 24
Output = 1
Hidden = 20
OPTIMIZER = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004) #learning rate = 0.01



'''
def calcTime(begin, end):
    bT = begin.split('/')
    eT = end.split('/')
    if eval(eT[0]) == eval(bT[0]):
        return eval(eT[1]) - eval(bT[1])
    else:
        if eval(bT[0]) == 2:
            return 29+eval(eT[1]) - eval(bT[1])
        elif eval(bT[0]) in [1, 3, 5, 7, 8, 10, 12]:
            return 31+eval(eT[1]) - eval(bT[1])
        else:
            return 30+eval(eT[1]) - eval(bT[1])
'''
def checkLabel(label):
    if label == 'C':
        return 30
    elif label == 'E':
        return 95
    elif label == 'A':
        return 75
    elif label == 'B':
        return 80
    else:
        return 100

def preprocess():
    dataset, results = [], []
    data=urlopen("https://raw.githubusercontent.com/microsoft/r-server-hospital-length-of-stay/master/Data/LengthOfStay.csv").read().decode('ascii','ignore')
    dataFile=StringIO(data)
    csvReader=csv.reader(dataFile)
    for row in csvReader:
        dataset.append(row)
    del dataset[0]
    for data in dataset:
        del data[0]
        del data[0]
        del data[-3]
        if data[1] == 'F':
            data[1] = '0'
        else:
            data[1] = '1'
        data[-2] = str(checkLabel(data[-2]))
        data.append(data[-1])
        results.append(eval(data.pop(-1)))
        if data[0] == '5+':
            data[0] = '1'
        else:
            data[0] = '0'
        for i in range(len(data)):
            data[i] = eval(data[i])
    '''
    Max = np.max(dataset, axis = 0)
    Min = np.min(dataset, axis = 0)
    for item in dataset:
        for j in range(len(item)):
            item[j] = (item[j]-Min[j])/(Max[j]-Min[j])
    '''
    x_train = StandardScaler().fit_transform(np.array(dataset[:80000]))
    x_test = StandardScaler().fit_transform(np.array(dataset[80000:]))
    pca1 = PCA()
    pca1.fit(x_train)
    pca1.fit(x_test)
    #explained_variance_ratio = pca1.explained_variance_ratio_
    #pc_vs_variance = np.cumsum(pca1.explained_variance_ratio_)
    return x_train, np.array(results[:80000]), x_test, np.array(results[80000:])



def buildModel():
    train_X, test_X, train_Y, test_Y = preprocess()
    model_sklearn = MLPClassifier(max_iter=10000, hidden_layer_sizes=(12,32,15), activation='relu', learning_rate_init=0.001, )
    model_sklearn.fit(train_X, test_X)

    pred_y_test_sklearn = model_sklearn.predict(train_Y)

    accu = accuracy_score(test_Y, pred_y_test_sklearn)
    print("Accuracy: ", accu)
    print('Confusion matrix of testing datasets is: \n', confusion_matrix(test_Y, pred_y_test_sklearn))



    

'''
    #get preprocessed data.
    trainData, trainResults, testData, testResults=preprocess()
    
    #build the model.
    model = Sequential()
    model.add(Dense(Hidden, input_shape=(Input,)))
    model.add(Activation('relu')) #use sigmoid function at each layer.
    model.add(Dense(Output))
    model.add(Activation('softmax'))
    print("There is Model Information:")
    model.summary()
    
    #use MSE as its loss and Adam gradiant descent as optimizer.
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    history = model.fit(trainData, trainResults, epochs=Epoch,verbose=0) #training process.

    #test the testing and training datasets.
    lossTest, accuracyTest = model.evaluate(trainData, trainResults, verbose=0)
    #predictedTest = np.argmax(model.predict(testData), axis=-1) #get a list containing all predict labels.
    #lossTrain, accuracyTrain = model.evaluate(trainData, trainResults, verbose=0)
    #predictedTrain = np.argmax(model.predict(trainData), axis=-1)



    #print out test results.
    #outputInfo(accuracyTest, accuracyTrain, lossTest, lossTrain, actualLabelTest, actualLabelTrain, predictedTest, predictedTrain, Epoch)
    print(accuracyTest)
'''
'''
def outputInfo(aTest, aTrain, lTest, lTrain, LabelTest, LabelTrain, pTest, pTrain, Epoch):
    print(len(LabelTest))
    
    #output the features.
    print("*" *30)
    print("**", " "*15, "Epoch: ", Epoch, " "*15, "**") 
    print("*"*30)
    print('Accuracy of testing datasets is: ', round(aTest, 5)*100, '%')
    print('MSE of testing datasets is: ', round(lTest, 4))
    print('Precision of testing dataset is: ', round(precision_score(LabelTest, pTest, average='macro'), 4))
    print('Recall of testing datasets is: ', round(recall_score(LabelTest, pTest, average='macro'), 4))
    print('Confusion matrix of testing datasets is: \n', confusion_matrix(LabelTest, pTest)) #build confusion matrix.
    print("-"*70)

    #test the training datasets.
    print('Accuracy of testing datasets is: ', round(aTrain, 5)*100, '%')
    print('MSE of testing datasets is: ', round(lTrain, 4))
    print('Precision of testing dataset is: ', round(precision_score(LabelTrain, pTrain, average='macro'), 4))
    print('Recall of testing datasets is: ', round(recall_score(LabelTrain, pTrain, average='macro'), 4))
    print('Confusion matrix of testing datasets is: \n', confusion_matrix(LabelTrain, pTrain))
    print("="*16, "Epoch", Epoch, "End", "="*16)
'''
def main():
    
    buildModel() #model with same features as module 1.
main()








         

