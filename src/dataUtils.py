import os
import csv
import pylab 
import numpy as np
import cPickle as pickle

""" dataDir hardcoded for test porposes only. Should not be"""
dataDir = '../Data/'

data1 = np.array([]).reshape(0,8)
data2 = np.array([]).reshape(0,8)
data3 = np.array([]).reshape(0,8)
data4 = np.array([]).reshape(0,8)
data5 = np.array([]).reshape(0,8)
data6 = np.array([]).reshape(0,8)

def loadDataSVM():
    
    global data1, data2, data3, data4, data5, data6
    
    nDataFiles = 2000
    nData = 6*nDataFiles
    nTrainSamples = 1900*6
    nTestData = nData - nTrainSamples
    
    data1 = pickle.load( open( "data1.p", "rb" ) )
    data2 = pickle.load( open( "data2.p", "rb" ) )
    data3 = pickle.load( open( "data3.p", "rb" ) )
    data4 = pickle.load( open( "data4.p", "rb" ) )
    data5 = pickle.load( open( "data5.p", "rb" ) )
    data6 = pickle.load( open( "data6.p", "rb" ) )
    
    trainData = np.zeros((nTrainSamples, 400), dtype="uint16")
    testData = np.zeros((nTestData, 400), dtype="uint16")
    yTrain = np.zeros((nTrainSamples), dtype="uint8")
    yTest = np.zeros((nTestData,), dtype="uint8")
    
    for i,j in zip(range(1,38),range(0,1900,6)):
        trainData[i-1] = np.reshape(data1[(i-1) * 50: i*50], 400)
        
    for i,j in zip(range(1,38),range(0,1900,6)):
        trainData[j] = np.reshape(data2[(i-1) * 50: i*50], 400)
        
        
        trainData[j] = np.reshape(data3[(i-1) * 50: i*50], 400)
        
        
        
        
def loadData(usePickle = True, loadFromFile = False):
    """
    Function to load the data into matrix [n_files * 50][8<channels>]
    3/4 of the files will used for trainning and 1/4 to test
    Number of files per gesture and train/test ratio hardcoded for now. 
    """
    
    global data1, data2, data3, data4, data5, data6
    
    nDataFiles = 2000
    nData = 6*nDataFiles
    nTrainSamples = 1900*6
    nTestData = nData - nTrainSamples
    
    trainData = np.zeros((nTrainSamples, 1, 50, 8), dtype="uint16")
    testData = np.zeros((nTestData, 1, 50, 8), dtype="uint16")
    yTrain = np.zeros((nTrainSamples,), dtype="uint8")
    yTest = np.zeros((nTestData,), dtype="uint8")
    
   
    yData = np.zeros(nData, dtype="uint8") 
    counter = 0;
    
    for i in range(11400):
        if 0 <= i < 1900:
            yTrain[counter] = 0
        elif 1900 <= i < 3800:
            yTrain[counter] = 1
        elif 3800 <= i < 5700:
            yTrain[counter] = 2
        elif 5700 <= i < 7600:
            yTrain[counter] = 3
        elif 7600 <= i < 9500:
            yTrain[counter] = 4
        elif 9500 <= i < 11400:
            yTrain[counter] = 5
        counter += 1
    counter = 0
    for i in range(600):
        if 0 <= i < 100:
            yTest[counter] = 0
        elif 100 <= i < 200:
            yTest[counter] = 1
        elif 200 <= i < 300:
            yTest[counter] = 2
        elif 300 <= i < 400:
            yTest[counter] = 3
        elif 400 <= i < 500:
            yTest[counter] = 4
        elif 500 <= i < 600:
            yTest[counter] = 5
        counter += 1
    counter = 0
    if(usePickle):
        trainData = pickle.load(open( "train.p", "rb" ) )
        testData = pickle.load(open( "test.p", "rb" ) )
        
    elif(loadFromFile):
    
        for filename in os.listdir(dataDir):
            gesture = int(filename[7:8])
            with open(dataDir + filename, 'r') as data:
                
                reader = csv.reader(data, delimiter=',')
                           
                if gesture == 1:
                    for row in reader:
                        data1 = np.r_[data1, [row]]
                if gesture == 2:
                    for row in reader:
                        data2 = np.r_[data2, [row]]
                if gesture == 3:
                    for row in reader:
                        data3 = np.r_[data3, [row]]
                if gesture == 4:
                    for row in reader:
                        data4 = np.r_[data4, [row]]
                if gesture == 5:
                    for row in reader:
                        data5 = np.r_[data5, [row]]
                if gesture == 6:
                    for row in reader:
                        data6 = np.r_[data6, [row]]
    
        pickle.dump(data1, open( "data1.p", "wb" ) )
        pickle.dump(data2, open( "data2.p", "wb" ) )
        pickle.dump(data3, open( "data3.p", "wb" ) )
        pickle.dump(data4, open( "data4.p", "wb" ) )
        pickle.dump(data5, open( "data5.p", "wb" ) )
        pickle.dump(data6, open( "data6.p", "wb" ) )
        
    else:
        data1 = pickle.load( open( "data1.p", "rb" ) )
        data2 = pickle.load( open( "data2.p", "rb" ) )
        data3 = pickle.load( open( "data3.p", "rb" ) )
        data4 = pickle.load( open( "data4.p", "rb" ) )
        data5 = pickle.load( open( "data5.p", "rb" ) )
        data6 = pickle.load( open( "data6.p", "rb" ) )
        """Train Data"""
        for i in range(0,1900):        
            trainData[i] = data1[(i) * 50: (i+1)*50]
            
        for i,j in zip(range(1900,3800),range(1,1900)):
            trainData[i] = data2[(j) * 50: (j+1)*50]
            
        for i,j in zip(range(3800,5700),range(1,1900)):
            trainData[i] = data3[(j) * 50: (j+1)*50]
            
        for i,j in zip(range(5700,7600),range(1,1900)):
            trainData[i] = data4[(j) * 50: (j+1)*50]
            
        for i,j in zip(range(7600,9500),range(1,1901)):
            trainData[i] = data5[(j) * 50: (j+1)*50]
            
        for i,j in zip(range(9500,11400),range(1,1900)):
            trainData[i] = data6[(j) * 50: (j+1)*50]
            
        """ Test Data"""
        for i,j in zip(range(0,100),range(1900,2000)):
            testData[i] = data1[(j-1) * 50: j*50]
        for i,j in zip(range(100,200),range(1900,2000)):
            testData[i] = data2[(j-1) * 50: j*50]
            
        for i,j in zip(range(200,300),range(1900,2000)):
            testData[i] = data3[(j-1) * 50: j*50]
    
        for i,j in zip(range(300,400),range(1900,2000)):
            testData[i] = data4[(j-1) * 50: j*50]
            
        for i,j in zip(range(400,500),range(1900,2000)):
            testData[i] = data5[(j-1) * 50: j*50]
            
        for i,j in zip(range(500,600),range(1900,2000)):
            testData[i] = data6[(j-1) * 50: j*50]
        
        
        print yTrain, yTrain.shape
        pickle.dump(trainData, open( "train.p", "wb" ) )
        pickle.dump(yTrain, open( "ytrain.p", "wb" ) )
        pickle.dump(testData, open( "test.p", "wb" ) )
        pickle.dump(yTest, open( "ytest.p", "wb" ) )
        
    yTrain = np.reshape(yTrain, (len(yTrain), 1))
    yTest = np.reshape(yTest, (len(yTest), 1))       
    
    return (trainData, yTrain),(testData, yTest)
    
def plotData(gesture):
    """Plot the data to see distribuiton and behaviour """
    
    x = np.arange(0,8,1)
    
    if gesture == 1:
        for i in range(len(data1)):
            pylab.plot(x, data1[i], 'bo')
    
    elif gesture == 2:
        for i in range(len(data2)):
            pylab.plot(x, data2[i], 'bo')
    
    elif gesture == 3:
        for i in range(len(data3)):
            pylab.plot(x, data3[i], 'bo')
    
    elif gesture == 4:
        for i in range(len(data4)):
            pylab.plot(x, data4[i], 'bo')
    
    elif gesture == 5:
        for i in range(len(data5)):
            pylab.plot(x, data5[i], 'bo')
    
    elif gesture == 6:
        for i in range(len(data6)):
            pylab.plot(x, data6[i], 'bo')
    
    pylab.show()



