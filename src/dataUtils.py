import os
import csv
import pylab
import numpy as np
import pickle
from matplotlib import pyplot

""" dataDir hardcoded for test porposes only. Should not be"""
dataDir = '../Data/'

data1 = np.array([]).reshape(0, 8)
data2 = np.array([]).reshape(0, 8)
data3 = np.array([]).reshape(0, 8)
data4 = np.array([]).reshape(0, 8)
data5 = np.array([]).reshape(0, 8)
data6 = np.array([]).reshape(0, 8)


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)


def rawpycount(filename):
    f = open(dataDir+filename, 'rb')
    f_gen = _make_gen(f.read)
    return sum(buf.count(b'\n') for buf in f_gen)


def loadDataSVM():

    global data1, data2, data3, data4, data5, data6

    nDataFiles = 2000
    nData = 6*nDataFiles
    nTrainSamples = 1900*6
    nTestData = nData - nTrainSamples

    data1 = pickle.load(open("data1.p", "rb"))
    data2 = pickle.load(open("data2.p", "rb"))
    data3 = pickle.load(open("data3.p", "rb"))
    data4 = pickle.load(open("data4.p", "rb"))
    data5 = pickle.load(open("data5.p", "rb"))
    data6 = pickle.load(open("data6.p", "rb"))

    trainData = np.zeros((nTrainSamples, 400), dtype="uint16")
    testData = np.zeros((nTestData, 400), dtype="uint16")
    yTrain = np.zeros((nTrainSamples), dtype="uint8")
    yTest = np.zeros((nTestData,), dtype="uint8")

    for i, j in zip(range(1, 38), range(0, 1900, 6)):
        trainData[i-1] = np.reshape(data1[(i-1) * 50: i*50], 400)

    for i, j in zip(range(1, 38), range(0, 1900, 6)):
        trainData[j] = np.reshape(data2[(i-1) * 50: i*50], 400)

        trainData[j] = np.reshape(data3[(i-1) * 50: i*50], 400)


def loadData(usePickle=True, loadFromFile=False):
    """
    Function to load the data into matrix [n_files * 50][8<channels>]
    3/4 of the files will used for trainning and 1/4 to test
    Number of files per gesture and train/test ratio hardcoded for now. 
    """

    global data1, data2, data3, data4, data5, data6

    nTrainFiles = 1895
    nTestFiles = 100
    nGestures = 6
    nfileLines = 50
    nChannels = 8
    nTrainSamples = nTrainFiles * nGestures
    nTestSamples = nTestFiles * nGestures

    trainData = np.zeros(
        (nTrainSamples, 1, nfileLines, nChannels), dtype="float32")
    testData = np.zeros(
        (nTestSamples, 1, nfileLines, nChannels), dtype="float32")

    yTrain = np.zeros((nTrainSamples,), dtype="float32")
    yTest = np.zeros((nTestSamples,), dtype="float32")

    counter = 0

    for i in range(nTrainSamples):
        if 0 <= i < nTrainFiles:
            yTrain[counter] = 0
        elif nTrainFiles <= i < nTrainFiles * 2:
            yTrain[counter] = 1
        elif nTrainFiles * 2 <= i < nTrainFiles * 3:
            yTrain[counter] = 2
        elif nTrainFiles * 3 <= i < nTrainFiles * 4:
            yTrain[counter] = 3
        elif nTrainFiles * 4 <= i < nTrainFiles * 5:
            yTrain[counter] = 4
        elif nTrainFiles * 5 <= i < nTrainFiles * 6:
            yTrain[counter] = 5
        counter += 1

    counter = 0
    for i in range(nTestSamples):
        if 0 <= i < nTestFiles:
            yTest[counter] = 0
        elif nTestFiles <= i < nTestFiles * 2:
            yTest[counter] = 1
        elif nTestFiles * 2 <= i < nTestFiles * 3:
            yTest[counter] = 2
        elif nTestFiles * 3 <= i < nTestFiles * 4:
            yTest[counter] = 3
        elif nTestFiles * 4 <= i < nTestFiles * 5:
            yTest[counter] = 4
        elif nTestFiles * 5 <= i < nTestFiles * 6:
            yTest[counter] = 5
        counter += 1

    counter = 0
    if(usePickle):
        trainData = pickle.load(open("train.p", "rb"))
        testData = pickle.load(open("test.p", "rb"))

    elif(loadFromFile):
        gesture1 = 0
        gesture2 = 0
        gesture3 = 0
        gesture4 = 0
        gesture5 = 0
        gesture6 = 0
        print('Starting on File reading')
        for filename in os.listdir(dataDir):
            count = rawpycount(filename)
            if(count < 50):
                print(filename)
                continue

            print('reading File ', filename)
            gesture = int(filename[7:8])

            with open(dataDir + filename, 'r') as data:
                reader = csv.reader(data, delimiter=',')
                if gesture == 1:
                    gesture1 += 1
                    for row in reader:
                        data1 = np.r_[data1, [row]]
                if gesture == 2:
                    gesture1 += 2
                    for row in reader:
                        data2 = np.r_[data2, [row]]
                if gesture == 3:
                    gesture1 += 3
                    for row in reader:
                        data3 = np.r_[data3, [row]]
                if gesture == 4:
                    gesture1 += 4
                    for row in reader:
                        data4 = np.r_[data4, [row]]
                if gesture == 5:
                    gesture1 += 5
                    for row in reader:
                        data5 = np.r_[data5, [row]]
                if gesture == 6:
                    gesture1 += 6
                    for row in reader:
                        data6 = np.r_[data6, [row]]

                data.close()

        print(gesture1, gesture2, gesture3, gesture4, gesture5, gesture6)

        pickle.dump(data1, open("data1.p", "wb"))
        pickle.dump(data2, open("data2.p", "wb"))
        pickle.dump(data3, open("data3.p", "wb"))
        pickle.dump(data4, open("data4.p", "wb"))
        pickle.dump(data5, open("data5.p", "wb"))
        pickle.dump(data6, open("data6.p", "wb"))

        for i in range(0, nTrainFiles):
            trainData[i] = data1[(i) * 50: (i+1)*50]

        for i, j in zip(range(nTrainFiles, nTrainFiles * 2), range(nTrainFiles)):
            trainData[i] = data2[(j) * 50: (j+1)*50]

        for i, j in zip(range(nTrainFiles * 2, nTrainFiles * 3), range(nTrainFiles)):
            trainData[i] = data3[(j) * 50: (j+1)*50]

        for i, j in zip(range(nTrainFiles * 3, nTrainFiles * 4), range(nTrainFiles)):
            trainData[i] = data4[(j) * 50: (j+1)*50]

        for i, j in zip(range(nTrainFiles * 4, nTrainFiles * 5), range(nTrainFiles)):
            trainData[i] = data5[(j) * 50: (j+1)*50]

        for i, j in zip(range(nTrainFiles * 5, nTrainFiles * 6), range(nTrainFiles)):
            trainData[i] = data6[(j) * 50: (j+1)*50]

        """ Test Data"""
        for i, j in zip(range(nTestFiles), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data1[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles, nTestFiles * 2), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data2[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles * 2, nTestFiles * 3), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data3[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles * 3, nTestFiles * 4), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data4[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles * 4, nTestFiles * 5), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data5[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles * 5, nTestFiles * 6), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data6[(j) * 50: (j + 1)*50]

        print(yTrain, yTrain.shape)
        pickle.dump(trainData, open("train.p", "wb"))
        pickle.dump(yTrain, open("ytrain.p", "wb"))
        pickle.dump(testData, open("test.p", "wb"))
        pickle.dump(yTest, open("ytest.p", "wb"))

    else:
        data1 = pickle.load(open("data1.p", "rb"))
        data2 = pickle.load(open("data2.p", "rb"))
        data3 = pickle.load(open("data3.p", "rb"))
        data4 = pickle.load(open("data4.p", "rb"))
        data5 = pickle.load(open("data5.p", "rb"))
        data6 = pickle.load(open("data6.p", "rb"))
        """Train Data"""
        for i in range(0, nTrainFiles):
            trainData[i] = data1[(i) * 50: (i+1)*50]

        for i, j in zip(range(nTrainFiles, nTrainFiles * 2), range(nTrainFiles)):
            trainData[i] = data2[(j) * 50: (j+1)*50]

        for i, j in zip(range(nTrainFiles * 2, nTrainFiles * 3), range(nTrainFiles)):
            trainData[i] = data3[(j) * 50: (j+1)*50]

        for i, j in zip(range(nTrainFiles * 3, nTrainFiles * 4), range(nTrainFiles)):
            trainData[i] = data4[(j) * 50: (j+1)*50]

        for i, j in zip(range(nTrainFiles * 4, nTrainFiles * 5), range(nTrainFiles)):
            trainData[i] = data5[(j) * 50: (j+1)*50]

        for i, j in zip(range(nTrainFiles * 5, nTrainFiles * 6), range(nTrainFiles)):
            trainData[i] = data6[(j) * 50: (j+1)*50]

        """ Test Data"""
        for i, j in zip(range(nTestFiles), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data1[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles, nTestFiles * 2), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data2[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles * 2, nTestFiles * 3), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data3[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles * 3, nTestFiles * 4), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data4[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles * 4, nTestFiles * 5), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data5[(j) * 50: (j + 1)*50]

        for i, j in zip(range(nTestFiles * 5, nTestFiles * 6), range(nTrainFiles, nTrainFiles+nTestFiles)):
            testData[i] = data6[(j) * 50: (j + 1)*50]

        print(yTrain, yTrain.shape)
        pickle.dump(trainData, open("train.p", "wb"))
        pickle.dump(yTrain, open("ytrain.p", "wb"))
        pickle.dump(testData, open("test.p", "wb"))
        pickle.dump(yTest, open("ytest.p", "wb"))

    yTrain = np.reshape(yTrain, (len(yTrain), 1))
    yTest = np.reshape(yTest, (len(yTest), 1))

    return (trainData, yTrain), (testData, yTest)


def plotData(gesture):
    """Plot the data to see distribuiton and behaviour """

    x = np.arange(0, 8, 1)
    pylab.figure()
    if gesture == 1:
        data1 = pickle.load(open("data1.p", "rb"))
        for i in range(len(data1)):
            print(data1[i])
            pylab.plot(x, data1[i], 'bo')

    elif gesture == 2:
        data2 = pickle.load(open("data2.p", "rb"))
        for i in range(len(data2)):
            pylab.plot(x, data2[i], 'bo')

    elif gesture == 3:
        data3 = pickle.load(open("data3.p", "rb"))
        for i in range(len(data3)):
            pylab.plot(x, data3[i], 'bo')

    elif gesture == 4:
        data4 = pickle.load(open("data4.p", "rb"))
        for i in range(len(data4)):
            pylab.plot(x, data4[i], 'bo')

    elif gesture == 5:
        data5 = pickle.load(open("data5.p", "rb"))
        for i in range(len(data5)):
            pylab.plot(x, data5[i], 'bo')

    elif gesture == 6:
        data6 = pickle.load(open("data6.p", "rb"))
        for i in range(len(data6)):
            pylab.plot(x, data6[i], 'bo')

    pylab.show()


def loadTest():
    global data1
    count = 0
    for filename in os.listdir(dataDir):
        gesture = int(filename[7:8])
        with open(dataDir + filename, 'r') as data:

            reader = csv.reader(data, delimiter=',')

            if gesture == 1:
                count += 1
                for row in reader:
                    data1 = np.r_[data1, [row]]
                    print(data1.astype('float32'))
                    # print(data1[0])
        if count:
            break

    data1 = data1.astype('float32')
    data1 /= np.max(data1)
    print(data1)

    trainData = np.zeros((11400, 1, 50, 8), dtype="uint16")
    trainData = pickle.load(open("train.p", "rb"))
    testData = np.zeros((600, 1, 50, 8), dtype="uint16")
    testData = pickle.load(open("test.p", "rb"))

    print(testData[300])
    data = pickle.load(open("data4.p", "rb"))
    print(data[1895*50:1896*50])

    print(testData[399])
    print(data[1994*50:1995*50])

#     for i,j in zip(range(0,100),range(1901,2001)):
#         testData[i] = data[(j-1) * 50: j*50]

#     # print(testData[0][0])
# loadData(usePickle=False, loadFromFile=True)
# loadTest()
# # plotPDatat()
# plotData(1)
