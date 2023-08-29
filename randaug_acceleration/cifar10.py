import numpy as np

#-----------------------------
# Read the CIFAR10 dataset
#-----------------------------
def readCifar10File(fname):
	fin    = open(fname,'rb')
	data   = bytearray(fin.read())
	npdata = np.frombuffer(data, dtype=np.uint8)
	npdata = npdata.reshape((10000, 3073))
	Y = npdata[:,0]
	X = npdata[:,1:]
	X = X.reshape(10000,3,32,32)
	return (X,Y)
def readCifar10():	
	trainX = np.zeros((50000,3,32,32), dtype=np.uint8)
	trainY = np.zeros((50000,), dtype=np.uint8)
	(trainX[    0:10000], trainY[    0:10000]) = readCifar10File('cifar-10-batches-bin/data_batch_1.bin')
	(trainX[10000:20000], trainY[10000:20000]) = readCifar10File('cifar-10-batches-bin/data_batch_2.bin')
	(trainX[20000:30000], trainY[20000:30000]) = readCifar10File('cifar-10-batches-bin/data_batch_3.bin')
	(trainX[30000:40000], trainY[30000:40000]) = readCifar10File('cifar-10-batches-bin/data_batch_4.bin')
	(trainX[40000:50000], trainY[40000:50000]) = readCifar10File('cifar-10-batches-bin/data_batch_5.bin')
	(testX, testY) = readCifar10File('cifar-10-batches-bin/test_batch.bin')
	return ((trainX,trainY),(testX,testY))
	
