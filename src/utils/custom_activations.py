# standard libraries

# third party libraries
from keras import backend as K

# local libraries


def ReLU(x):
    return K.relu(x)


def leakyReLU(x):
    return K.relu(x, alpha=0.1)


def leakyReLU_03(x):
    return K.relu(x, alpha=0.3)


def Abs(x):
    return K.abs(x)
