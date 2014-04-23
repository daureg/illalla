#! /usr/bin/env python
"""
Python wrapper to execute c++ tSNE implementation
for more information on tSNE, go to :
http://ticc.uvt.nl/~lvdrmaaten/Laurens_van_der_Maaten/t-SNE.html

HOW TO USE
Just call the method calc_tsne(dataMatrix)

Created by Philippe Hamel
hamelphi@iro.umontreal.ca
October 24th 2008
"""

import struct
import sys
import os
import numpy as np
xprint = lambda x: None


class tSNE(object):
    """Mock Scikit manifold interface (at least the part needed)"""
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, data):
        return calc_tsne(data, self.n_components)


def calc_tsne(dataMatrix, NO_DIMS=2, PERPLEX=30, INITIAL_DIMS=30, LANDMARKS=1):
    """
    This is the main function.
    dataMatrix is a 2D numpy array containing data (each row is a data point)
    Remark : LANDMARKS is a ratio (0<LANDMARKS<=1)
    If LANDMARKS=1, returns the list of points in the same order as the input
    """

    dataMatrix = PCA(dataMatrix, INITIAL_DIMS)
    writeDat(dataMatrix, NO_DIMS, PERPLEX, LANDMARKS)
    call_tSNE()
    Xmat, LM, _ = readResult()
    clearData()
    if LANDMARKS == 1:
        return reOrder(Xmat, LM)
    return Xmat, LM


def PCA(dataMatrix, INITIAL_DIMS):
    """
    Performs PCA on data.
    Reduces the dimensionality to INITIAL_DIMS
    """
    xprint('Performing PCA')

    if dataMatrix.shape[1] < INITIAL_DIMS:
        return dataMatrix

    dataMatrix = dataMatrix - dataMatrix.mean(axis=0)
    eigValues, eigVectors = np.linalg.eig(np.cov(dataMatrix.T))
    perm = np.argsort(-eigValues)
    eigVectors = eigVectors[:, perm[0:INITIAL_DIMS]]
    dataMatrix = np.dot(dataMatrix, eigVectors)
    return dataMatrix


def readbin(type, file):
    """
    used to read binary data from a file
    """
    return struct.unpack(type, file.read(struct.calcsize(type)))


def writeDat(dataMatrix, NO_DIMS, PERPLEX, LANDMARKS):
    """
    Generates data.dat
    """
    xprint('Writing data.dat')
    info = 'Projection: %i D \nPerplexity: %i \nLandmarks(ratio): %f'
    xprint(info % (NO_DIMS, PERPLEX, LANDMARKS))
    n, d = dataMatrix.shape
    f = open('data.dat', 'wb')
    f.write(struct.pack('=iiid', n, d, NO_DIMS, PERPLEX))
    f.write(struct.pack('=d', LANDMARKS))
    for inst in dataMatrix:
        for el in inst:
            f.write(struct.pack('=d', el))
    f.close()


def call_tSNE():
    """
    Calls the tsne c++ implementation depending on the platform
    """
    platform = sys.platform
    xprint('Platform detected : %s' % platform)
    if platform in ['mac', 'darwin']:
        cmd = './tSNE_maci'
    elif platform == 'win32':
        cmd = './tSNE_win'
    elif platform == 'linux2':
        cmd = './tSNE_linux'
    else:
        xprint('Not sure about the platform, we will try linux version...')
        cmd = './tSNE_linux'
    xprint('Calling executable "%s"' % cmd)
    os.system(cmd)


def readResult():
    """
    Reads result from result.dat
    """
    xprint('Reading result.dat')
    try:
        f = open('result.dat', 'rb')
    except IOError:
        url = 'http://homepage.tudelft.nl/19j49/t-SNE.html'
        print('Download binary from '+url)
        raise
    n, ND = readbin('ii', f)
    Xmat = np.empty((n, ND))
    for i in range(n):
        for j in range(ND):
            Xmat[i, j] = readbin('d', f)[0]
    LM = readbin('%ii' % n, f)
    costs = readbin('%id' % n, f)
    f.close()
    return (Xmat, LM, costs)


def reOrder(Xmat, LM):
    """
    Re-order the data in the original order
    Call only if LANDMARKS==1
    """
    xprint('Reordering results')
    X = np.zeros(Xmat.shape)
    for i, lm in enumerate(LM):
        X[lm] = Xmat[i]
    return X


def clearData():
    """
    Clears files data.dat and result.dat
    """
    xprint('Clearing data.dat and result.dat')
    os.system('rm data.dat')
    os.system('rm result.dat')
