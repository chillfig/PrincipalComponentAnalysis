# coding: utf-8




from numpy import *
from matplotlib import pyplot as plt
import sys

import numpy


def loadDataSet(fileName = 'iyer.csv'):
    dataMat=[]
    labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArray=line.strip().split(',')
        records = []
        for attr in lineArray[:-1]:
            records.append(float(attr))
        dataMat.append(records)
        labelMat.append(int(lineArray[-1]))
    dataMat = array(dataMat)
    
    labelMat = array(labelMat)
    
    
    return dataMat,labelMat

def pca(dataMat, PC_num=2):
    '''
    Input:
        dataMat: obtained from the loadDataSet function, each row represents an observation
                 and each column represents an attribute
        PC_num:  The number of desired dimensions after applyting PCA. In this project keep it to 2.
    Output:
        lowDDataMat: the 2-d data after PCA transformation
    '''
    xPrimeMat = []      # The matrix containing the difference of the mean of each column from the data x
    means = []          # array of means for each column
    lowDDataMat = []    # the 2-d data after PCA transformation

    # retrieve means of each column of input matrix
    for column in dataMat.T: 
        columnMean = average(column)   
        means.append(columnMean)

    # calculate x prime and contain values in a new matrix
    for row in dataMat:
        lcv = 0 # loop control variable for keeping record of which column mean to use
        rowRecords = []
        for observation in row:
            xPrime = observation - means[lcv]
            rowRecords.append(xPrime)
            lcv+=1
        xPrimeMat.append(rowRecords)
    xPrimeMat = numpy.array(xPrimeMat)

    # compute the covariance matrix S of adjusted xPrime
    #S = array((numpy.matmul(numpy.transpose(xPrimeMat),xPrimeMat)) / (xPrimeMat.shape[0] - 1))
    S = numpy.cov(xPrimeMat.T)
    # eigenvalue eigenvector decomposition
    [eigenvalues, eigenvectors] = numpy.linalg.eig(S)
    eigenvectors = eigenvectors.T
    
    # find max eigenvalue and find corresponding eigenvector of greatest eigenvalue
    mostCovarianceEigenvalues = []  # ascending list of eigenvalues for dimensions required 
    mostCovarianceEigenvectors = [] # eigenvalues in orded corresponding to highest eigenvalues

    for i in range(PC_num):
        index = numpy.where(eigenvalues == numpy.amax(eigenvalues))
        mostCovarianceEigenvectors.append(eigenvectors[index])
        mostCovarianceEigenvalues.append(eigenvalues[index])
        eigenvalues = numpy.delete(eigenvalues,index)               # delete the eigenvalue with greatest value from the search list
        eigenvectors = numpy.delete(eigenvectors,index, axis = 0)   # delete thd corresponding eigenvector from the search list
        
    # compute low dimension data matrix using eigen decomposition
    for row in xPrimeMat:
        y = []  # start with clear new y dimension list
        for eig in mostCovarianceEigenvectors: 
            sum = 0 
            for index in range(row.size):
                sum += (row[index] * (eig[0])[index]) # xPrime * coefficient of eigenvector eig
            y.append(sum)
        lowDDataMat.append(y)


    return array(lowDDataMat)


def plot(lowDDataMat, labelMat, figname):
    '''
    Input:
        lowDDataMat: the 2-d data after PCA transformation obtained from pca function
        labelMat: the corresponding label of each observation obtained from loadData
    '''
    y1 = []
    y2 = []
    colors = []
    lcv = 0

    # create discrete y1 and y2 vectors for the 2 new dimensions as well as a color vector. 
    for rows in lowDDataMat:
        y1.append(rows[0])
        y2.append(rows[1])
        colors.append(labelMat[lcv])
        lcv+=1
    
    plt.scatter(y1,y2,c=colors)
    plt.title(figname)
    plt.savefig(figname) #save file
    plt.show()
    


if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = 'iyer.csv'
    figname = filename
    figname = figname.replace('csv','jpg')
    dataMat, labelMat = loadDataSet(filename)

    lowDDataMat = pca(dataMat)
    plot(lowDDataMat, labelMat, figname)