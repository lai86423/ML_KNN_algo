#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np 
import random
import operator
import math


# In[2]:


def loadDataSet(filename,split,trainingSet=[],testSet=[]):
    with open(filename, newline='') as csvfile:
        lines=csv.reader(csvfile)
        dataset = list(lines)
        print(dataset)    
        dataset=np.random.permutation(dataset)#打亂資料
        #print("len(dataset)=",len(dataset))
        for x in range(len(dataset)): #目前還不知為何不用減1 跑才對
            #print("x=",x)
            for y in range(4):
                #print(x,y,dataset[x][y])
                dataset[x][y]=float(dataset[x][y])
            if x/(len(dataset)) < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
    print("train", trainingSet)
    print("test", testSet)
    
#     return trainingSet, testSet
#     print("Train:",repr(len(trainingSet)))
#     print("Test:",repr(len(testSet)))


# In[3]:


def EuclideanDis(d1,d2,length):
    dis=0
    #print("d1,d2=",d1,d2)
    for x in range(length):
        #print(float(d1[x])-float(d2[x]))
        dis+=pow(float(d1[x])-float(d2[x]),2)
    #print('EuclideanDis=',math.sqrt(dis))
    return math.sqrt(dis)


# In[4]:


def getNeighbors(trainingSet, testInstance, k):
    disSet=[]
    length = len(testInstance)-1
    #print("testInstance length=",length)
    for x in range(len(trainingSet)):
        dist= EuclideanDis(testInstance,trainingSet[x],length)
        disSet.append((trainingSet[x],dist))
    disSet.sort(key=operator.itemgetter(1)) #用第二個域'距離'來由小到大排
    #print(disSet)
    neighbors = []
    for x in range(k):
        #print("nei X=",x)
        neighbors.append(disSet[x][0])
    return neighbors


# In[5]:


def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] +=1
        else:
            classVotes[response] =1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1),reverse=True)
    #print("sortedVotes",sortedVotes)
    #print(sortedVotes)
    return sortedVotes[0][0]


# In[6]:


def getAccuracy(testSet,predittions):
    correct =0
    for x in range (len(testSet)):
        if testSet[x][-1] == predittions[x]:
            correct +=1
    print(("Error="), 75-correct)    
    #print((correct/float(len(testSet)))*100.0)
    return (1-correct/float(len(testSet)))*100.0


# In[7]:


# 畫圖 
import matplotlib.pyplot as plt
def Draw(k,accuracy):      
    plt.figure(figsize=(12, 8))
    #for i in range (k):
     #   print(i,accuracy[i])
    plt.xticks(np.arange(0, k+1, 1.0))
    plt.plot(range(1,21),accuracy, alpha=0.6) 
    plt.show()


# In[8]:


from sklearn import datasets
from sklearn.model_selection import train_test_split 

def data():
    iris_dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size = 0.5, random_state = 0)
    trainSet = []
    testSet = []
    
    for index, item in enumerate(X_train):
        item = np.append(item, y_train[index])
        trainSet.append(item)
    for index, item in enumerate(X_test):
        item = np.append(item, y_test[index])
        testSet.append(item)
    return trainSet, testSet
    
def test1():
#     dataset = 'iris.data'
    split = 0.5
    trainingSet = []
    testSet = []
#     trainingSet, testSet = loadDataSet(dataset,split,trainingSet,testSet)
#     print(trainingSet)
    dataset='iris.data'
    accuracy=[]
    accuracyTrain=[]
    k=20
    for i in range(k):                
        x=0
        trainingSet=[]
        testSet=[]
        split=0.5 
#         loadDataSet(dataset,split,trainingSet,testSet)
        trainingSet, testSet = data()
        predittions=[]
        for x in range(len(testSet)):
                neighbors= getNeighbors(trainingSet,testSet[x],i+1)
                result = getResponse(neighbors)
                predittions.append(result)
#                 print(testSet[x])
                #print("predict=",result,"real=",testSet[x][-1])
#         print(i,accuracy)
        accuracy.append(getAccuracy(testSet,predittions))
    for i in range(k):                
        x=0
        trainingSet=[]
        testSet=[]
        split=0.5 
#         loadDataSet(dataset,split,trainingSet,testSet)
        trainingSet, testSet = data()
        predittions=[]
        for x in range(len(trainingSet)):
                neighbors= getNeighbors(trainingSet,trainingSet[x],i+1)
                result = getResponse(neighbors)
                predittions.append(result)
#                 print(testSet[x])
                #print("predict=",result,"real=",testSet[x][-1])
#         print(i,accuracy)
        accuracyTrain.append(getAccuracy(trainingSet,predittions))
        print("K=",(i+1))    
    Draw(k,accuracy)
    Draw(k,accuracyTrain)
test1()
#data()


# ##### 
