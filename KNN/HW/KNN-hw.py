#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np 
import random


# In[2]:


def loadDataSet(filename,split,trainingSet=[],testSet=[]):
    with open(filename, newline='') as csvfile:
        lines=csv.reader(csvfile)
        dataset = list(lines)
        print(dataset)    
        print(len(dataset))
        dataset=np.random.permutation(dataset)#打亂資料
        for x in range(len(dataset)):
            for y in range(4):
                print(x,y,dataset[x][y])
                dataset[x][y]=float(dataset[x][y])
            if x/(len(dataset)) < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])  
                    
    print("Train:",trainingSet,repr(len(trainingSet)))
    print("Test:",testSet,repr(len(testSet)))


# In[3]:


dataset='iris1.data.txt'
trainingSet=[]
testSet=[]
split=0.5 
open(dataset,'r')
loadDataSet(dataset,split,trainingSet,testSet)


# In[4]:


import math
def EuclideanDis(d1,d2,length):
    dis=0
    for x in range(length):
        dis+=pow((d1[x]-d2[x]),2)
    return math.sqrt(dis)


# In[5]:


test1=[2,2,2,'a']
test2=[4,2,4,'b']
dis=EuclideanDis(test1,test2,2)
print("Dis=",repr(dis))


# In[6]:


import operator
def getNeighbors(trainingSet, testInstance, k):
    disSet=[]
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist= EuclideanDis(testInstance,trainingSet[x],length)
        disSet.append((trainingSet[x],dist))
        disSet.sort(key=operator.itemgetter(1)) #用第二個域'距離'來由小到大排
        neighbors = []
    for x in range(k):
        neighbors.append(disSet[x][0])
    #print(disSet)
    return neighbors


# In[7]:


#test
testTrainSet=[[2,2,2,'a'],[4,4,4,'b'],[1,1,1,'c'],[4.5,4,4,'b']]
test=[5,5,5]
testNeighbors=getNeighbors(testTrainSet,test,3)
print(testNeighbors)


# In[8]:


import operator
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] +=1
        else:
            classVotes[response] =1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]


# In[9]:


#test
print(getResponse(testNeighbors))


# In[10]:


def getAccuracy(testSet,predittions):
    correct =0
    for x in range (len(testSet)):
        if testSet[x][-1] is predittions[x]:
            correct +=1
    return (correct/float(len(testSet)))*100.0


# In[11]:


#test
predittions=['a','b','b','b']
accuracy=getAccuracy(testTrainSet,predittions)
print(accuracy)


# In[12]:


def main():
    dataset='iris.data'
    trainingSet=[]
    testSet=[]
    split=0.5 
    loadDataSet(dataset,split,trainingSet,testSet)
    #print(trainingSet)
    predittions=[]
    k=int(input("k="))
    for x in range(len(testSet)):
            neighbors= getNeighbors(trainingSet,testSet[x],k)
            result = getResponse(neighbors)
            predittions.append(result)
            print("predict=",result,"real=",testSet[x][-1])
    accuracy = getAccuracy(testSet,predittions)
    print("accuracy=",accuracy,"%")       


# In[13]:


main()


# In[ ]:




