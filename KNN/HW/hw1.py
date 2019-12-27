#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tkinter import *
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.filedialog import askopenfilename
#from hw1 import HW1
#結果要有 訓練辨識率、測試辨識率、鍵結值
X=[]
PointData=[]
train_X=[]
test_X=[]
train_Y=[]
test_Y=[]
train_d=[]#期望輸出
test_d=[]#期望輸出
train_m=0
test_m=0
train_Accuracy=0.0
test_Accuracy=0.0
learning_rate=0.8 #預設學習率
N=int(input("輸入收斂條件")) #輸入收斂條件
learning_rate=float(input("輸入學習率")) #輸入學習率
w_record=[] #記憶鍵結值


# In[2]:


datafile_list= ['perceptron1.txt','perceptron2.txt','2Ccircle1.txt','2Circle1.txt',
                        '2Circle2.txt','2CloseS.txt','2CloseS2.txt','2CloseS3.txt','2cring.txt',
                        '2CS.txt','2Hcircle1.txt','2ring.txt','Number.txt']
#def get_data(self):
#        return self.datafile_list
filename = 'dataSet\\2CS.txt'
with open(filename,'r') as f :
    #讀資料 
    for line in f :
        X.append(list(map(float, line.split(' '))))       


# In[3]:


def get_data(self):
    filename = askopenfilename()
    with open(filename,'r') as f :
        #讀資料 
        for line in f :
            X.append(list(map(float, line.split(' '))))   


# In[4]:


#打亂資料
X=np.random.permutation(X) #print("打亂後資料\n",X) #<class 'numpy.ndarray'>
X=np.array(X) 
#計算輸入檔案之數量 維度 row,col 
m,n=np.shape(X) 
n=n-1#扣掉最後一筆是期望輸出
print("所有資料數和維度",m,n)

# 檢查是否二類問題
if(n>2):
    print("非二類問題")
    tk.messagebox.showinfo("非二類問題","非二類問題")
    #return 

#訓練資料和期望輸出的切割
temp_X=np.array_split(X,n,axis=1)#將最後一筆期望輸出切出
temp_d=temp_X[1]
temp_X=temp_X[0]
PointData=temp_X #存資料座標點

x0=-(np.ones(m))#X運算時需減掉閥值 用X0=-1來運算
#將x0加在資料最後一筆
temp_X=np.column_stack((temp_X,x0))#記得 加在最後一筆 跟課本是加在第0筆
#print(temp_X)

#切割訓練與測試資料
train_m=round((m/3)*2) #訓練資料數2/3
test_m=m-train_m #測試資料1/3 #print(train_m,test_m)
train_X=temp_X[:train_m]
test_X=temp_X[train_m:]
#print("訓練資料=",train_X,"測試資料",test_X)

#切割訓練與測試預期輸出
train_d=temp_d[:train_m]
test_d=temp_d[train_m:]
train_temp = []
test_temp = []
for i in train_d:
    for j in i:
        train_temp.append(j)

for x in test_d:
    for u in x:
        test_temp.append(u)

train_d=np.array(train_temp)
test_d=np.array(test_temp)
print("訓練預期輸出=",train_d,"測試預期輸出=",test_d)
train_Y=np.zeros(int(train_m)) #實際輸出 預設0 #print(train_Y)
test_Y=np.zeros(int(test_m))
#print("train_Y=",train_Y,"test_Y=",test_Y)         

# label非0/1組合 改變label-> 0~1
if (0 not in train_d) or (1 not in train_d):
    for i in range(int(train_m)):
        train_d[i]=train_d[i]%2
if (0 not in test_d) or (1 not in test_d):
    for i in range(int(test_m)):
        test_d[i]=test_d[i]%2     
print("修改0/1後訓練預期輸出=",train_d,"測試預期輸出=",test_d)


# In[5]:


def sgn(y):
    if y > 0:
        return 1
    else:
        return 0    


# In[6]:


#訓練資料
def Percetron_Learning(x,y,m,d,P_w):
        #w_record.append(w.copy())
        WchangeNum=0
        AllCorrect=False
        print("閥值,收斂條件,學習率=",P_w,N,learning_rate)          
        #print("訓練資料=",x)
        #print("實際輸出與預期輸出=",y,d)
        for n in range(N):   
            if(AllCorrect==False):
                for i in range(int(m)):
                    #a=np.random.randint(m)
                    print("第%d回的第%d次訓練，值為"%(n+1,i+1),x[i,:])            
                    print("w與x取內積值=",P_w.dot(x[i,:]))
                    y[i]=sgn(P_w.dot(x[i,:])) # y=sign((w．X))
                    print("經活化函數後w．x 的值",y[i])
                    print("y[i]=",y[i],"d[i]=",d[i])#測
                    print("W=",P_w)
                    if(y[i]!=d[i]):
                        if(y[i]<d[i]):
                            P_w=P_w+learning_rate*x[i,:] #+或-學習率判斷 ，由乘上期望輸出的正負號即可知
                        else:
                            P_w=P_w-learning_rate*x[i,:] #+或-學習率判斷 ，由乘上期望輸出的正負號即可知                    
                        w_record.append(P_w.copy())
                        WchangeNum+=1
                        print("W第"+str(WchangeNum)+"次修正=",P_w)
                        continue                      
                    if np.all(y==d):
                        print("提前修正!")
                        AllCorrect=True
                        break
        print("w最終為",P_w)
        #print(y,d,type(y),type(d))
        Adapted_Y=y
        print("Adapt y=",Adapted_Y)
        return P_w,Adapted_Y,WchangeNum


# In[7]:


w=[]
TrainNum=0
w=np.array([0,1,-1])#w初始值(0,1)  閥值視為最後一筆 (課本的w0)
FinalW,Adapted_train_Y,TrainNum=Percetron_Learning(train_X,train_Y,train_m,train_d,w)


# In[8]:


def Accuracy(A_x,A_y,A_d,m,final_w):    
    Error=0
    for i in range(int(m)):
        print("第%d筆資料="%(i+1),A_x[i,:])            
        print("w與x取內積值=",final_w.dot(A_x[i,:]))
        A_y[i]=sgn(final_w.dot(A_x[i,:])) # y=sign((w．X))
        print("經活化函數後w．x 的值",A_y[i])
        print("y[i]=",A_y[i],"d[i]=",A_d[i])#測
        if(A_y[i]!=A_d[i]):
            Error+=1
    A=((m-Error/m)*100)  
    return A,A_y[i]     


# In[9]:


#訓練辨識率
AccuRate,Last_train_Y=Accuracy(train_X,Adapted_train_Y,train_d,train_m,FinalW)
train_Accuracy=AccuRate/train_m
print("train_Accuracy=",train_Accuracy)


# In[10]:


TestAccuRate,Last_test_Y=Accuracy(test_X,test_Y,test_d,test_m,FinalW)
test_Accuracy=TestAccuRate/test_m
print("test_Accuracy=",test_Accuracy)


# In[13]:


# 畫圖 
import matplotlib.pyplot as plt
print("修正後最終鍵結值 W1=",FinalW[0],"W2=",FinalW[1])
print("訓練次數=",TrainNum,"學習率=",learning_rate)
print("訓練正確率=",train_Accuracy,"測試正確率=",test_Accuracy)
      
plt.scatter(train_X[:,0],train_X[:,1],c = 'c',marker = '+')
plt.scatter(test_X[:,0],test_X[:,1])
x1 = np.arange(X.min()-0.1,X.max()+0.1,0.01)
x2 = -(FinalW[0]*x1-FinalW[2])/FinalW[1]
line, = plt.plot(x1,x2, '-r', label='graph')
plt.ylim(train_X[:,1].min()-0.1,train_X[:,1].max()+0.1,0.01)
plt.grid()
plt.show()


# In[12]:


#def Draw(X,k,TestAccuracy,TrainAccuracy):   
#print("PointData=",train_X)
#print("PointDataX=",train_X[:,0])
#print("PointDataY=",train_X[:,1])

#for i in range (int(train_m)):
#    if Adapted_train_Y[i]==1:
#        Group1[i]=(train_X[i])
#    else:
#        Group2.append(train_X[i])
#print(Adapted_train_Y)

#print("Group1=",Group1)        
#print("Group2=",Group2) 
f.close()

