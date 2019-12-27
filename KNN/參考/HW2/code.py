from sklearn import datasets
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt

def KNN(Test_input, X_train, y_train, K):
    #Calculate dist. due to Test_input
    dist_train = np.zeros(X_train.shape[0], dtype=float)
    for i in range(0, X_train.shape[0]):
        dist_train[i] = np.linalg.norm(Test_input - X_train[i])
        
    Train_set_with_dist = np.concatenate((X_train, y_train[:, None], dist_train[:, None]), axis = 1)
    Sorting_Train_set = Train_set_with_dist[Train_set_with_dist[:,5].argsort()]
    
    Train_set_with_sort = Sorting_Train_set[:, 0 : -1]
    #Take out Top K entry as neighbor
    Neighbor_set = Train_set_with_sort[0 : K]
    A = Neighbor_set[:,-1].tolist()
    c = Counter(Neighbor_set[:,-1].tolist())
    value, count = c.most_common()[0]
    return value

#Load data
iris_dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size = 0.5, random_state = 0)

predicted_y_train = np.zeros((X_train.shape[0], 1), dtype = float)
predicted_y_test  = np.zeros((X_test.shape[0], 1), dtype = float)

train_error_array = np.zeros((20,1), dtype = float)
test_error_array = np.zeros((20, 1), dtype = float)
for i in range(0 , 20):
    train_error_point = 0.
    test_error_point = 0.
    
    for j in range(0, X_train.shape[0]):
        predicted_y_train[j] = KNN(X_train[j], X_train, y_train, i + 1)
        predicted_y_test[j] =  KNN(X_test[j], X_train, y_train, i + 1)
        
    
    train_error = 1 - metrics.accuracy_score(y_train, predicted_y_train)
    test_error = 1 - metrics.accuracy_score(y_test, predicted_y_test)
    print("K : {0:d} Train_Error: {1:.2f} Test_Error: {2:.2f}".format(i + 1, train_error, test_error))
    train_error_array[i] = train_error
    test_error_array[i] = test_error

y =  np.concatenate((train_error_array, test_error_array), axis = 1)
# Draw the outcome
plots = plt.plot(range(1, 21), y)
plt.legend(plots, ('Train_Error', 'Test_Error'), loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})
plt.show()