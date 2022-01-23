# Noah Conway
# CS 422 Project 1
# Problem 2: K_nn
import numpy as np

def calc_Accuracy(test, predicted):
    TN = 0
    TP = 0
    FN = 0
    FP = 0

 #compare actual values and predicted values
    for i in range(len(test)):
            if (test[i] == -1 and predicted[i] == -1):
                TN = TN+1
            if (test[i] == -1 and predicted[i] == 1):
                TP = TP+1
            if (test[i] == 1 and predicted[i] == -1):
                FN = FN + 1
            if (test[i] == -1 and predicted[i] == 1):
                FP = FP + 1

    denom = (TP + TN + FP + FN)

    if denom == 0:
        print("error: denom = 0")
        return 0

    acc = (TP + TN)/denom
    return acc

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    Y_pred = []
    for i in range(np.shape(X_test)[0]):
        distances_arr = [np.linalg.norm(np.subtract(X_test[i], p)) for p in X_train]
        neighbors_arr = np.argsort(distances_arr)[:K]
        label = 1
        for j in neighbors_arr:
            label = label*Y_train[j]
            #because all labels are either -1 or 1, an odd number of -1 (meaning majority -1) labels would give a negative result, where even would give positive

        Y_pred = np.append(Y_pred, label)

    return calc_Accuracy(Y_test, Y_pred)


           

        
    

    
    
