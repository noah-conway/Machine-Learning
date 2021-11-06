# Noah Conway
# CS 422 Project 1
# Problem 1: Decision Trees

import numpy as np

def entropy(f):
    #calculates the entropy for feature f
    num_occur = np.unique(f, return_counts=True)[1]
    prob = num_occur/np.size(f)
    Hs = -np.sum([p * np.log2(p) for p in prob])
    return Hs

def split(feature_col):
    #splits data along binary feature value
    left_ids = np.argwhere(feature_col == 0).flatten()
    right_ids = np.argwhere(feature_col == 1).flatten()

    return left_ids, right_ids

def infogain(feature, Y):
    #calculates the information gain of feature
    Hs_whole = entropy(Y)
    
    left_ids, right_ids = split(feature)

    if(np.size(Y) == 0):
        return 0

    Hs_left = entropy(Y[left_ids])
    Hs_right = entropy(Y[right_ids])
    p_left = np.size(left_ids)/np.size(Y)
    p_right = np.size(right_ids)/np.size(Y)

    ig = Hs_whole - (p_left*Hs_left + p_right*Hs_right)
    return ig

def get_most_label(Y):
    #returns the label that is most common out of a set of labels
    label, freq = np.unique(Y, return_counts = True) 
    most_common_idx = np.argsort(freq)[::-1][0]
    return label[most_common_idx]


class Node:
    def __init__(self, feature=None, left=None, right=None, label=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.label = label
    
    def is_leaf(self):
        return (self.left is None and self.right is None)

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, X, Y):
        self.root = self.build_tree_helper(X, Y)
    
    def build_tree_helper(self, X, Y, depth=0):
        is_unique=len(np.unique(Y))
        #when to exit, also when a node is a leaf
        if((depth >= self.max_depth) or (is_unique == 1)):
                leaf_label = get_most_label(Y)
                return Node(label=leaf_label)

        best_ig = -1
        feat_to_split = None
        feat_to_split_idx = None
        for feat_idx in range(np.shape(X)[1]):
            feature = X[:, feat_idx]
            curr_ig = infogain(feature, Y) 
            if curr_ig > best_ig:
                best_ig = curr_ig
                feat_to_split = feature
                feat_to_split_idx = feat_idx

        left_ids, right_ids = split(X[:, feat_to_split_idx])


        left =self.build_tree_helper(X[left_ids, :], Y[left_ids], depth+1)
        right = self.build_tree_helper(X[right_ids, :], Y[right_ids], depth+1)
        return Node(feat_to_split_idx, left, right)

    def traverse(self, sample, node):

        if node.is_leaf():
            return node.label


        if (np.array_equal(sample[node.feature], 0)):
            return self.traverse(sample, node.left)
        if (np.array_equal(sample[node.feature], 1)):
            return self.traverse(sample, node.right)




def DT_train_binary(X,Y,max_depth):
    if max_depth == -1:
        max_depth = np.shape(X)[1]

    dtt = DecisionTree(max_depth=max_depth)
    dtt.build_tree(X, Y)
    return dtt

def DT_make_prediction(x, DT):
    result = DT.traverse(x, DT.root)
    return result

def DT_test_binary(X, Y, DT):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_Y = np.zeros(len(Y))

    
    #for point p in fature data X, make prediction and add it to predicted Y values array 
    Y_idx = 0
    for p in X:
        num = DT_make_prediction(p, DT)
        pred_Y[Y_idx] = num
        Y_idx = Y_idx+1

    #compare actual values and predicted values
    for i in range(len(Y)):
            if (Y[i] == 0 and pred_Y[i] == 0):
                TN = TN+1
            if (Y[i] == 1 and pred_Y[i] == 1):
                TP = TP+1
            if (Y[i] == 1 and pred_Y[i] == 0):
                FN = FN + 1
            if (Y[i] == 0 and pred_Y[i] == 1):
                FP = FP + 1

    denom = (TP + TN + FP + FN)

    if denom == 0:
        print("error: denom = 0")
        return 0

    acc = (TP + TN)/denom
    return acc
    
    



