#!/usr/bin/env python
# coding: utf-8

# # Take Home DISH

# You need to implement uplift-tree with DeltaDeltaP split criterion.
# 
# 
# To measure uplift we calculate the difference between average of the target variable in
# treatment and control groups (M). 
# The best split maximizes the difference between uplift
# measure on the left and right. difference(M_left, M_right)

# Hot to grow a tree:
# 
#     1. Create root with all values
#     2. Recursively run function Build:
# 
#     Build(node):
#         If node is at max depth, stop.
#     
#         For feature in all features:
#             For value in threshold values:
#                 make_split(feature, value) -> data_left, data_right
#                 If number of values in a split is lower than the parameter value
#                     Continue (do not consider this split)
#                 Calculate delte_delta_p
# 
#         Select the best split
#         Create node_left, node_right
#         Build(node_left)
#         Build(node_right)

# ## Import tools

# In[1]:


import numpy as np
import pandas as pd


# ## Node class

# In[2]:


class Node():
    def __init__(
        self, 
        feature_index = None, 
        left = None, 
        right = None, 
        info_gain = None, 
        value = None):
        ''' constructor '''
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value


# ## Sample data load

# In[3]:


preds = np.load('example_preds.npy') # prediction example
treatment = np.load('example_treatment.npy') # treatment flag
X = np.load('example_X.npy') # feature X
y = np.load('example_y.npy') # target variable

print('preds: ', preds)
print('treatment: ', treatment)
print('X: ', X)
print('y: ', y)


# In[4]:


type(X)


# ## Tree class

# In[ ]:


class UpliftTreeRegressor():
    def __init__(
        self,
        Max_depth: int =3, # max tree depth
        Min_samples_leaf: int = 1000, # min number of values in leaf
        Min_samples_leaf_treated: int = 300, # min number of treatment values in leaf
        Min_samples_leaf_control: int = 300, # min number of control values in leaf
        depth: int = 0,
        X : numpy.ndarray,
        y : numpy.ndarray,
        best_feature: int = 0,
        best_value: int = 0
    ):    
        ''' constructor '''

        # initialize the root of the tree
        self.root = None
        self.depth = 0
        
        self.X = X
        self.y = y
        
        # stopping conditions
        self.Max_depth = Max_depth
        self.Min_samples_leaf = Min_samples_leaf
        self.Min_samples_leaf_treated = Min_samples_leaf_treated
        self.Min_samples_leaf_control = Min_samples_leaf_control

    def build(self, node):
        ''' build apply DeltaDeltaP and iterate over relevant parameters '''
        
        # split until stopping condition are met
        #if node is at max depth, stop:
        if self.depth == self.Max_depth:
            return 

        for feature in self.X:
            # threshold algorithm
            unique_values = np.unique(feature)

            if len(unique_values) >10:
                percentiles = np.percentile(feature, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
            else:
                percentiles = np.percentile(unique_values, [10, 50, 90])
                threshold_options = np.unique(percentiles)
                
            for value in threshold_options:
                # make split
                left, right = best_split(feature, value)
                if left == None or right == None:
                    return
                deltadeltaP = left - right
                
                if best_value < deltadeltaP:
                    best_value = deltadeltaP

        build(left)
        build(right)
        
    def fit(
        self,
        X: np.ndarray, # (n * k) array with features
        Treatment: np.ndarray, # (n) array with treatment flag
        Y: np.ndarray # (n) array with the target
        ) -> None:
        # fit the model
        
        parameters = some_parameters
        tree = Tree(*some_init_params)
        test_performance = {}
        for kfold_cv:
            training_data, eval_data = cross_validation(Treatment)
            for relevant loop:
                build_tree(tree, params, training_data)
            test_performance[iteration] = evaluate(tree, eval_data) 
        

    def predict(self, X: np.ndarray) -> Iterable(float):
        # compute predictions
        
        cur_node = self
        while cur_node.depth < cur_node.Max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if cur_node.n < cur_node.min_samples_split:
                break 

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
            
        return predictions


# In[ ]:




