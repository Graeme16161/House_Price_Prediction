# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:07:21 2019

@author: gakel
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
import math

'''
Load data and seperate into training (X) and target features (y)
'''
#Load training data
X = pd.read_csv('train.csv', index_col = 'Id')
#set target feature to y
y = X['SalePrice']
# drop target feature from training data
X.drop('SalePrice',axis=1,inplace=True)


'''
apply log transform to target data
note the histogram of the transformed y_log is far more balanced than y
'''
plt.hist(y)

y_log = y.apply(math.log)
plt.hist(y_log)

'''
Seperate training features into numeric and catagorical data frames
Note MSSubClass should be categorical not numeric
'''
# list of all training features
X_all_feature_names = X.columns.tolist()

# data frame of just numeric training data
X_numeric_features = X._get_numeric_data().drop('MSSubClass',axis=1)

# list of only numeric training data features
X_numeric_feature_names = X_numeric_features.columns.tolist()

# list of only categoric training data features
X_categoric_feature_names = list(set(X_all_feature_names)-set(X_numeric_feature_names))

# categoric data
X_categoric_features = X[X_categoric_feature_names]


'''
Turn NANs in catagorical features and the column as a whole into
categorical
'''
X_categoric_features.fillna('Missing', inplace = True)


for name in X_categoric_feature_names:
    X_categoric_features[name] = X_categoric_features[name].astype('category')


'''
For numeric features, turn NAs into zeros, then create a dummy variable for
every zero in the numeric df
'''
X_numeric_features.fillna(0, inplace = True)

for name in X_numeric_feature_names:
    new_name = name + "_is_Zero"
    
    X_numeric_features[new_name] = X_numeric_features[name].apply(lambda x: "Zero" if x == 0 else "Non_Zero").astype('category')
    
    
    
'''
Combine numeric and categorical data frames back into one
'''
X_processed = X_numeric_features.join(X_categoric_features,how = 'inner')
    
    
X_dummified = pd.get_dummies(X_processed)



'''
baseline lasso model
'''

lassocv = LassoCV(cv=10)
baseline_lasso_model = lassocv.fit(X_dummified,y_log)

baseline_lasso_model.score(X_dummified,y_log)





#reg.alphas_
mse_path = np.mean(baseline_lasso_model.mse_path_,axis=1)
alphas = baseline_lasso_model.alphas_
fig = plt.figure(figsize=(5,8))
plt.plot(alphas, mse_path,color='blue',linewidth=1)
#plt.plot(df_y['y_estimated'].tolist(),color='black',linewidth=0.1)
plt.legend()
plt.show()



X_test = pd.read_csv('test.csv', index_col = 'Id')


def process_predictive_features(X):
    X_all_feature_names = X.columns.tolist()
    
    # data frame of just numeric training data
    X_numeric_features = X._get_numeric_data().drop('MSSubClass',axis=1)
    
    # list of only numeric training data features
    X_numeric_feature_names = X_numeric_features.columns.tolist()
    
    # list of only categoric training data features
    X_categoric_feature_names = list(set(X_all_feature_names)-set(X_numeric_feature_names))
    
    # categoric data
    X_categoric_features = X[X_categoric_feature_names]
    
    X_categoric_features.fillna('Missing', inplace = True)
    
    
    for name in X_categoric_feature_names:
        X_categoric_features[name] = X_categoric_features[name].astype('category')
        
    X_numeric_features.fillna(0, inplace = True)
    
    for name in X_numeric_feature_names:
        new_name = name + "_is_Zero"
        
        X_numeric_features[new_name] = X_numeric_features[name].apply(lambda x: "Zero" if x == 0 else "Non_Zero").astype('category')

    X_processed = X_numeric_features.join(X_categoric_features,how = 'inner')
        
        
    X_dummified = pd.get_dummies(X_processed)
    
    return(X_dummified)    




X_test_processed = process_predictive_features(X_test)

#find columns in both data sets:
test_cols = X_test_processed.columns.tolist()
train_cols = X_dummified.columns.tolist()

mutual_cols = list(set(test_cols) & set(train_cols))


drop_train = X_dummified[mutual_cols]
drop_test = X_test_processed[mutual_cols]


'''
baseline lasso model
'''

lassocv = LassoCV(cv=10)
baseline_lasso_model = lassocv.fit(drop_train,y_log)


f = baseline_lasso_model.predict(drop_test)

test_result = pd.DataFrame({'Id':drop_test.index, 'SalePrice' : f})

test_result['SalePrice'] = test_result['SalePrice'] .apply(math.exp)

test_result.to_csv('test_sub.csv',index=False)
























