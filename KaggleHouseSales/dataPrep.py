
# coding: utf-8

# In[99]:

# get_ipython().magic(u'matplotlib nbagg')
import pickle
import pandas as pd
import pickle
import numpy as np
import re
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer,StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
norm=Normalizer()

#performing a basic analysis on the data from kaggle house sales
train=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\train.csv')
test=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\test.csv')
train=train.drop('Id',1)
test=test.drop('Id',1)

numeric_features_train=train.dtypes[train.dtypes!='object'].index
object_features_train=train.dtypes[train.dtypes=='object'].index
file1=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\featureIndices','wb')

# print object_features_index
# print numeric_features_index
numeric_features_test=test.dtypes[test.dtypes!='object'].index
object_features_test=test.dtypes[test.dtypes=='object'].index
pickle.dump([numeric_features_test,object_features_test],file1)
file1.close()



# In[106]:

for i in numeric_features_train:
    train[i].fillna(train[i].mean())


for i in numeric_features_test:
    test[i].fillna(test[i].mean())

print train.info()


# In[107]:

le =LabelEncoder()

for i in train.dtypes.index:
    
    if i not in (numeric_features_train):
        train[i]=le.fit_transform(train[i])
for i in test.dtypes.index:
    
    if i not in (numeric_features_test):
        test[i]=le.fit_transform(test[i])

# In[108]:

# print train.head(5)


# In[109]:

train.info()


# In[110]:

for i in train.dtypes.index:
    train[i].drop(pd.isnull(train[i]))
list_y=train['SalePrice'].values.tolist()    

numeric_features_train.tolist().remove(numeric_features_train[0])
numeric_features_test.tolist().remove(numeric_features_test[0])


for i in train.dtypes[train.dtypes=='float64'].index:
    train[i]=train[i].fillna(np.mean(train[i]))
    # print len(train[i])
for i in test.dtypes[test.dtypes=='float64'].index:
    test[i]=test[i].fillna(np.mean(test[i]))


cols_to_plot=object_features_train.tolist()
# print cols_to_plot
hist_Contour= [i for i in train['LandContour'].values.tolist() ]
# print train.columns
corrcoeff =[]
# numeric_features_index
for i in numeric_features_train:
    corrcoeff.append(np.corrcoef(train[i],train['SalePrice']))

from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
regr =XGBRegressor(n_estimators=100)

numeric_features_train=train.dtypes.index
numeric_features_test=test.dtypes.index

numeric_features_train=(numeric_features_train).tolist()

numeric_features_train.remove('SalePrice')

xTrain=train[numeric_features_train]
yTrain=train['SalePrice']
xTest=test[numeric_features_test]
file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\DumpFile','wb')
pickle.dump([xTrain,yTrain,xTest],file)
file.close()
xtrain,xvalid,ytrain,yvalid=train_test_split(train[numeric_features_train],train['SalePrice'])


# In[140]:

# Performing a simple linear regression with continuous vars
import xgboost
regr_xgb=xgboost.XGBRegressor(n_estimators=100)

regr_xgb.fit(xtrain,ytrain)
ypred=[]
ypred=regr_xgb.predict(xvalid)
rmse=np.sqrt(mean_squared_error(np.log(ypred+1),np.log(yvalid+1)))
# xgboost.plot_importance(regr_xgb)
print rmse


# In[121]:

x=[]
for i in range(len(ypred)):
    x.append(i)
      
plt.plot(x,yvalid)
plt.plot(x,ypred)
# plt.colorbar()
plt.show()


# In[ ]:



