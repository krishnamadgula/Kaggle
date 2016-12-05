
# coding: utf-8

# In[48]:

get_ipython().magic(u'matplotlib nbagg')
import pickle
import pandas as pd
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
print train.head(5)



# In[49]:

print train.describe()


# In[50]:

print train.columns


# In[51]:

# # # let us check the skewness of the data
# # from scipy.stats import skew
# # skewed_features=train[numeric_features_index].apply(lambda x: skew(x))
# cols_to_plot=object_features_index.tolist()
# print cols_to_plot
# # cols_to_plot.remove('SalePrice')

# # train.plot(x=cols_to_plot[7:8],kind='bar')


# In[52]:

print train.dtypes


# In[53]:

print train.info()


# In[54]:

numeric_features_index=train.dtypes[train.dtypes!='object'].index
object_features_index=train.dtypes[train.dtypes=='object'].index
print object_features_index
print numeric_features_index


# In[55]:

for i in numeric_features_index:
    train[i].fillna(train[i].mean())
print train.info()


# In[56]:

le =LabelEncoder()

for i in train.dtypes.index:
    
    if i not in (numeric_features_index):
        train[i]=le.fit_transform(train[i])


# In[57]:

print train.head(5)


# In[58]:

train.info()


# In[59]:

for i in train.dtypes.index:
    train[i].drop(pd.isnull(train[i]))
list_y=train['SalePrice'].values.tolist()    


# In[60]:

print numeric_features_index
numeric_features_index.tolist().remove(numeric_features_index[0])


    


# In[61]:

plt.hist(x=train['SaleCondition'])





# In[62]:

for i in train.dtypes[train.dtypes=='float64'].index:
    train[i]=train[i].fillna(np.mean(train[i]))
    print len(train[i])


# In[63]:

print train.info()


# In[64]:

# now we see the missing data has all been filled since the len of all columns match in our dataframe


# In[65]:

cols_to_plot=object_features_index.tolist()
print cols_to_plot
hist_Contour= [i for i in train['LandContour'].values.tolist() ]
print train.columns
corrcoeff =[]
numeric_features_index
for i in numeric_features_index:
    corrcoeff.append(np.corrcoef(train[i],train['SalePrice']))
print corrcoeff


# In[66]:

from sklearn.linear_model import LinearRegression
regr =LinearRegression(n_jobs=-1)
print numeric_features_index
numeric_features_index=numeric_features_index.tolist().remove(numeric_features_index[-1])
numeric_features_index=Index(numeric_features_index)
# xtrain,xvalid,ytrain,yvalid=train_test_split(train[numeric_features_index],)

