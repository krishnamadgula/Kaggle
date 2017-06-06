
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



def featureEngineer(data_tr,data_ts,cont_feature,cat_feature):
	# cont_index=cont_index.tolist()
	# cont_index.remove('Id')
	# data=data[cont_index]
	
	data_tr['HouseAge']=2017-data_tr['YearBuilt']
	data_tr['RemodAfter']=data_tr['YearRemodAdd']-data_tr['YearBuilt']
	data_tr['OwnedPeriod']=data_tr['YrSold']-data_tr['YearBuilt']
	data_tr['GarageYrBlt']=2017-data_tr['GarageYrBlt']
	data_tr=data_tr.drop('YearBuilt',1)
	data_tr=data_tr.drop('YearRemodAdd',1)
	data_tr=data_tr.drop('YrSold',1)
	data_tr=data_tr.drop('Utilities',1)
	data_tr=data_tr.drop('Street',1)
	data_tr=data_tr.drop('Condition2',1)

	data_ts['HouseAge']=2017-data_ts['YearBuilt']
	data_ts['RemodAfter']=data_ts['YearRemodAdd']-data_ts['YearBuilt']
	data_ts['OwnedPeriod']=data_ts['YrSold']-data_ts['YearBuilt']
	data_ts['GarageYrBlt']=2017-data_ts['GarageYrBlt']
	data_ts=data_ts.drop('YearBuilt',1)
	data_ts=data_ts.drop('YearRemodAdd',1)
	data_ts=data_ts.drop('YrSold',1)
	data_ts=data_ts.drop('Utilities',1)
	data_ts=data_ts.drop('Street',1)
	data_ts=data_ts.drop('Condition2',1)
	print data_ts.info()

	# data=data.drop('MiscVal',1)
	# data=data.drop('PoolArea',1)
	# data=data.drop('ScreenPorch',1)
	# data=data.drop('3SsnPorch',1)
	# data=data.drop('EnclosedPorch',1)
	# data=data.drop('OpenPorchSF',1)
	# data=data.drop('BsmtHalfBath',1)
	# data=data.drop('LowQualFinSF',1)
	# data=data.drop('BsmtFinSF2',1)
	
	# cont_feature.remove('MiscVal')	
	# cont_feature.remove('PoolArea')	
	# cont_feature.remove('ScreenPorch')	
	# cont_feature.remove('3SsnPorch')	
	# cont_feature.remove('EnclosedPorch')	
	# cont_feature.remove('OpenPorchSF')	
	# cont_feature.remove('BsmtHalfBath')	
	# cont_feature.remove('LowQualFinSF')	
	# cont_feature.remove('BsmtFinSF2')		
	# print type(cont_feature)
	# cont_feature=(cont_feature).tolist()
	cont_feature.remove('YearBuilt')
	cont_feature.remove('YearRemodAdd')
	cont_feature.remove('YrSold')
	# data=data.drop('PoolQC',1)
	
	# cat_feature.remove('PoolQC')
	cat_feature.remove('Utilities')
	cat_feature.remove('Street')
	cat_feature.remove('Condition2')
	# cont_feature=temp
	return data_tr,data_ts,cont_feature,cat_feature


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
numeric_features_test=numeric_features_test.tolist()
object_features_test=object_features_test.tolist()



# In[106]:

# for i in numeric_features_train:
#     train[i].fillna(train[i].mean())


for i in numeric_features_test:
    test[i].fillna(test[i].mean())

# print train.info()
count=0
col_to_be_dropped=[]

for i in numeric_features_test:
	x=0
	# if i=='GarageArea':
	# print  train[i].values
	for j in train[i].values:
		# print j
		if pd.isnull(j):
			# print j
			x+=1

		# 	x+=1
	if x>=100 and x<=145:
		# print x,"--",i
		train[i]=train[i].fillna(0)
	elif x<100 and x>=0:
		# print x,"--",i
		train[i]=train[i].fillna(train[i].mean())	
		# print len(train[i])
	else:
		# print x,"--",i
		col_to_be_dropped.append(i)
			

for i in numeric_features_test:
	x=0
	# if i=='GarageArea':
	# print  train[i].values
	for j in test[i].values:
		# print j
		if pd.isnull(j):
			# print j
			x+=1

		# 	x+=1
	if x>=100 and x<=145:
		# print x,"--",i
		test[i]=test[i].fillna(0)
	elif x<100 and x>=0:
		# print x,"--",i
		test[i]=test[i].fillna(test[i].mean())	
		# print len(train[i])
	else:
		if i not in col_to_be_dropped:
			col_to_be_dropped.append(i)	
	# count=0
for i in object_features_test:
	x=0	
	for j in test[i].values:
		if pd.isnull(j):
			# print j
			x+=1

		# 	x+=1
	if x>=100 and x<=145:
		# print x,"--",i
		test[i]=test[i].fillna(0)
	elif x<100 and x>=0:
		# print x,"--",i
		test[i]=test[i].fillna(test[i].mode().iloc[0])
		# print train[i].values
		# print len(train[i])
	else:
		# print x,"--",i
		if i not in col_to_be_dropped:
			col_to_be_dropped.append(i)	




print  col_to_be_dropped
for i in col_to_be_dropped:
	train=train.drop(i,1)
	test=test.drop(i,1)
	if i in object_features_test:
		object_features_test.remove(i)
	elif i in numeric_features_test:
		numeric_features_test.remove(i)	
	# numeric_features_train.remove(i)
	# numeric_features_test.remove(i)




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
train=train.drop(train[train.MasVnrArea>1200].index,axis=0)
train=train.drop(train[train.BsmtFinSF1>3000].index,axis=0)
train=train.drop(train[train.TotalBsmtSF>3000].index,axis=0)
train=train.drop(train[train['1stFlrSF']>3000].index,axis=0)
train=train.drop(train[train.GrLivArea>4500].index,axis=0)
train=train.drop(train[train.MasVnrArea>1200].index,axis=0)
train=train.drop(train[train.BsmtHalfBath>1.5].index,axis=0)
train=train.drop(train[train.GarageArea>1200].index,axis=0)


# In[109]:




# In[110]:

# for i in train.dtypes.index:
#     train[i].drop(pd.isnull(train[i]))
list_y=train['SalePrice'].values.tolist()    

# numeric_features_train.tolist().remove(numeric_features_train[0])
# numeric_features_test.tolist().remove(numeric_features_test[0])


# for i in train.dtypes[train.dtypes=='float64'].index:
#     train[i]=train[i].fillna(np.mean(train[i]))
#     # print len(train[i])
# for i in test.dtypes[test.dtypes=='float64'].index:
#     test[i]=test[i].fillna(np.mean(test[i]))




numeric_features_train=train.dtypes.index
# numeric_features_test=test.dtypes.index

numeric_features_train=(numeric_features_train).tolist()

numeric_features_train.remove('SalePrice')

xTrain=train[numeric_features_train]
yTrain=train['SalePrice']
xTest=test[numeric_features_train]
	
print xTrain.info()

# print numeric_features_test
dummy=numeric_features_test
dummy2=object_features_test
xTrain,xTest,continuous_feature_index,categorical_feature_index=featureEngineer(xTrain,xTest,numeric_features_test,object_features_test)
# print dummy
# xTest,continuous_feature_index,categorical_feature_index=featureEngineer(xTest,dummy,dummy2)	
pickle.dump([continuous_feature_index,categorical_feature_index],file1)
file1.close()

	# print xTrain.info()

# print xTest.info()

file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\DumpFile','wb')
pickle.dump([xTrain,yTrain,xTest],file)
file.close()



