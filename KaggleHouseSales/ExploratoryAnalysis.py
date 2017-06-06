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
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold,train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
import xgboost
import random
file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\featureIndices','rb')
continuous_feature_index,categorical_feature_index=pickle.load(file)
file.close()
file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\DumpFile','rb')
xTrain,yTrain,xTest=pickle.load(file)
xTrain=[xTrain,yTrain]
xTrain=pd.concat(xTrain,axis=1)
file.close()
cont_feat=continuous_feature_index
# cont_feat.remove('Id')
# xTrain=xTrain[cont_feat]
# xTest=xTest[cont_feat]
# Continuous
# Index([u'Id', u'MSSubClass', u'LotFrontage', u'LotArea', u'OverallQual',
#        u'OverallCond', .u'YearBuilt', u'YearRemodAdd', u'MasVnrArea',
#        u'BsmtFinSF1', .u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF', u'1stFlrSF',
#        u'2ndFlrSF', .u'LowQualFinSF', u'GrLivArea', u'BsmtFullBath',
#        .u'BsmtHalfBath', u'FullBath', u'HalfBath', u'BedroomAbvGr',
#        u'KitchenAbvGr', u'TotRmsAbvGrd', u'Fireplaces', u'GarageYrBlt',
#        u'GarageCars', u'GarageArea', u'WoodDeckSF', .u'OpenPorchSF',
#        .u'EnclosedPorch', .u'3SsnPorch', .u'ScreenPorch', .u'PoolArea', .u'MiscVal',
#        u'MoSold', u'YrSold', u'SalePrice']
# 
# Categorical
# ['MSZoning', .'Street', .'Alley', 'LotShape'
# , 'LandContour', .'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood'
# , 'Condition1', .'Condition2', 'BldgType', 'HouseStyle',
# 'RoofStyle', .'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
# 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
# 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
# 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', '
# GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
# 'PavedDrive', .'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
# print xTrain[continuous_feature_index].info()
# plt.scatter(-xTrain['YearRemodAdd']+xTrain['YrSold'],yTrain)
# plt.show()
print xTrain.info()
norm=Normalizer()
# xTrain[cont_feat]=norm.fit_transform(xTrain[cont_feat])

# f,ax=plt.subplots()
plt.figure(1)
# df=[xTrain[continuous_feature_index]]
# df.append(yTrain)
# print df
# df=pd.dataframe()
# print df.info()
# res=[xTrain,yTrain]
# correlation=pd.concat(res,axis=1)
# corr = correlation.corr()
# fig, ax = plt.subplots(figsize=(20,20))
# ax.matshow(corr)
# plt.xticks(range(len(corr.columns)), corr.columns);
# plt.yticks(range(len(corr.columns)), corr.columns);
# # plt.matshow(xTrain[cont_feat].corr())
# plt.show()
xTrain=xTrain.drop(xTrain[xTrain.MasVnrArea>1200].index,axis=0)
xTrain=xTrain.drop(xTrain[xTrain.BsmtFinSF1>3000].index,axis=0)
xTrain=xTrain.drop(xTrain[xTrain.TotalBsmtSF>3000].index,axis=0)
xTrain=xTrain.drop(xTrain[xTrain['1stFlrSF']>3000].index,axis=0)
xTrain=xTrain.drop(xTrain[xTrain.GrLivArea>4500].index,axis=0)
xTrain=xTrain.drop(xTrain[xTrain.MasVnrArea>1200].index,axis=0)
xTrain=xTrain.drop(xTrain[xTrain.BsmtHalfBath>1.5].index,axis=0)
xTrain=xTrain.drop(xTrain[xTrain.GarageArea>1200].index,axis=0)

for i in continuous_feature_index:

	# plt.subplot(211)
	# plt.scatter(np.log(xTrain[i]+1),yTrain)
	# plt.xlabel(i)
	# plt.subplot(212)
	plt.scatter(xTrain[i],xTrain['SalePrice'])
	plt.xlabel(i)
	plt.show()
# plt.hist(yTrain)
# plt.ylabel('SalePrice')
# plt.show()
plt.scatter(xTrain['MasVnrArea'],yTrain)
plt.show()
# plt.hist(xTrain[categorical_feature_index])
# plt.show()