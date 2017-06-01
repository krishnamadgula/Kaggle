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
file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\featureIndices','rb')
continuous_feature_index,categorical_feature_index=pickle.load(file)
file.close()
file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\DumpFile','rb')
xTrain,yTrain,xTest=pickle.load(file)
file.close()

# Continuous
# Index([u'Id', u'MSSubClass', u'LotFrontage', u'LotArea', u'OverallQual',
#        u'OverallCond', u'YearBuilt', u'YearRemodAdd', u'MasVnrArea',
#        u'BsmtFinSF1', u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF', u'1stFlrSF',
#        u'2ndFlrSF', u'LowQualFinSF', u'GrLivArea', u'BsmtFullBath',
#        u'BsmtHalfBath', u'FullBath', u'HalfBath', u'BedroomAbvGr',
#        u'KitchenAbvGr', u'TotRmsAbvGrd', u'Fireplaces', u'GarageYrBlt',
#        u'GarageCars', u'GarageArea', u'WoodDeckSF', u'OpenPorchSF',
#        u'EnclosedPorch', u'3SsnPorch', u'ScreenPorch', u'PoolArea', u'MiscVal',
#        u'MoSold', u'YrSold', u'SalePrice']
# 
# Categorical
# ['MSZoning', 'Street', 'Alley', 'LotShape'
# , 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood'
# , 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
# 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
# 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
# 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
# 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', '
# GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
# 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
# print xTrain[continuous_feature_index].info()
# plt.scatter(-xTrain['YearRemodAdd']+xTrain['YrSold'],yTrain)
# plt.show()
plt.scatter(xTrain['Neighborhood'],yTrain)
plt.show()