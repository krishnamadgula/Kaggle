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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso
# from sklearn
from sklearn.cross_validation import KFold,train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import xgboost

# Continuous
# Index([u'Id', u'MSSubClass', u'LotFrontage', u'LotArea',. u'OverallQual',
#        u'OverallCond', .u'YearBuilt', u'YearRemodAdd',. u'MasVnrArea',
#        u'BsmtFinSF1', u'BsmtFinSF2', u'BsmtUnfSF', .u'TotalBsmtSF', u'1stFlrSF',
#        u'2ndFlrSF', u'LowQualFinSF', .u'GrLivArea', u'BsmtFullBath',
#        u'BsmtHalfBath', .u'FullBath', u'HalfBath', u'BedroomAbvGr',
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
def featureEngineer(data,cont_feature,cat_feature):
	# cont_index=cont_index.tolist()
	# cont_index.remove('Id')
	# data=data[cont_index]
	
	data['HouseAge']=2017-data['YearBuilt']
	data['RemodAfter']=data['YearRemodAdd']-data['YearBuilt']
	data['OwnedPeriod']=data['YrSold']-data['YearBuilt']
	data['GarageYrBlt']=2017-data['GarageYrBlt']
	data=data.drop('YearBuilt',1)
	data=data.drop('YearRemodAdd',1)
	data=data.drop('YrSold',1)

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
	data=data.drop('Utilities',1)
	data=data.drop('Street',1)
	data=data.drop('Condition2',1)
	# cat_feature.remove('PoolQC')
	cat_feature.remove('Utilities')
	cat_feature.remove('Street')
	cat_feature.remove('Condition2')
	# cont_feature=temp
	return data,cont_feature,cat_feature

def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


def runAlgorithm():
	file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\DumpFile','rb')
	xTrain,yTrain,xTest=pickle.load(file)
	file.close()
	
	file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\featureIndices','rb')
	continuous_feature_index,categorical_feature_index=pickle.load(file)
	file.close()


	continuous_feature_index_tr=continuous_feature_index
	categorical_feature_index_tr=categorical_feature_index
	# print len(continuous_feature_index_tr),len(categorical_feature_index_tr)
	print xTrain.info(),xTest.info()
	# print len(continuous_feature_index_ts),len(categorical_feature_index_ts)


	# cont_feat=continuous_feature_index.tolist()
	# cont_feat.remove('Id')

	# xTrain=xTrain[cont_feat]
	# xTest=xTest[cont_feat]
	# print type(continuous_feature_index)
	# xTrain,dummy,dummy=featureEngineer(xTrain,continuous_feature_index_tr,categorical_feature_index_tr)
	# xTest,continuous_feature_index,categorical_feature_index=featureEngineer(xTest,continuous_feature_index_ts,categorical_feature_index_ts)
	# print continuous_feature_index
	xTrain[continuous_feature_index]=np.log(xTrain[continuous_feature_index]+1)
	xTest[continuous_feature_index]=np.log(xTest[continuous_feature_index]+1)
	norm=Normalizer()
	# xTrain[continuous_feature_index]=norm.fit_transform(xTrain[continuous_feature_index])
	# xTest[continuous_feature_index]=norm.fit_transform(xTest[continuous_feature_index])
	
	# print xTrain.info(),xTest.info()
	# print yTrain

	x_train_xgb=np.array(xTrain)
	y_train=np.array(yTrain)
	# for_linear=['HouseAge','TotalBsmtSF','GrLivArea','FullBath','GarageQual','ExterQual']
	x_train_lin=np.array(xTrain)
	regr_lin=SVR(C=0.5,kernel='linear',epsilon=0.2)
	# pca=PCA(n_components=10)
	# pca.fit(xTrain[continuous_feature_index])
	# xt=pca.transform(xTrain[continuous_feature_index])
	# print xt.shape
	regr_xgb=xgboost.XGBRegressor(colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
	cv=KFold(len(xTrain),n_folds=2,shuffle=True)
	total=0
	total_xgb=0
	for train,test in cv:
		regr_xgb.fit(x_train_xgb[train],y_train[train])
		# regr_lin.fit(x_train_lin[train],y_train[train])
		predictions1=regr_xgb.predict(x_train_xgb[test])
		# predictions=(predictions1+regr_lin.predict(x_train_lin[test]))/2
		# plt.scatter(predictions1,predictions)
		# plt.show()
		mse_xgb=mean_squared_error((y_train[test]),(predictions1))
		# mse=mean_squared_error((y_train[test]),(predictions))
		total_xgb+=mse_xgb
		# total+=mse
	total=total/2
	total_xgb=total_xgb/2
	# rmse=np.sqrt(total)
	rmse_xgb=np.sqrt(total_xgb)
	x_test_xgb=np.array(xTest)
	x_test_lin=np.array(xTest)
	print "result from only XGB" , rmse_xgb

	# print "result from both linear regr and xgb is",rmse
	salesTest=(regr_xgb.predict(x_test_xgb))#+regr_lin.predict(x_test_lin))/2
	print salesTest

	return salesTest
if __name__=='__main__':
	sales_submission=runAlgorithm()
	subData=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\sample_submission.csv')
	subData['SalePrice']=sales_submission
	subData.to_csv('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\sub2.csv')



# regr_xgb.fit(xtrain,ytrain)
# ypred=[]
# ypred=regr_xgb.predict(xvalid)
# rmse=np.sqrt(mean_squared_error(np.log(ypred+1),np.log(yvalid+1)))
