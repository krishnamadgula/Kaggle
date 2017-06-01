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
def runAlgorithm():
	file=open('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\DumpFile','rb')
	xTrain,yTrain,xTest=pickle.load(file)
	print type(xTrain),type(yTrain),type(xTest)
	# print xTrain.info(),xTest.info()
	print yTrain
	list_x_train=np.array(xTrain)
	list_y_train=np.array(yTrain)
	regr_xgb=xgboost.XGBRegressor(n_estimators=500,max_depth=2)
	cv=KFold(len(xTrain),n_folds=2,shuffle=True)
	total=0
	for train,test in cv:
		regr_xgb.fit(list_x_train[train],list_y_train[train])
		predictions=regr_xgb.predict(list_x_train[test])
		mse=mean_squared_error(list_y_train[test],predictions)
		total+=mse
	total=total/2	
	rmse=np.sqrt(total)
	list_x_test=np.array(xTest)
	print (xTrain.dtypes.index)
	print (xTest.dtypes.index)
	salesTest=regr_xgb.predict(list_x_test)
	return salesTest
if __name__=='__main__':
	sales_submission=runAlgorithm()
	subData=pd.read_csv('C:\Users\Krishna\DataScienceCompetetions\Kaggle\KaggleHouseSales\\submission.csv')
	subData['SalePrice']=sales_submission

# regr_xgb.fit(xtrain,ytrain)
# ypred=[]
# ypred=regr_xgb.predict(xvalid)
# rmse=np.sqrt(mean_squared_error(np.log(ypred+1),np.log(yvalid+1)))
