#!/usr/bin/env

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 

# ML Libraries
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn import linear_model

pd.options.mode.chained_assignment = None 


def impute(Xi):
	Xi['LotFrontage']=Xi.LotFrontage.fillna(Xi.LotFrontage.mean())
	Xi['Alley'] = Xi['Alley'].fillna('None')
	Xi['MasVnrType'] = Xi['MasVnrType'].fillna('None')
	Xi['MasVnrArea'] = Xi['MasVnrArea'].fillna(0)

	Xi[['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinType2', 'BsmtExposure',\
	 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']] =\
	 Xi[['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinType2', 'BsmtExposure',\
	 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']] .fillna('NA')
	
	Xi['GarageType'] = Xi['GarageType'].fillna('None')
	Xi['GarageYrBlt'] = (Xi['GarageYrBlt'] - Xi['GarageYrBlt'].max())*-1.0
	Xi['GarageYrBlt'] = Xi['GarageYrBlt'].fillna(-1)
	## Make sure other years are similarly scaled
	Xi['YearBuilt'] = (Xi['YearBuilt'] - Xi['YearBuilt'].max())*-1.0
	Xi['YearRemodAdd'] = (Xi['YearRemodAdd'] - Xi['YearRemodAdd'].max())*-1.0
	Xi['YrSold'] = (Xi['YrSold'] - Xi['YrSold'].max())*-1.0

	float_cols = Xi.select_dtypes(include = ['float64']).columns
	for col in float_cols:
		Xi[col] = Xi[col].fillna(Xi[col].mean())
	return Xi

def encodeCols(Xi, cols1):
	Qs = {'Ex':5, 'Gd': 4, 'TA':3, 'Fa':2, 'Po':1, 'NA':-1}
	for col in cols1:
		Xi[col] = Xi[col].fillna('NA')
		Xi[col] = Xi[col].apply(lambda x: Qs[x])
	return Xi

def convCat2Num(Xi, cols2):
	for col in cols2:
		Xi = pd.concat([Xi, pd.get_dummies(Xi[col]).rename(columns = lambda x:str(col)+'_'+x)], axis = 1)
	for col in cols2:
		del Xi[col]
	return Xi


def preprocessing_code(Xi):
	Xi = impute(Xi)

	# Encode ranked data
	ranked_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',\
	'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
	Xi = encodeCols(Xi, ranked_cols)

	# One-hot-encode non categorical data
	cols_that_need_to_be_encoded = Xi.select_dtypes(include = ['object']).columns
	Xi = convCat2Num(Xi, cols_that_need_to_be_encoded)

	# Min-Max Scaling
	minmax = preprocessing.MinMaxScaler()
	Xi.iloc[:,1:] = minmax.fit_transform(Xi.iloc[:,1:])
	return Xi

def rmse(y, y_pred):
	mse_grad = mean_squared_error(y, y_pred)
	return np.sqrt(mse_grad)

def main():
	# Load in the dataset
	trainDf = pd.read_csv('train.csv')
	testDf = pd.read_csv('test.csv')

	# Split data sets
	X = trainDf.iloc[:,:-1]
	y = trainDf.iloc[:,-1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	# Important Features (ipython notebook) have been determined as:
	not_zero = ['OverallQual', 'GrLivArea', '1stFlrSF', 'GarageCars', 'BsmtFinSF1',\
	'BsmtQual', 'KitchenQual', 'TotRmsAbvGrd', 'FireplaceQu',\
	'YearBuilt', 'ExterQual', 'LotArea', 'WoodDeckSF', 'MasVnrArea',\
	'GarageFinish_Unf']

	X_train = preprocessing_code(X_train)
	y_train = y_train[X_train.index]

	#clf_grad = ensemble.GradientBoostingRegressor(random_state = 42, min_samples_leaf = 17,max_depth = 8, n_estimators=70, learning_rate= 0.07)
	clf_ElasticNet = linear_model.ElasticNet(alpha=0.01, l1_ratio = 0.4)
	model = Pipeline([('poly', preprocessing.PolynomialFeatures(degree=2)), ('linear', clf_ElasticNet)])
	grad_tree = model.fit(X_train[not_zero], y_train)

	predicted_grad_train = grad_tree.predict(X_train[not_zero])
	print rmse(y_train, predicted_grad_train)

	## test
	X_test = preprocessing_code(X_test)
	y_test = y_test[X_test.index]
	predicted_grad_test = grad_tree.predict(X_test[not_zero])
	print rmse(y_test, predicted_grad_test)

	# Final test
	X_test_fin = testDf
	X_test_fin = preprocessing_code(X_test_fin)
	
	predicted_grad_test_fin = grad_tree.predict(X_test_fin[not_zero])

	dictx = {'Id': X_test_fin.Id, 'SalePrice': predicted_grad_test_fin}
	outDf = pd.DataFrame(dictx)
	outDf.to_csv('out.csv', index = False)

	print len(outDf)



if __name__ == '__main__':
	main()

















