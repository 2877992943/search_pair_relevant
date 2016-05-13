#!/usr/bin/env python
# encoding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error, make_scorer


def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

def fmean_squared_error(ground_truth, predictions):
    print ground_truth.shape,predictions.shape
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_



if __name__=='__main__':
	 
	print 'load csv...'
	stemmer = SnowballStemmer('english')

	df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")[:]
	df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")[:]
	# df_attr = pd.read_csv('../input/attributes.csv')
	df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
	df_attr = pd.read_csv('../input/attributes.csv')

	num_train = df_train.shape[0]

	print 'concatenate train test...'
	df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
	print 'train test',df_all.values.shape

	
	print 'merge csv on uid...'
	df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
	print 'brand',df_brand.values.shape
	df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid');print df_all.values.shape
	df_all = pd.merge(df_all, df_brand, how='left', on='product_uid');print df_all.values.shape, 
	
	

	print 'stem...'
	df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x));
	df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
	df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
	print np.unique(df_all['search_term'].values).shape,np.unique(df_all['product_title'].values).shape,np.unique(df_all['product_description'].values).shape

	"""

	print 'generate fea...'
	df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

	df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

	df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
	df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

	df_all.to_csv('../df_all_simpleVersion.csv')
	"""
	
	##################
	"""
	df_all = pd.read_csv('../df_all_simpleVersion_whole.csv', encoding="ISO-8859-1")
	df_all = df_all.drop(['search_term','product_title','product_description','product_info','brand'],axis=1)

	df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")[:]
	num_train = df_train.shape[0]

	df_train = df_all.iloc[:num_train]
	df_test = df_all.iloc[num_train:]
	id_test = df_test['id']
	

	y_train = df_train['relevance'].values;print 'y',y_train.shape
	X_train = df_train.drop(['id','relevance'],axis=1).values;print 'x1',X_train.shape
	X_test = df_test.drop(['id','relevance'],axis=1).values;print 'x2',X_test.shape

	#####cross valid
	from sklearn.cross_validation import train_test_split
	missing_indicator=-1
	xtrain,xvalid,ytrain,yvalid=train_test_split(X_train,y_train,test_size=0.3)

	print 'train...'
	rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
	clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
	clf.fit(xtrain, ytrain)
	y_pred_train = clf.predict(xtrain);print fmean_squared_error(ytrain, y_pred_train)
	y_pred_valid = clf.predict(xvalid);print fmean_squared_error(yvalid, y_pred_valid)
	y_pred_test=clf.predict(X_test)

	pd.DataFrame({"id": id_test, "relevance": y_pred_test}).to_csv('../submission.csv',index=False)
	"""
	 


	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



