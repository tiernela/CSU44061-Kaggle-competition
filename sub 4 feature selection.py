#feature selection : instance, wears glasses, hair color

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

## Data 
train =pd.read_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\tcd ml 2019-20 income prediction training (with labels).csv')
test = pd.read_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\tcd ml 2019-20 income prediction test (without labels).csv')

train['train'] =1
test['test'] = 0

income =  pd.concat([train, test])


## data cleaning : age and record
mu_age = round(np.mean(income['Age']), 0)
income['Age']= income['Age'].fillna(mu_age)

mu_recordyear = round(np.mean(income['Year of Record']), 0)
income['Year of Record']= income['Year of Record'].fillna(mu_recordyear)

##Data cleaning : Gender
income.Gender[income.Gender == 'male' ] = 1
income.Gender[income.Gender == 'female' ] = 2
income.Gender[income.Gender == 'other'  ] = 0
income.Gender[income.Gender == 'unknown' ] = 0
income.Gender= income.Gender.fillna(0)

income = pd.concat([income ,pd.get_dummies(income['Gender'], prefix='Gender' )],axis=1)
income.drop(['Gender'], axis =1, inplace = True)
income.drop(['Gender_0'], axis =1, inplace = True)

##Data cleaning: Hair
income['Hair Color'][income['Hair Color'] == 'Unknown' ] = 'Other' 
income['Hair Color'][income['Hair Color'] == '0' ] = 'Other'
income['Hair Color']= income['Hair Color'].fillna('Other')

income = pd.concat([income ,pd.get_dummies(income['Hair Color'], prefix='Hair_Col' )],axis=1)
income.drop(['Hair Color'], axis =1, inplace = True)
income.drop(['Hair_Col_Other'], axis =1, inplace = True)


## Data cleaning : University Degree
income['University Degree'][income['University Degree'] == '0' ] = 'No'
income['University Degree']= income['University Degree'].fillna('No')
income = pd.concat([income ,pd.get_dummies(income['University Degree'], prefix='Degree' )],axis=1)
income.drop(['University Degree'], axis =1, inplace = True)
income.drop(['Degree_No'], axis =1, inplace = True)



### Features selection
#income.drop(['Body Height [cm]'], axis =1, inplace = True)
income.drop(['Instance'], axis =1, inplace = True)
income.drop(['Wears Glasses'], axis =1, inplace = True)
income.drop(['Hair_Col_Black'], axis =1, inplace = True)
income.drop(['Hair_Col_Brown'], axis =1, inplace = True)
income.drop(['Hair_Col_Blond'], axis =1, inplace = True)
income.drop(['Hair_Col_Red'], axis =1, inplace = True)
#income.drop(['Size of City'], axis =1, inplace = True)
#income.drop(['Year of Record'], axis =1, inplace = True)

#country
income = pd.concat([income ,pd.get_dummies(income['Country'], prefix='Country' )],axis=1)
income.drop(['Country'], axis =1, inplace = True)
income.drop(['Country_Afghanistan'], axis =1, inplace = True)


#profession
income = pd.concat([income ,pd.get_dummies(income['Profession'], prefix='Prof' )],axis=1)
income.drop(['Prof_youth initiatives lead advisor'], axis =1, inplace = True)
income.drop(['Profession'], axis =1, inplace = True)



## Splitting Data
income_train = income[income['train']==1]
income_test = income[income['test']==0]


## Data cleaning : salary
#income_train['Income in EUR'] = round(income_train['Income in EUR'], 0)
##min(income_train['Income in EUR'])
##max(income_train['Income in EUR'])
#mu = np.average(income_train['Income in EUR'])
#sigma = np.std(income_train['Income in EUR'])
#LL = mu - 2*sigma
#UL = mu + 2*sigma
#income_train['Income in EUR'] = income_train['Income in EUR'].clip(LL, UL)

income_train.drop(['Income'], axis=1, inplace=True)
income_test.drop(['Income in EUR'], axis=1, inplace=True)
income_train.drop(['train'], axis=1, inplace=True)
income_train.drop(['test'], axis=1, inplace=True)
income_test.drop(['train'], axis=1, inplace=True)
income_test.drop(['test'], axis=1, inplace=True)


## Multilinear regression
#
# 
#X = income_train.loc[:, income_train.columns !='Income in EUR']
#Y = income_train.loc[:, 'Income in EUR']
##xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#
#
#regress = linear_model.LinearRegression()
#regress.fit(X,Y)
#coeff = regress.coef_
#print('Intercept: \n', regress.intercept_)
#print('Coefficients: \n', regress.coef_)
#
### predictions
#income_test['Income'] =  regress.predict(income_test.loc[:, income_test.columns != 'Income'])
#
#Results = income_test[['Income']].copy()
#Results.to_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Results.csv')



## Multilinear regression 
#X = income_train.loc[:, income_train.columns !='Income in EUR']
#Y = income_train.loc[:, 'Income in EUR']
#
#regress = linear_model.LinearRegression()
#regress.fit(X,Y)
#print('Intercept: \n', regress.intercept_)
#print('Coefficients: \n', regress.coef_)
#
### predictions
#income_test['Income'] =  regress.predict(income_test.loc[:, income_test.columns != 'Income'])
#
#Results = income_test[[ 'Income']].copy()
#Results.to_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Results.csv')



## Multilinear regressionon train set divide
X = income_train.loc[:, income_train.columns !='Income in EUR']
Y = income_train.loc[:, 'Income in EUR']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


regress = linear_model.LinearRegression()
regress.fit(X_train, Y_train)
Y_pred = regress.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))




