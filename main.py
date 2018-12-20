import os
import time
from pyfiglet import Figlet

import numpy as np 
import pandas as pd
from IPython.display import display

f = Figlet()

pd.set_option('display.max_columns', 15)



#Some founctions

def doLines(lns = 1):
    print(('\n' * lns))

def sleep(sec = 1):
    time.sleep(sec)

def customPrint(text, linesAfter = 0, LinesBefore = 0):
    if LinesBefore > 0:
        doLines(LinesBefore)
    print(f.renderText(text))
    if linesAfter > 0:
        doLines(linesAfter)

def wait():
    input('\nPress \'Enter\' key to continue ')
    

try:
    data = pd.read_csv('./data/vgsales.csv')
    customPrint('Dataset loaded', 10, 10)
except:
    customPrint('Unable to load dataset', 10, 10)

sleep(2)




customPrint('Data')

#Displaying data (only 15 rows)
display(data[:15])

wait()






#Clean some 'Year' missing values
data = data[np.isfinite(data['Year'])]


euSales = data['EU_Sales']
features = data.drop(['Name', 'Global_Sales', 'EU_Sales'], axis = 1)


customPrint('target columns', 0, 10)

# Displaying our features and target columns... 
display(euSales[:5])

customPrint('features', 1, 1)

display(features[:5])

wait()



customPrint('Principal Component Analysis', 0, 10)

#Principal Component Analysis
#Firstly, I am dividing the features data set into two as follows. 

salesFeatures = features.drop(['Rank', 'Platform', 'Year', 'Genre', 'Publisher'], 
                              axis = 1)
otherFeatures = features.drop(['NA_Sales', 'JP_Sales', 'Other_Sales', 'Rank'], 
                              axis = 1)

#Secondly, I am obtaining the PCA transformed features...

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
pca.fit(salesFeatures)
salesFeaturesTransformed = pca.transform(salesFeatures)

# inally, I am merging the new transfomed salesFeatures 
#(...cont) column back together with the otherFeatures columns...

salesFeaturesTransformed = pd.DataFrame(data = salesFeaturesTransformed, 
                                        index = salesFeatures.index, 
                                        columns = ['Sales'])
rebuiltFeatures = pd.concat([otherFeatures, salesFeaturesTransformed], 
                            axis = 1)

display(rebuiltFeatures[:5])

wait()



customPrint('Processing our data', 0, 10)

#This code is inspired by udacity project 'student intervention'.
temp = pd.DataFrame(index = rebuiltFeatures.index)

for col, col_data in rebuiltFeatures.iteritems():
    
    if col_data.dtype == object:
        col_data = pd.get_dummies(col_data, prefix = col)
        
    temp = temp.join(col_data)
    
rebuiltFeatures = temp
display(rebuiltFeatures[:5])

wait()







#Dividing the data into training and testing sets...
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rebuiltFeatures, 
                                                    euSales, 
                                                    test_size = 0.2, 
                                                    random_state = 2)











customPrint('Model Selection', 0, 10)

#Creating & fitting a Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

regDTR = DecisionTreeRegressor(random_state = 4)
regDTR.fit(X_train, y_train)
y_regDTR = regDTR.predict(X_test)

from sklearn.metrics import r2_score
print ('r2_score on the DTR model: ')
print (r2_score(y_test, y_regDTR))

doLines(1)
#Creating a K Neighbors Regressor
from sklearn.neighbors import KNeighborsRegressor

regKNR = KNeighborsRegressor()
regKNR.fit(X_train, y_train)
y_regKNR = regKNR.predict(X_test)

print ('r2_score on the KNR model: ')
print (r2_score(y_test, y_regKNR))


wait()








customPrint('Optimizing DTR Model', 0, 10)


#This code is inspired by udacity project 'student intervention'
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
cv_sets = ShuffleSplit(n_splits = 10, 
                       test_size = 0.2, random_state = 2)
regressor = DecisionTreeRegressor(random_state = 4)
params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
          'splitter': ['best', 'random']}
scoring_func = make_scorer(r2_score)
    
grid = GridSearchCV(regressor, params, cv = cv_sets, 
                    scoring = scoring_func)
grid = grid.fit(X_train, y_train)

optimizedReg = grid.best_estimator_
y_optimizedPrediction = optimizedReg.predict(X_test)

print ('The r2_score of the optimal regressor is:')
print (r2_score(y_test, y_optimizedPrediction))
wait()
