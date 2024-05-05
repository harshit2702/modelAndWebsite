import pandas as pd 
from sklearn.model_selection import train_test_split

train = pd.read_csv('/workspaces/codespaces-blank/tobacco.csv') 
test = pd.read_csv('/workspaces/codespaces-blank/tobacco_test.csv')   

columns_to_drop = ['Area', 'Current tobacco smokers (%)', 'School heads aware of COTPA, 2003  (%)', 'Students who saw anyone smoking inside the  school building or outside school property (%)', 'Students who thought it is difficult to quit once someone starts smoking tobacco (%)']
trainX = train.drop(columns=columns_to_drop)
testX = test.drop(columns=columns_to_drop)

trainY = train['Current tobacco smokers (%)']
testY = test['Current tobacco smokers (%)']

print('linear regression')
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(trainX, trainY)

pridict = reg.predict(testX)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(testY, pridict)
print('MSE: {}\n'.format(mse))

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(testY, pridict)
print('MAE: {}\n'.format(mae))

from sklearn.metrics import r2_score
r2 = r2_score(testY, pridict)
print('R2: {}\n'.format(r2))


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('Ridge')
reg = linear_model.Ridge(alpha=0.5)
reg.fit(trainX, trainY)

pridict = reg.predict(testX)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(testY, pridict)
print('MSE: {}\n'.format(mse))

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(testY, pridict)
print('MAE: {}\n'.format(mae))

from sklearn.metrics import r2_score
r2 = r2_score(testY, pridict)
print('R2: {}\n'.format(r2))

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('RidgeCV')
alphas = [0.1, 0.2, 0.3, 0.4 , 0.5, 0.6,0.7,0.8,0.9]
reg = linear_model.RidgeCV(alphas=alphas)
reg.fit(trainX, trainY)

pridict = reg.predict(testX)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(testY, pridict)
print('MSE: {}\n'.format(mse))

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(testY, pridict)
print('MAE: {}\n'.format(mae))

from sklearn.metrics import r2_score
r2 = r2_score(testY, pridict)
print('R2: {}\n'.format(r2))

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('Lasso')
