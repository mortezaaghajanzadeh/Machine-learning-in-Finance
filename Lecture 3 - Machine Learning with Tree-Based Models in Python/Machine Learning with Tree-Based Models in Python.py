#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
#%%
path = r"Machine-Learning-with-Tree-Based-Models-in-Python-master/{}"
wbc = pd.read_csv(path.format('wbc.csv'))
X = wbc[['radius_mean', 'concave points_mean']]
y = wbc['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
#%%
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
# Instatiate logreg
logreg = LogisticRegression(random_state=1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
dt = DecisionTreeClassifier(criterion='gini', random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
mpg = pd.read_csv(path.format('auto.csv'))
mpg.head()
#%%
X = mpg[['displ', 'hp']]
y = mpg['mpg']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2)
rmse_dt
#%%
seed = 123
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=seed)
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.14, random_state=seed)
MSE_CV = - cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
dt.fit(X_train, y_train)
y_predict_train = dt.predict(X_train)
print('Train RMSE: {:.2f}'.format((MSE(y_train, y_predict_train))**(1/2)))
y_predict_test = dt.predict(X_test)
print('Test RMSE: {:.2f}'.format((MSE(y_test, y_predict_test))**(1/2)))