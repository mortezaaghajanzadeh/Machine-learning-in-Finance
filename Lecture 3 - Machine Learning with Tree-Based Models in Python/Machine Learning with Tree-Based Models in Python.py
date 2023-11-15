#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN

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
# %% 
wbc = pd.read_csv(path.format('wbc.csv'))
wbc.head()
#%%
X = wbc[wbc.columns[2:-1]]
y = wbc['diagnosis']

seed = 1
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=seed)

lr = LogisticRegression(random_state=seed)
knn = KNN()
dt = DecisionTreeClassifier(random_state=seed)

classifiers = [
    ('Logistic Regression', lr),
    ('K Nearest Neighbours', knn),
    ('Classification Tree', dt)]
for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(X_train, y_train)

    # Predict y_pred
    y_pred = clf.predict(X_test)
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
print('Voting Classifier: {:.3f}'.format(accuracy_score(y_test, y_pred)))
#%%
SEED = 1

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=SEED)

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)

dt.fit(X_train, y_train)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)

accuracy_score(y_test, y_pred), accuracy_score(y_test, dt.predict(X_test))
#%%
SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=SEED)
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300,oob_score=True, n_jobs=-1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
oob_accuracy = bc.oob_score_
print('Test set accuracy: {:.3f}'.format(test_accuracy))
print('OOB accuracy: {:.3f}'.format(oob_accuracy))
#%%
X = mpg[['displ', 'hp']]
y = mpg['mpg']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=3)

rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=3)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
#%%
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)
sorted_importances_rf = importances_rf.sort_values()
sorted_importances_rf.plot(kind='barh', color='lightgreen')
#%%
SEED = 1
X = wbc[wbc.columns[2:-1]]
y = wbc['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=SEED)
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
adb_clf.fit(X_train, y_train)
# Compute the probabilities of obtaining the positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))
#%%
X = mpg[['displ', 'hp']]
y = mpg['mpg']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=3)
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=3)
gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of gbt: {:.3f}'.format(rmse_test))
#%%
X = mpg[['displ', 'hp']]
y = mpg['mpg']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=3)
sgbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, subsample=0.8, max_features=0.2, random_state=3)
sgbt.fit(X_train, y_train)
y_pred = sgbt.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of sgbt: {:.3f}'.format(rmse_test))
#%%
SEED = 1
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)
print(dt.get_params())
#%%
params_dt = {'max_depth': [2,3,4], 'min_samples_leaf': [0.12,0.14,0.16,0.18]}
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
grid_dt.fit(X_train, y_train)
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)
best_CV_score = grid_dt.best_score_
print('Best CV accuracy'.format((-best_CV_score)**(1/2)))
best_model = grid_dt.best_estimator_
test_acc = best_model.score(X_test, y_test)
print('Test set accuracy of best model: {:.3f}'.format(test_acc))
#%%
rf = RandomForestRegressor(random_state=2)
print(rf.get_params())
params_rf = {'n_estimators': [100,350,500], 'max_depth': [4,6,8], 'min_samples_leaf': [0.1,0.2], 'max_features': ['log2','sqrt']}
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_hyperparams = grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)
best_CV_score = grid_rf.best_score_
print('Best CV accuracy'.format((-best_CV_score)**(1/2)))
best_model = grid_rf.best_estimator_
test_acc = best_model.score(X_test, y_test)
print('Test set accuracy of best model: {:.3f}'.format(test_acc))

