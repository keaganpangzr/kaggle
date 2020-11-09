import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier

pd.set_option("display.max_columns", 20)

'''
Notes

key decisions
1. how to deal with missin age and embarked data -> imputation vs drop values
2. hold-out vs cross-validation
4. scaling features or not
3. model selection -> single model or ensemble 

Features checklist
- Sex -> one-hot -
- Pclass -> label encoded, leave as is
- Age -> normalize
- SibSp & Parch -> label encoded, leave as is
- Fare -> normalize
- Embarked -> one-hot (test leaving zero-hot)


How to improve:
1. use cross-validation instead of hold-out, train model on all data - DONE
2. impute age and fare in training set - DONE with mean
3. test different individual models (add XGboost)
4. use ensemble of methods (add voting)
5. add hyper-parameter tuning - DONE with gridsearch


'''

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

#Data pre-processing: Drop rows where age data is missing
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())


#df_train = df_train[df_train['Age'].isnull() == False]
df_train = df_train[df_train['Embarked'].isnull() == False]

def feature_engineering(df_train):
    #Sex: One-hot
    df_train['Sex'] = pd.get_dummies(df_train['Sex'])
   
    #Age & Fare: Normalize
    scaler = MinMaxScaler()
    df_train[['Age','Fare']] = scaler.fit_transform(df_train[['Age', 'Fare']])

    #Embarked: One-hot
    df_train[['Embarked_C', 'Embarked_Q', 'Embarked_S']] = pd.get_dummies(df_train['Embarked'])

    return df_train

df_train = feature_engineering(df_train)

#Split data with Hold-out method, do not include Embarked_S to avoid dummy variable trap
features =['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q']
X = df_train[features]
y = df_train['Survived']


#cross-validator method
cv = KFold(n_splits=5, shuffle=True, random_state=1)

#KNN
param_grid_knn = {'n_neighbors':range(2,21,2), 'metric':['euclidean','manhattan','minkowski'],'weights':['uniform','distance']}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid_knn, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search_knn.fit(X, y)
optimized_KNN = grid_search_knn.best_estimator_
print('Optimized KNN accuracy: ', grid_search_knn.best_score_)

#Random Forest
param_grid_rf = {'n_estimators':range(50, 250, 50), 'max_depth': range(5,20,5), 'max_features': [5,6,7,8], 'bootstrap':[True]}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_rf, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search_rf.fit(X,y)
optimized_random_forest = grid_search_rf.best_estimator_
print('Optimized Random Forest accuracy: ', grid_search_rf.best_score_)

#Logistic regression
param_grid_logreg = {'solver':['newton-cg','lbfgs','liblinear'], 'C':[0.01, 0.1, 1, 10, 100], 'penalty':['l2']}
grid_search_logreg = GridSearchCV(LogisticRegression(), param_grid= param_grid_logreg, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search_logreg.fit(X,y)
optimized_logreg = grid_search_logreg.best_estimator_
print('Optimized LogReg accuracy: ', grid_search_logreg.best_score_)

#Adaboost
param_grid_adaboost = {'n_estimators':range(50,200,50), 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.5]}
grid_search_adaboost = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid_adaboost, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search_adaboost.fit(X,y)
optimized_adaboost = grid_search_adaboost.best_estimator_
print('Optimized Adaboost accuracy: ', grid_search_adaboost.best_score_)


#Compare estimators
models = [grid_search_knn, grid_search_logreg, grid_search_rf]

majority_vote_clf = VotingClassifier([('optimized_KNN', optimized_KNN), ('optimized_RF', optimized_random_forest), ('optimized_logreg', optimized_logreg), ('optimized_adaboost', optimized_adaboost)], voting='soft')
majority_vote_clf.fit(X,y)
print('Majority vote of optimized models accuracy: ', cross_val_score(majority_vote_clf, X, y, cv=cv, n_jobs=-1).mean())

#Feature engineering for submission test data set
df_test = feature_engineering(df_test)

#Impute missing age and fare in df_test with mean of df_train
df_test['Age'] = df_test['Age'].fillna(df_train['Age'].mean())
df_test['Fare'] = df_test['Fare'].fillna(df_train['Fare'].mean())

X = df_test[features]
y_pred = majority_vote_clf.predict(X)

df_submission = pd.DataFrame()
df_submission['PassengerId'] = df_test['PassengerId']
df_submission['Survived'] = y_pred

df_submission.to_csv('submission1.csv', index=False)









