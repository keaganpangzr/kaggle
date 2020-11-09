import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

pd.set_option("display.max_columns", 20)

'''
Notes

Features checklist
- Sex -> one-hot
- Pclass -> label encoded, leave as is
- Age -> normalize
- Family size
- is alone
- Fare -> normalize
- Embarked -> one-hot (test leaving zero-hot)

'''

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

def preprocess(df_train):
    #Impute data
    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
    df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
    df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())

    return df_train

df_train = preprocess(df_train)

#df_train['Deck'] = df_train['Cabin'].astype(str).str[0]
#print(df_train.groupby('Deck')['Survived'].mean())
#print(df_train['Survived'].mean())
#print(df_train[df_train['Deck']=='B'])




def feature_engineering(df_train):
    #Sex: One-hot
    df_train['Sex'] = pd.get_dummies(df_train['Sex'])
   
    #Age & Fare: Normalize
    scaler = MinMaxScaler()
    df_train[['Age','Fare']] = scaler.fit_transform(df_train[['Age', 'Fare']])

    #Embarked: One-hot
    df_train[['Embarked_C', 'Embarked_Q', 'Embarked_S']] = pd.get_dummies(df_train['Embarked'])

    #Family size and is_alone
    df_train['Family_Size'] = df_train['SibSp'] + df_train['Parch'] + 1
    df_train['Is_Alone'] = np.where(df_train['Family_Size'] == 1, 1, 0)

    #Name Prefixes
    df_train['Name_Prefix'] = [name.split(', ')[1].split('.')[0] for name in df_train['Name']]
    df_train['Name_Prefix'] = [prefix if prefix in ['Mr', 'Mrs', 'Miss', 'Master'] else "Others" for prefix in df_train['Name_Prefix']]
    df_train['Name_Prefix'] = LabelEncoder().fit_transform(df_train['Name_Prefix'])

    #Deck
    decks = df_train['Cabin'].astype(str).str[0]
    df_train['Is_Upper_2_Decks'] = [1 if deck in ['A', 'B'] else 0 for deck in decks] 

    return df_train

df_train = feature_engineering(df_train)

print(df_train.head(10))



#Split data with Hold-out method, do not include Embarked_S to avoid dummy variable trap
features =['Sex', 'Pclass', 'Age', 'Family_Size', 'Is_Alone', 'Fare', 'Embarked_C', 'Embarked_Q', 'Name_Prefix', 'Is_Upper_2_Decks']
X = df_train[features]
y = df_train['Survived']


#cross-validator
cv = KFold(n_splits=5, shuffle=True, random_state=1)

#KNN
param_grid_knn = {'n_neighbors':range(2,21,2), 'metric':['euclidean','manhattan','minkowski'],'weights':['uniform','distance']}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid_knn, scoring='accuracy', cv=cv, n_jobs=-1)

#Random Forest
param_grid_rf = {'n_estimators':range(50, 250, 50), 'max_depth': range(5,20,5), 'max_features': range(3, len(features)), 'bootstrap':[True]}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_rf, scoring='accuracy', cv=cv, n_jobs=-1)

#Logistic regression
param_grid_logreg = {'solver':['newton-cg','lbfgs','liblinear'], 'C':[0.01, 0.1, 1, 10, 100], 'penalty':['l2']}
grid_search_logreg = GridSearchCV(LogisticRegression(), param_grid= param_grid_logreg, scoring='accuracy', cv=cv, n_jobs=-1)

#Adaboost
param_grid_adaboost = {'n_estimators':range(50,200,50), 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.5]}
grid_search_adaboost = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid_adaboost, scoring='accuracy', cv=cv, n_jobs=-1)

#XGboost
param_grid_xgb ={'learning_rate':[0.1, 0.2, 0.5], 'min_child_weight':[1,5,10], 'subsample':[0.6, 0.8, 1], 'colsample_bytree': [0.6, 0.8, 1], 'max_depth':[3, 4, 5]}
grid_search_xgb = GridSearchCV(XGBClassifier(), param_grid=param_grid_xgb, cv=cv, scoring='accuracy', n_jobs=-1)

#Fit and cross-validate models
models = {'KNN': grid_search_knn, 'Random Forest': grid_search_rf, 'Logistic Regression': grid_search_logreg, 'AdaBoost': grid_search_adaboost, 'XGBoost': grid_search_xgb}

for name, model in models.items():
    model.fit(X,y)
    print(name + ': ', model.best_score_)

#Majority voting 
majority_vote_clf = VotingClassifier([(name, model.best_estimator_) for name, model in models.items()], voting='soft')
print('Majority Vote: ', cross_val_score(majority_vote_clf, X, y, cv=cv, n_jobs=-1).mean())
majority_vote_clf.fit(X, y)





#Submission
df_test = preprocess(df_test)
df_test = feature_engineering(df_test)

X_test = df_test[features]
y_pred = majority_vote_clf.predict(X_test)

df_submission = pd.DataFrame()
df_submission['PassengerId'] = df_test['PassengerId']
df_submission['Survived'] = y_pred

df_submission.to_csv('submission1.csv', index=False)









