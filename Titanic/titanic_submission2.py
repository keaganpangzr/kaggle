import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


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
3. test different individual models
4. use ensemble of models
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

#Split data with Hold-out method
features =['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q','Embarked_S']
X = df_train[features]
y = df_train['Survived']

#model selection with GridSearch
param_grid = {'n_estimators': [50, 100, 200,300], 'max_depth': [5, 10, 20, 50, 100], 'max_features': [5,6,7,8,9], 'bootstrap':[True]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X,y)

print(grid_search.best_params_)
print(grid_search.best_score_)

optimized_random_forest = grid_search.best_estimator_



#Feature engineering for submission test data set
df_test = feature_engineering(df_test)

#Impute missing age and fare in df_test with mean of df_train
df_test['Age'] = df_test['Age'].fillna(df_train['Age'].mean())
df_test['Fare'] = df_test['Fare'].fillna(df_train['Fare'].mean())

X = df_test[features]
y_pred = optimized_random_forest.predict(X)

df_submission = pd.DataFrame()
df_submission['PassengerId'] = df_test['PassengerId']
df_submission['Survived'] = y_pred

df_submission.to_csv('submission1.csv', index=False)









