import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

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


Submission 1 - Quick and dirty
- drop observation with missing age/embarked
- hold-out using train test split
- single model


How to improve:
1. use cross-validation instead of hold-out
2. impute age and fare in training set
3. test different individual models
4. use ensemble of models
5. add hyper-parameter tuning


'''

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

#Data pre-processing: Drop rows where age data is missing
df_train = df_train[df_train['Age'].isnull() == False]
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
features =['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = df_train[features]
y = df_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)

#classifier 
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

#Feature engineering for submission test data set
df_test = feature_engineering(df_test)

#Impute missing age and fare in df_test with mean of df_train
df_test['Age'] = df_test['Age'].fillna(df_train['Age'].mean())
df_test['Fare'] = df_test['Fare'].fillna(df_train['Fare'].mean())

X = df_test[features]
y_pred = clf.predict(X)

df_submission = pd.DataFrame()
df_submission['PassengerId'] = df_test['PassengerId']
df_submission['Survived'] = y_pred

df_submission.to_csv('submission1.csv', index=False)










