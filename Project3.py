# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:15:26 2018
"""

# Core Libraries - Data manipulation and analysis
import pandas as pd
import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
  
# Core Libraries - Machine Learning
import sklearn
import xgboost as xgb


# Importing Classifiers - Modelling
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

## Importing train_test_split,cross_val_score,GridSearchCV,KFold - Validation and Optimization
from sklearn.model_selection import  train_test_split, cross_val_score, GridSearchCV, KFold 


# Importing Metrics - Performance Evaluation
from sklearn import metrics

# Warnings Library - Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import pickle


train_set = pd.read_csv('adult_data.csv', header = None)
test_set =  pd.read_csv('adult_test.csv', skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation','relationship',
                  'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels


# Understand the Dataset and Data


train_set.shape,test_set.shape



train_set.columns
train_set.head()
test_set.columns
test_set.head()
train_set.info()
test_set.info()
train_set.get_dtype_counts()
test_set.get_dtype_counts()


#Clean the data

#Clean Column Names


train_set.columns
test_set.columns


#The columns don't have any nonsensical values, therefore there is no need to clean or change column names

#Clean Numerical Columns

#Null values
num_cols = train_set.select_dtypes(include="int64").columns.values

# num_cols = test_set.select_dtypes(include="int64").columns.values can also be used because the columns are the same
train_set[num_cols].isna().sum()
test_set[num_cols].isna().sum()
#No null values in the numerical columns of both the train_set and test_set

# Zeros

#Check if there are any rows with all row values = zero that need our consideration so that we can decide to study those rows

train_set.loc[(train_set==0).all(axis=1),num_cols].shape
test_set.loc[(train_set==0).all(axis=1),num_cols].shape


#There are no rows which have all row values == 0


#Check if there are any rows with any row values = zero that need our consideration so that we can decide to study those rows

train_set.loc[(train_set==0).any(axis=1),num_cols].shape
train_set.loc[(train_set==0).any(axis=1),num_cols].head()
train_set.loc[(train_set.drop(["capital_gain", "capital_loss"],axis=1)==0).any(axis=1),num_cols].shape
test_set.loc[(train_set==0).any(axis=1),num_cols].shape
test_set.loc[(test_set.drop(["capital_gain", "capital_loss"],axis=1)==0).any(axis=1),num_cols].shape


#There are no rows which have any row values == 0, except in captital_gain, capital_loss columns(where 0 is a valid value)

#Nonsensical values

#There are no nonsensical values in the Numerical Columns

# Clean Categorical Columns

# Null values

cat_cols = train_set.select_dtypes(include="object").columns.values
cat_cols
train_set[cat_cols].isna().sum()

test_set[cat_cols].isna().sum()


# Check Empty Values
train_set.loc[(train_set=="").any(axis=1),cat_cols].shape
test_set.loc[(train_set=="").any(axis=1),cat_cols].shape

#There are no empty strings in any of the rows

#Check Nonsensical values 


train_set[cat_cols].nunique()

for col in cat_cols:
    print(train_set[col].unique(),"\n")

test_set['workclass'].unique()

for col in cat_cols:
    print(test_set[col].unique(),"\n")


#The columns workclass, occupation and native_country have rows that have garbage values which need to be imputed or dropped in the test_set
plt.figure(figsize=(20,10))
plt.subplot(2,2,1) 
plt.title("Workclass Count Distribution")
train_set['workclass'].value_counts().plot.bar()
plt.subplot(2,2,2) 


plt.title("Occupation Count Distribution")
train_set['occupation'].value_counts().plot.bar()


plt.figure(figsize=(20,5))
plt.subplot(1,1,1) 
plt.title("Native Country Count Distribution")
train_set['native_country'].value_counts().plot.bar()

plt.figure(figsize=(20,10))
plt.subplot(2,2,1) 
plt.title("Workclass Count Distribution")
test_set['workclass'].value_counts().plot.bar()

plt.subplot(2,2,2) 
plt.title("Occupation Count Distribution")
test_set['occupation'].value_counts().plot.bar()


plt.figure(figsize=(20,5))
plt.subplot(1,1,1) 
plt.title("Native Country Count Distribution")
test_set['native_country'].value_counts().plot.bar()

train_set[train_set.workclass.str.contains("\?")].head()

test_set[test_set.workclass.str.contains("\?")].head()


(train_set.loc[(train_set==" ?").any(axis=1),cat_cols].shape[0]/train_set.shape[0])*100


(test_set.loc[(test_set==" ?").any(axis=1),cat_cols].shape[0]/test_set.shape[0])*100


# If we drop the rows containing ? values, we incur a data loss of approximately 7.5% data loss in the train_set and the test_set. Therefore we choose to drop it

train_set.drop(train_set.loc[(train_set==" ?").any(axis=1)].index, inplace= True)
train_set.shape[0]
test_set.drop(test_set.loc[(test_set==" ?").any(axis=1)].index, inplace= True)
test_set.shape[0]

test_set.loc[(test_set==" ?").any(axis=1),cat_cols].shape[0]/test_set.shape[0]


# Get Basic Statistical Information

train_set.describe()

train_set.describe(include='object')

test_set.describe()

test_set.describe(include='object')

train_set.corr()

test_set.corr()


#Explore Data

#Uni-variate

train_set[num_cols].hist(bins=50, figsize=(20,20), layout=(4,2))
plt.show()

test_set[num_cols].hist(bins=50, figsize=(20,20), layout=(3,2))
plt.show()


# Categorical Columns

for i, col in enumerate(cat_cols):
    if(col!='native_country'):
        plt.figure(i,figsize = (20,5))
        sns.countplot(y=col, data=train_set,)
    else:
        plt.figure(i,figsize = (20,10))
        sns.countplot(y=col, data=train_set)


for i, col in enumerate(cat_cols):
    if(col!='native_country'):
        plt.figure(i,figsize = (20,5))
        sns.countplot(y=col, data=test_set)
    else:
        plt.figure(i,figsize = (20,10))
        sns.countplot(y=col, data=test_set)


#Bi-variate

sns.pairplot(train_set[num_cols],kind ='reg',diag_kind='kde')

sns.pairplot(test_set[num_cols],kind ='reg',diag_kind='kde')


#None of the numerical columns are strongly correlated with each other, either in train_set or test_set. However, it is interesting to note that education_num is more correlated with capital_gain than capital_loss

for i, col in enumerate(num_cols):
    plt.figure(i,figsize = (20,5))
    sns.violinplot(x=col,y='wage_class', data=train_set)


for i, col in enumerate(num_cols):
    plt.figure(i,figsize = (20,5))
    sns.violinplot(x=col,y='wage_class', data=test_set)


# Multi-variate

plt.figure(figsize=(10,10))
sns.heatmap(train_set.corr(), annot = True,cmap= "PRGn")

plt.figure(figsize=(10,10))
sns.heatmap(test_set.corr(), annot = True,cmap= "PRGn")


#Engineer Features

# Encode Categorical Columns

for col in train_set.columns: # Loop through all columns in the dataframe
    if train_set[col].dtype == 'object': # Only apply for columns with categorical strings
        train_set[col] = pd.Categorical(train_set[col]).codes # Replace strings with an integer

for col in test_set.columns: # Loop through all columns in the dataframe
    if test_set[col].dtype == 'object': # Only apply for columns with categorical strings
        test_set[col] = pd.Categorical(test_set[col]).codes # Replace strings with an integer


#Generate Input Vector X and Output Y, and Split the Data for Training and Testing
x_train = train_set.drop('wage_class', axis =1)
y_train = train_set['wage_class']
x_test = test_set.drop('wage_class', axis =1)
y_test = test_set['wage_class']


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# Fit the Base Models and Collect the Metrics

# Logistic Regression
print("\n","***"*40,"\n")
log_res = LogisticRegression()
model_lr = log_res.fit(x_train, y_train)

y_test_pred = model_lr.predict(x_test)

y_test_pred_prob = model_lr.predict_proba(x_test)

# Generate model evaluation metrics for the Logistic Regression
print("Performance metrics of the model for the Logistic Regression")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))


#Other Classifiers


classifiers = [
            ("Logistic Regression - ", LogisticRegression()),
            ("K-Nearest Neighbors - ",  KNeighborsClassifier(2)),
            ("Naive Bayes - ",  GaussianNB()),
            ("Decision Tree - ",  DecisionTreeClassifier(max_depth=5)),
            ("Random Forest - ",  RandomForestClassifier(n_estimators=100)),
            ("AdaBoost - ",  AdaBoostClassifier(n_estimators=100)),
            ("XGBoost - ", XGBClassifier(n_estimators=100,objective='binary:logistic'))]



# Generate model evaluation metrics
for clf in classifiers:
    clf[1].fit(x_train, y_train)
    y_test_pred= clf[1].predict(x_test)
    y_test_pred_prob= clf[1].predict_proba(x_test)
    print(clf[0],
          "\n\t Accuracy: ", metrics.accuracy_score(y_test, y_test_pred),
          "\n\t Precision Score: ",metrics.precision_score(y_test, y_test_pred),
          "\n\t Recall Score: ",metrics.recall_score(y_test, y_test_pred),
          "\n\t AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]),
          "\n\t Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred),
          "\n\t Classification Report:\n ",metrics.classification_report(y_test, y_test_pred),"\n")


#Select Features
rndf = RandomForestClassifier(n_estimators=150)
rndf.fit(x_train, y_train)
importance = pd.DataFrame.from_dict({'cols':x_train.columns, 'importance': rndf.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)
plt.figure(figsize=(20,15))
sns.barplot(importance.cols, importance.importance)
plt.xticks(rotation=90)

imp_cols = importance[importance.importance >= 0.03].cols.values
imp_cols

# Generate model evaluation metrics
print("Base Models")
print('-'*60)
accuracy_base = []
precision_base =[]
recall_base = []
model_names = [i[0] for i in classifiers]
for clf in classifiers:
    clf[1].fit(x_train, y_train)
    y_test_pred= clf[1].predict(x_test)
    accuracy_base.append(metrics.accuracy_score(y_test, y_test_pred))
    precision_base.append(metrics.precision_score(y_test, y_test_pred))
    recall_base.append(metrics.recall_score(y_test, y_test_pred))
    print(clf[0],
          "\n\t Accuracy: ", metrics.accuracy_score(y_test, y_test_pred),
          "\n\t Precision Score: ",metrics.precision_score(y_test, y_test_pred),
          "\n\t Recall Score: ",metrics.recall_score(y_test, y_test_pred))

# Plotting the classification metrics for all the base models
plt.figure(figsize=(15,5))
plt.plot(model_names , accuracy_base, label = "Base - Accuracy")
plt.plot(model_names , precision_base, label = "Base - Precision")
plt.plot(model_names , recall_base, label = "Base - Recall")
plt.legend()
plt.show()

# Generate model evaluation metrics
print("Models generated with features having feature importances threshold >= 0.03")
print('-'*60)
accuracy_thresh_03 = []
precision_thresh_03 =[]
recall_thresh_03 = []
for clf in classifiers:
    clf[1].fit(x_train[imp_cols], y_train)
    y_test_pred= clf[1].predict(x_test[imp_cols])
    accuracy_thresh_03.append(metrics.accuracy_score(y_test, y_test_pred))
    precision_thresh_03.append(metrics.precision_score(y_test, y_test_pred))
    recall_thresh_03.append(metrics.recall_score(y_test, y_test_pred)) 
    print(clf[0],
          "\n\t Accuracy: ", metrics.accuracy_score(y_test, y_test_pred),
          "\n\t Precision Score: ",metrics.precision_score(y_test, y_test_pred),
          "\n\t Recall Score: ",metrics.recall_score(y_test, y_test_pred))


# Plotting the classification metrics for all the base models and models generated from features which have feature importance >=0.03

plt.figure(figsize=(15,10))
plt.plot(model_names , accuracy_base, label = "Accuracy - Base",c = 'blue')
plt.plot(model_names , accuracy_thresh_03, label = "Accuracy - Threshold 0.03", c = 'green')
 
plt.plot(model_names , precision_base, label = "Precision - Base",c = 'blue')
plt.plot(model_names , precision_thresh_03, label = "Precision - Threshold 0.03", c = 'green')
 

plt.plot(model_names , recall_base, label = "Recall - Base",c = 'blue')
plt.plot(model_names , recall_thresh_03, label = "Recall - Threshold 0.03", c = 'green')
 
plt.legend()
plt.show()

imp_cols = importance[importance.importance >= 0.014 ].cols.values
imp_cols

# Generate model evaluation metrics
print("Models generated with features having feature importances threshold >= 0.014")
print('-'*60)
accuracy_thresh_014 = []
precision_thresh_014 =[]
recall_thresh_014 = []
     
for clf in classifiers:
    clf[1].fit(x_train[imp_cols], y_train)
    y_test_pred= clf[1].predict(x_test[imp_cols])
    y_test_pred_prob= clf[1].predict_proba(x_test[imp_cols])
    accuracy_thresh_014.append(metrics.accuracy_score(y_test, y_test_pred))
    precision_thresh_014.append(metrics.precision_score(y_test, y_test_pred))
    recall_thresh_014.append(metrics.recall_score(y_test, y_test_pred))
    print(clf[0],
          "\n\t Accuracy: ", metrics.accuracy_score(y_test, y_test_pred),
          "\n\t Precision Score: ",metrics.precision_score(y_test, y_test_pred),
          "\n\t Recall Score: ",metrics.recall_score(y_test, y_test_pred))
         


# Plotting the classification metrics for all the base models and models generated from features which have feature importance >=0.03, >=0.014

plt.figure(figsize=(15,10))
plt.plot(model_names , accuracy_base, label = "Accuracy - Base",c = 'blue')
plt.plot(model_names , accuracy_thresh_03, label = "Accuracy - Threshold 0.03", c = 'green')
plt.plot(model_names , accuracy_thresh_014, label = "Accuracy - Threshold 0.014", c= 'red')


plt.plot(model_names , precision_base, label = "Precision - Base",c = 'blue')
plt.plot(model_names , precision_thresh_03, label = "Precision - Threshold 0.03", c = 'green')
plt.plot(model_names , precision_thresh_014, label = "Precision - Threshold 0.014", c= 'red')


plt.plot(model_names , recall_base, label = "Recall - Base",c = 'blue')
plt.plot(model_names , recall_thresh_03, label = "Recall - Threshold 0.03", c = 'green')
plt.plot(model_names , recall_thresh_014, label = "Recall - Threshold 0.014",c= 'red')

plt.legend()
plt.show()


#Our base model with all the features performs as good as the models for which features were removed with a feature importance threshold of 0.03, 0.014.The difference recall and precision metrics along with accuracy are also too small to notice in models where the features are removed.So we stick with the  models with all the features

# However, we choose Decision Tree, Random Forest, Adaboost and XGBoost classifiers for further optimization.

#Validate Model
scoring = 'accuracy'
results=[]
names=[]
for classifier_name, model in classifiers:
    kfold = KFold(n_splits=10, random_state=100)
    cv_results = cross_val_score(model, x_train,y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(classifier_name)
    print(classifier_name,
                      "\n\t CV-Mean:", cv_results.mean(),
                    "\n\t CV-Std. Dev:",  cv_results.std(),"\n")


scoring = 'f1'
results=[]
names=[]
for classifier_name, model in classifiers:
    kfold = KFold(n_splits=10, random_state=100)
    cv_results = cross_val_score(model, x_train,y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(classifier_name)
    print(classifier_name,
                      "\n\t CV-Mean:", cv_results.mean(),
                    "\n\t CV-Std. Dev:",  cv_results.std(),"\n")


#We have better CV mean and Std deviation scores for Decision Tree, Random Forest, Adaboost and XGBoost classifiers than other classifiers. So these models are robust and in addition have good accuracy. I chose f1 as the CV parameter in addition to accuracy because precision and recall as metrics are just as important as accuracy in classification models.

#We however, still need to optimize the hyper-parameters on these models.



#Optimize or Tune Model for better Performance
#Decision Tree


param_grid = {'criterion':['gini','entropy'],
              'max_depth':[2, 3, 4,5, 6, 7, 8, 9],
              'random_state':[100],
              'splitter':['best']}

DT_grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv = 5, verbose=1)




DT_grid.fit(x_train, y_train)



DT_grid.best_params_



model = DT_grid.best_estimator_
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)


model.score(x_test, y_test)

# Generate model evaluation metrics for the Decision Tree Classifier - Hyperparameter Tuned
print("Performance metrics of the model for the Decision Tree Classifier - Hyperparameter Tuned")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))



DT_best = pickle.dumps(DT_grid.best_estimator_)


#Random Forest


param_grid = {'criterion':['gini','entropy'],
              'max_depth':[2, 3, 4, 5, 6, 7, 8, 9],
              'random_state':[100],
              'n_estimators':[200,400,600],
              'n_jobs':[-1], 
              'random_state':[100],
              'verbose': [0]}

RF_grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv = 5, verbose=1)

RF_grid.fit(x_train, y_train)

RF_grid.best_params_

model = RF_grid.best_estimator_
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)

model.score(x_test, y_test)


# Generate model evaluation metrics for the RandomForest Classifier - Hyperparameter Tuned
print("Performance metrics of the model for the RandomForest Classifier - Hyperparameter Tuned")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))


RF_best = pickle.dumps(RF_grid.best_estimator_)


#Adaboost

AdaBoostClassifier()


param_grid = {'algorithm':['SAMME.R'],
              'learning_rate':[0.1, 0.2, 0.3],
              'n_estimators':[200,400,600],
              'random_state':[100]}

AB_grid = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid, cv = 5, verbose=1)


AB_grid.fit(x_train, y_train)


AB_grid.best_params_


model = AB_grid.best_estimator_
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)


model.score(x_test, y_test)


# Generate model evaluation metrics for the AdaBoost Classifier - Hyperparameter Tuned
print("Performance metrics of the model for the AdaBoost Classifier - Hyperparameter Tuned")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))



AB_best = pickle.dumps(AB_grid.best_estimator_)


#XGBoost

param_grid = {'learning_rate':[0.1, 0.2, 0.3],
              'max_depth':[2, 4, 7],
              'n_estimators':[200,400,600],
              'n_jobs':[-1], 
              'objective':['binary:logistic'],
              'random_state':[100],
              'reg_alpha':[0.1, 1, 10], 
              'scale_pos_weight':[1], 
              'silent':[True]}

XGB_grid = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv = 5, verbose=1)


XGB_grid.fit(x_train, y_train)


XGB_grid.best_params_


model = XGB_grid.best_estimator_
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)

model.score(x_test, y_test)

# Generate model evaluation metrics for the XGBOOST - Hyperparameter Tuned
print("Performance metrics of the model for the XGBOOST Classifier - Hyperparameter Tuned")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))

XGB_best = pickle.dumps(XGB_grid.best_estimator_)


# Choose the model for deployment

#We choose the hyperparameter tuned models because they have the better accuracy score even though all other average metrics(from classification report) are the same.


best_classifiers = [
                    ("Decision Tree - ",  DT_grid.best_estimator_),
                    ("Random Forest - ",  RF_grid.best_estimator_),
                    ("AdaBoost - ",  AB_grid.best_estimator_),
                    ("XGBoost - ", XGB_grid.best_estimator_)]
accuracy_best = []
precision_best = []
recall_best = []
best_model_names = [i[0] for i in best_classifiers]

for clf in best_classifiers:
    clf[1].fit(x_train[imp_cols], y_train)
    y_test_pred= clf[1].predict(x_test[imp_cols])
    accuracy_best.append(metrics.accuracy_score(y_test, y_test_pred))
    precision_best.append(metrics.precision_score(y_test, y_test_pred))
    recall_best.append(metrics.recall_score(y_test, y_test_pred)) 
    print(clf[0])
    print("-"*100)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
    print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
    print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
    print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
    print()
    print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
    print()
    print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))



plt.figure(figsize=(10,5))
plt.plot(best_model_names , accuracy_best, label = "Accuracy - Best",c = 'blue')
plt.plot(best_model_names , precision_best, label = "Precision - Best", c = 'green')
plt.plot(best_model_names , recall_best, label = "Recall - Best",c= 'red')
plt.legend()
plt.show()


#Clearly XGBoost offers better Accuracy, Precision and Recall when compared to the other Classifiers. Therefore, we choose it as our model. The following are the hyper-parameters of the model:


XGB_grid.best_estimator_



# Saving the the chosen model in the pickle object
chosen_model = pickle.dumps(XGB_grid.best_estimator_)


#To Load:
pickle.loads(chosen_model)


