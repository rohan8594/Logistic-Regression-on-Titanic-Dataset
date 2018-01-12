
# coding: utf-8

# Logistic Regression on the Titanic Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

train = pd.read_csv('titanic_train.csv')

train.head()

train.isnull().head()



sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# If we glimpse at the data, we're missing some age information, we're missing a lot of cabin info and we're missing one row of embarked.
# We'll come back to this problem of missing data a little later. But before that lets focus on some exploratory data analysis on a visual level.

sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train)

sns.countplot(x='Survived',data=train,hue='Sex',palette='RdBu_r')


# Clearly there's a trend here. It looks like people that did not survive were much more likely to be men. While those who survived were twice as likely to be female.

sns.countplot(x='Survived',data=train,hue='Pclass')


# Also it looks like the people who did not survive were overwhelmingly part of 3rd class. People that did survive were from the higher classes.

# Now lets try and understand the age of the onboard passengers.

sns.distplot(train['Age'].dropna(),bins=30,kde=False)


# There seems to be an interesting bi-modal distribution where there are quite a few young passengers between age 0 and 10. Then the average age tends to be around 20-30.

sns.countplot(x='SibSp',data=train)

train['Fare'].hist(bins=40,figsize=(10,4))


# Cleaning Data

# As we saw earlier there are few columns that are missing some data. We need to clean our dataset before we begin to train our logistic regression model. Lets first try and fill in the missing age values. I'm going to do this by filling in the missing age with the mean age of the passenger class that the passenger belongs to.

plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)

train.groupby('Pclass').mean()['Age'].round()

mean_class1 = train.groupby('Pclass').mean()['Age'].round().loc[1]
mean_class2 = train.groupby('Pclass').mean()['Age'].round().loc[2]
mean_class3 = train.groupby('Pclass').mean()['Age'].round().loc[3]

train.loc[train['Pclass']==1,'Age'] = train.loc[train['Pclass']==1,'Age'].fillna(value=mean_class1)
train.loc[train['Pclass']==2,'Age'] = train.loc[train['Pclass']==2,'Age'].fillna(value=mean_class2)
train.loc[train['Pclass']==3,'Age'] = train.loc[train['Pclass']==3,'Age'].fillna(value=mean_class3)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# I'm going to just drop the cabin column since there's too much missing information.

train.drop('Cabin',axis=1,inplace=True)

train.dropna(inplace=True) # dropping the 1 missing value in Embarked column


# I will now convert some of the categorical features in the dataset into dummy variables that our machine learning model can accept.

sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

train = pd.concat([train,sex,embark],axis=1)

train.head(2)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train.drop('PassengerId',axis=1,inplace=True)

train.head()


# Now lets perform similar data cleaning on the test data.

test = pd.read_csv('titanic_test.csv')

test.loc[test['Pclass']==1,'Age'] = test.loc[test['Pclass']==1,'Age'].fillna(value=mean_class1)
test.loc[test['Pclass']==2,'Age'] = test.loc[test['Pclass']==2,'Age'].fillna(value=mean_class2)
test.loc[test['Pclass']==3,'Age'] = test.loc[test['Pclass']==3,'Age'].fillna(value=mean_class3)


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')



test.drop('Cabin',axis=1,inplace=True)

test.dropna(inplace=True)

sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)

test = pd.concat([test,sex,embark],axis=1)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test.head()


# Train and build Classifier

X = train.drop('Survived',axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

logmodel.score(X_train,y_train)

logmodel.score(X_test,y_test)


# Making Predictions

test_x = test.drop('PassengerId',axis=1)

predictions = logmodel.predict(test_x)

final_prediction = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

final_prediction.head()

