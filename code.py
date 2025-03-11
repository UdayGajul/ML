#Assigment 4

#Question 1 ------------------------K-nearest Neighbour Classifier on Adult dataset----------------------

import pandas as pd

adult_X=pd.read_csv('adult/adult.data',header=None)
# print(adult_X)

adult_y=pd.read_csv('adult/adult.test',header=None)
# print(adult_y)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in adult_X.columns:
    adult_X[col]=le.fit_transform(adult_X[col])

X=adult_X.iloc[:,:-1]
# print(X)

y=adult_X.iloc[:,-1:]

for col in adult_y.columns:
    adult_y[col]=le.fit_transform(adult_y[col])

# print(adult_y)

adult_X_test=adult_y.iloc[:,:-1]
# print(adult_X_test)

adult_y_test=adult_y.iloc[:,-1:]
# print(adult_y_test)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=100)
knn.fit(X,y)
y_pred=knn.predict(adult_X_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_pred,adult_y_test)
print("Accuracy",acc*100)


#Question 2  ------------------------Decision Tree Classifier on Recruitment dataset----------------------

import pandas as pd

rec=pd.read_csv('recruitment.csv')
# print(rec)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in rec.columns:
    rec[col]=le.fit_transform(rec[col])

X=rec.iloc[:,:4]

y=rec.iloc[:,4:]

# print(rec)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier, plot_tree

dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_pred,y_test)
print("Accuracy to getting Job",acc*100)

import matplotlib.pyplot as plt

plt.figure(figsize=(25,10),dpi=54,facecolor='lightgray')
plot_tree(dtc,feature_names=X.columns,rounded=True,class_names=['Yes','No'],filled=True)
plt.show()

y_new=[[2,1,0,0]]
# print(y_new)

y_pred_new=dtc.predict(y_new)
# print(y_pred_new)

# print("For data [2,1,0,0] i.e. ['CGPA','Communication','Aptitude','Programming Skill'] the model predicting 1 value i.e. Job offered? is Yes")


#Question 3  ------------------------Random Forest Classifier on Parkinsons dataset----------------------

import pandas as pd

parkinsons=pd.read_csv('parkinsons/parkinsons.data')
print(parkinsons)

X=parkinsons.drop(['status'],axis=1)
print(X)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

X['name']=le.fit_transform(X['name'])
print(X)

y=parkinsons['status']
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=50)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

acc=accuracy_score(y_test,y_pred)
print("Accuracy",acc*100)

cm=confusion_matrix(y_test,y_pred)
print(cm)



#Question 4 ------------------------Support Vector Machine Classifier on Bank dataset----------------------

import pandas as pd

bank=pd.read_csv('bank-full.csv')
# print(bank)

X=bank.drop(['y'],axis=1)
# print(X)

y=bank['y']
# print(y)

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
ohe=OneHotEncoder()

df_X=ohe.fit_transform(X).toarray()
X=pd.DataFrame(df_X,columns=ohe.get_feature_names_out())
# print(X)

le=LabelEncoder()

df_y=le.fit_transform(y)
y=pd.DataFrame(df_y,columns=['y'])
# print(y)

from sklearn.feature_selection import SelectKBest,chi2

df_X=pd.DataFrame(X,columns=X.columns)
# print(df_X)
selector=SelectKBest(k=2,score_func=chi2)
imp_X=selector.fit_transform(df_X,y)
X=pd.DataFrame(imp_X,columns=selector.get_feature_names_out())
# print(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC

svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_pred,y_test)
print("Accuracy for y",acc*100)


#Question 5 ------------------------Naive Bayes Classifier on Match dataset----------------------

import pandas as pd

match=pd.read_excel('match_data.xlsx')
# print(match)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in match.columns:
    match[col]=le.fit_transform(match[col])

print(match)

X=match.iloc[:,:4]

y=match.iloc[:,4:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import CategoricalNB

cnb=CategoricalNB()
cnb.fit(X_train,y_train)
y_pred=cnb.predict(X_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_pred,y_test)
print("Accuracy to win match",acc*100)

y_new=[[0,3,0,1]]
y_pred_new=cnb.predict(y_new)
print(y_pred_new)
print("For data [0,3,0,1] i.e. ['Weather Condition','Wins in last 3 matches','Humidity','Win toss'] the model predicting 1 value i.e. Won match? is Yes")



# Assignment 5 ____________________________________________________________________________________________


#Question 1 ------------------------Linear Regression on Height and Age dataset----------------------

import pandas as pd

rushi=pd.read_excel('Datasets/height_age.xlsx')
print(rushi)

age=rushi.drop('Height (in inches)',axis=1)
height=rushi['Height (in inches)']

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(age,height)

y_pred_new=[[16]]
y_new=lr.predict(y_pred_new)
print(f"The height of rushi when he was 16 years old is: {y_new.mean()} inches")

import matplotlib.pyplot as plt

plt.scatter(age,height,label='Data Points',color='red')
plt.scatter(16,y_new.mean(),label='Predicted Value',color='blue')
plt.xlabel('Age')
plt.ylabel('Height')
plt.legend()
plt.show()


#Question 2 ------------------------Multiple Regression on Auto mpg dataset----------------------

import pandas as pd

mpg=pd.read_csv('auto_mpg/auto-mpg_orignal.data',header=None,delim_whitespace=True)  
mpg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
# print(mpg)

X=mpg.drop(['mpg'],axis=1)
print(X)

y=mpg['mpg']
# print(y)

y=y.fillna(y.mean())
print(y)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X['car name']=le.fit_transform(X['car name'])
print(X)

X=X.fillna(X.mean())

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)
y_pred=lr.predict(X)
print("Predicted Values",y_pred.mean())


#Question 3 ------------------------Polynomial Regression on Height and Age dataset----------------------

import pandas as pd

rushi=pd.read_excel('Datasets/height_age.xlsx')
X=rushi['Age']
y=rushi['Height (in inches)']

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(rushi)
# print(X_poly)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_poly,y)
y_pred=lr.predict(X_poly)
print(y_pred.mean())

import matplotlib.pyplot as plt
plt.scatter(X,y,label='Data Points')
plt.scatter(X,y_pred,label='Predicted Points',color='red',marker='X')
plt.plot(X,y_pred,color='green',label='Curved Line')
plt.xlabel('Age')
plt.ylabel('Height')
plt.legend()
plt.show()


#Question 4 ------------------------Logistic Regression on Diabetes dataset----------------------

import pandas as pd

diabetes=pd.read_csv('diabetes.csv')
print(diabetes)

X=diabetes.drop('Outcome',axis=1)
# print(X)

y=diabetes.iloc[:,8:]
print(y)

# # y=pd.Series(diabetes['Outcome'])
# # print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
lgst=LogisticRegression()
lgst.fit(X_train,y_train)
y_pred=lgst.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

cm=confusion_matrix(y_pred,y_test)
print(cm)

acc=accuracy_score(y_test,y_pred)
print(acc)

import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(cm,annot=True,xticklabels=y['Outcome'],yticklabels=X.columns)
plt.show()



# Assignment 6 ____________________________________________________________________________________________

# Q4 ------------------------DBSCAN on xy/given dataset----------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

d = pd.read_csv('xy.csv')

db = DBSCAN(eps=2, min_samples=6)
db.fit(d)

print(db.labels_)

plt.scatter(d['x'], d['y'], c=db.labels_, cmap='viridis', marker='o')

plt.show()
