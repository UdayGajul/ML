# 1)Categorical NB
import pandas as pd
weather_nom=pd.read_excel('weather_nominal.xlsx')
print(weather_nom)

# Encode categorical variables
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()

# weather_nom['outlook']=le.fit_transform(weather_nom['outlook'])
# weather_nom['temperature']=le.fit_transform(weather_nom['temperature'])
# weather_nom['humidity']=le.fit_transform(weather_nom['humidity'])
# weather_nom['windy']=le.fit_transform(weather_nom['windy'])

for col in weather_nom.columns:
    weather_nom[col]=le.fit_transform(weather_nom[col])
print(weather_nom)

# Feature matrix and target variable
x = weather_nom.iloc[:, :-1] # Select all columns except the last one
y=weather_nom.iloc[:,4] #or
# y = weather_nom.iloc[:, -1]   # Select the last column as target

# Split the dataset
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_test=train_test_split(x,y,test_size=0.4)

# Train Naive Bayes classifier
from sklearn.naive_bayes import CategoricalNB
cnb=CategoricalNB()
cnb.fit(x_train,y_train)
y_pred=cnb.predict(x_text)


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print('Accuracy:',acc*100)

# **********************************************************************************************************
# 2)Gaussian NB
from sklearn.datasets import load_iris
x,y=load_iris(return_X_y=True)

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_test=train_test_split(x,y,test_size=0.4)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_text)

# Predict class probabilities
import numpy as np
np.set_printoptions(suppress=True,precision=3)
y_pred=gnb.predict_proba(x_text)
print(y_pred)

# **********************************************************************************************************
# # #3) Multinominal NB
import pandas as pd
feedback=pd.read_excel('feedback.xlsx')
print(feedback)

# Convert text to numerical feature vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
data = cv.fit_transform(feedback['Opinion']).toarray()
df_data = pd.DataFrame(data, columns=cv.get_feature_names_out())

# Define features and labels
x = df_data
y = feedback['Class']

# Train Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(x,y)

# Create new data for prediction
new_data=pd.DataFrame([[0,1,0,0,1,0,1]],columns=cv.get_feature_names_out())

# Predict
y_pred=mnb.predict(new_data)
print("Predicted Class:", y_pred)