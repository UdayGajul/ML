#1)Simple Linear Regression
import pandas as pd
linear=pd.read_excel('linear.xlsx')
print(linear)

independent=linear.drop('Dependent',axis=1)
dependent=linear['Dependent']

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(independent,dependent)
pred_dependent=lr.predict(independent)

from sklearn.metrics import mean_squared_error, r2_score
mse=mean_squared_error(dependent,pred_dependent)
print('Mean Squared Error:',mse)
r2=r2_score(dependent,pred_dependent)
print('R2 Score:',r2)

import matplotlib.pyplot as plt
plt.scatter(independent,dependent,color='blue',label='Actual Data')
plt.plot(independent,pred_dependent,color='red',marker='X',label='Predicted Data')
#plt.plot(independent,pred_dependent,color='green',label='Regression Line')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()

# *******************************************************************************************************************

#2)Multiple Linear Regression
import pandas as pd
linear=pd.read_excel('multi.xlsx')
print(linear)


independent=linear.drop('Dependent',axis=1)
dependent=linear['Dependent']

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(independent, dependent)
pred_dependent = mlr.predict(independent)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(dependent, pred_dependent)
print('Mean Squared Error (Multiple):', mse)
r2 = r2_score(dependent, pred_dependent)
print('R2 Score (Multiple):', r2)

plt.plot(range(len(dependent)), dependent, color='blue', label='Actual Data', marker='o')
plt.plot(range(len(pred_dependent)), pred_dependent, color='red', label='Predicted Data', marker='x')
plt.xlabel('Data Points')
plt.ylabel('Dependent Variable')
plt.legend()
plt.title('Multiple Linear Regression')
plt.show()

# ************************************************************************************************************

#3)Polynomial Regression
import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_excel('poly.xlsx')
print(data)

independent = data.drop('Dependent', axis=1)
dependent = data['Dependent']

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
independent_poly = poly.fit_transform(independent)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(independent_poly, dependent)
pred_dependent = lr.predict(independent_poly)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(dependent, pred_dependent)
print('Mean Squared Error (Polynomial):', mse)
r2 = r2_score(dependent, pred_dependent)
print('R2 Score (Polynomial):', r2)

import matplotlib.pyplot as plt
plt.scatter(range(len(dependent)), dependent, color='blue', label='Actual Data')
plt.plot(range(len(pred_dependent)), pred_dependent, color='red', marker='X', label='Predicted Data')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')
plt.legend()
plt.title('Polynomial Regression')
plt.show()

# ****************************************************************************************************************

#4)Logistic Regression
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
x=pd.DataFrame(cancer.data,columns=cancer.feature_names)
y=pd.Series(cancer.target)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report
print('Accuracy:',accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test,y_pred))
