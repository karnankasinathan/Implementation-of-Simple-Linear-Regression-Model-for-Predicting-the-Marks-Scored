# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: karnan k
RegisterNumber:  212222230062
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

X = df.iloc[:,:-1].values
X

Y = df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
# df.head()
![Screenshot 2023-09-06 224725](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/a6f87d7f-89b6-4481-8a55-9454e7be6621)
# df.tail()

![Screenshot 2023-09-06 224733](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/bb9fbab9-1588-4d4c-a1c0-d3a050559f19)
# Array value of X

![Screenshot 2023-09-06 224742](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/6fe93516-1586-446a-b32f-c55bec1d3383)
# Array value of Y
![Screenshot 2023-09-06 224751](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/7547db07-5b37-4ac9-b4ca-96a0381a096d)
# Values of Y prediction
![Screenshot 2023-09-06 224802](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/70473bb4-b973-40f1-a14a-156ff7cd6845)
# Array values of Y test
![Screenshot 2023-09-06 224808](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/d0790d80-b226-4e08-9f04-8b98d4ec4ca3)

# Training Set Graph
![Screenshot 2023-09-06 224817](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/54c2714a-fa6b-450a-9505-a30e3d385c47)
# Test Set Graph
![Screenshot 2023-09-06 224829](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/e6587674-5c47-49e8-a8fd-e921e83ba37d)
# Values of MSE, MAE and RMSE
![Screenshot 2023-09-06 224836](https://github.com/karnankasinathan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787064/92d2efc6-5379-4e8a-9b17-32c7cd6dabdc)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
