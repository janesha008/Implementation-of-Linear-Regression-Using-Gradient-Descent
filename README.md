# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize slope (m), intercept (b), learning rate, and number of iterations.
2. Predict output using:
       y_pred= mx+b
3. Compute gradients and update m and b using gradient descent formula.
4. Repeat steps 2–3 for all iterations to get the best fit line, and plot the data points and regression line.

## Program:
```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Startup.csv")
x=df['R&D Spend'].values
y=df['Profit'].values

x=(x-np.mean(x))/np.std(x)

m=0
b=0
learning_rate=0.01
epochs=1000
n=len(x)

for i in range(epochs):
    y_pred= m*x+b
    dm=((-2/n)*np.sum(x*(y-y_pred)))
    db=((-2/n)*np.sum(y-y_pred))
    m=m-learning_rate*dm
    b=b-learning_rate*db
print(f'Slope(m): {m}')
print(f'Intercept(b): {b}')

y_pred=m*x+b

plt.scatter(x,y,label='Data Points')
plt.plot(x,y_pred,label='Fit Line')
plt.title("Gradient Descent")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.legend()
plt.show()
```

## Output:

<img width="906" height="715" alt="image" src="https://github.com/user-attachments/assets/c0a15ee0-801b-42ff-a470-8938f885b061" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
