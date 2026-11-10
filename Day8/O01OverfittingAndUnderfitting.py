'''
What is Polinomial Regression?
Polynomial Regression is a type of regression analysis in which the relationship between the
independent variable x and the dependent variable y is modeled as an nth degree polynomial. 
It is used when the data shows a curvilinear relationship rather than a linear one.

Example of monomial (single term) polynomial regression:
5x^2
Example of binomial (two terms) polynomial regression:
3x^2 + 4x
Example of trinomial (three terms) polynomial regression:
3x^2 + 5x + 7
Example of quartic (four terms) polynomial regression:
ax^2 + bx + c
polynomial regression can have any number of terms, depending on the degree of the polynomial.


Degree of a polynomial:
The degree of a polynomial is the highest power of the independent variable x in the polynomial.
For example:
- In 5x^2, the degree is 2.
- In 3x^2 + 4x, the degree is 2.
- In 3x^2 + 5x + 7, the degree is 2.
- In ax^2 + bx + c, the degree is 2.

degree 1 : linear  : straight line : y = b0 + b1x   : y = mx + c
degree 2 : quadratic  : parabola : y = b0 + b1x + b2x^2 : y = ax^2 + bx + c
degree 3 : cubic      : cubic curve : y = b0 + b1x + b2x^2 + b3x^3 : y = ax^3 + bx^2 + cx + d
degree 4 : quartic    : quartic curve : y = b0 + b1x + b2x^2 + b3x^3 + b4x^4

degree 1 : no bend : straight line      : example [expereience v/s salary]
degree 2 : 1 bend (U curve or Arch)     : example [rpm v/s fuel efficiency, crop yield vs fertilizer]
degree 3 : 2 bends (S curve)            : example [new technology adoption v/s time]
degree 4 : 3 bends (W curve)            : example [complex economic models]
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

X = pd.DataFrame({"study_hours": [1, 2,  4, 6, 7]})
y =np.array([50, 60, 70, 80, 60])

print(pd.DataFrame({"Hours": X["study_hours"], "score": y}))
print("="*60)

# Compare Degrees 1, 2,  4
degrees = [1, 2, 4]
#model = make_pipeline(PolynomialFeatures(degree =1 ), LinearRegression()) ; model.fit(X, y)
# maths used for degree 1 for X and y is simple linear regression: y = b0 + b1*x
#model = make_pipeline(PolynomialFeatures(degree =2 ), LinearRegression()) ; model.fit(X, y)
# maths used for degree 2 for X and y is quadratic regression: y = b0 + b1*x + b2*x^2
#model = make_pipeline(PolynomialFeatures(degree =4 ), LinearRegression()) ; model.fit(X, y)
# maths used for degree 4 for X and y is quartic regression: y = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4


for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"Degree {degree} RMSE: {rmse:.2f}")

# identify the best degree based on RMSE
# underfitting occurs when the model is too simple to capture the underlying pattern in the data
# overfitting occurs when the model is too complex and captures noise in the data as well
# typically, the best degree is a balance between underfitting and overfitting
# if rmse is zero means memorizing the training data




