# Simple & Multiple Linear Regression

Linear Regression is the fundamental baseline algorithm for predicting continuous target outcomes. It models the relationship between a dependent target variable ($y$) and one or more independent predictor features ($X$) by fitting a linear hypothesis function.

## Learning Objectives


By completing this submodule, you will be able to:
1. Formulate the mathematical hypothesis for **Simple Linear Regression** ($1$ feature) and **Multiple Linear Regression** ($p$ features).
2. Understand the mechanics of **Ordinary Least Squares (OLS)** optimization and the **Mean Squared Error (MSE)** loss function.
3. Check and validate key statistical assumptions (Linearity, Independence, Homoscedasticity, Normality of Residuals, and No Multicollinearity).
4. Evaluate continuous model performance using **MAE**, **MSE**, **RMSE**, **$R^2$**, and **Adjusted $R^2$**.
5. Build and deploy a modular salary prediction pipeline using `scikit-learn`.

---

## Mathematical Foundations


### 1. Simple Linear Regression
$$\hat{y} = \beta_0 + \beta_1 x$$

* $\hat{y}$: Predicted target value
* $\beta_0$: Intercept (value of $y$ when $x = 0$)
* $\beta_1$: Slope coefficient (change in $y$ per unit change in $x$)

### 2. Multiple Linear Regression
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

In matrix vector notation:
$$\hat{y} = X \boldsymbol{\beta}$$

### 3. OLS Loss Function (Mean Squared Error)
$$J(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - (\beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}) \right)^2$$

---

## Evaluation Metrics Quick Reference


| Metric | Formula | Key Insight |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\frac{1}{n} \sum \Vert{}y_i - \hat{y}_i\Vert{}$ | Interpretable in target units; robust to extreme outliers. |
| **MSE** (Mean Squared Error) | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Heavily penalizes large estimation errors. |
| **RMSE** (Root Mean Squared Error) | $\sqrt{\text{MSE}}$ | Returns error penalty back to target unit scale. |
| **$R^2$** (Coefficient of Determination) | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | Percentage of variance explained by the model ($0 \le R^2 \le 1$). |
| **Adjusted $R^2$** | $1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$ | Penalizes unnecessary predictor features that do not improve fit. |
