import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Input X: Study hours (1 to 8 hours)
X = pd.DataFrame({"study_hours": [1, 2, 3, 4, 5, 6, 7, 8]})

# Target y: Exam score out of 100
y = [45, 65, 80, 90, 92, 88, 75, 55]

print("--- Beginner Dataset ---")
for h, s in zip(X["study_hours"], y):
    print(f"Study Hours: {h} hr  -->  Exam Score: {s}/100")
print("="*60)
# poly_transform = PolynomialFeatures(degree=2, include_bias=False)  # Example for degree 2
# # include_bias=False means we do not include the bias (intercept) term in the polynomial features
# # math formula for degree 2 polynomial features: [x, x^2] for each input x
# X_poly_sample = poly_transform.fit_transform(X)
# print(X_poly_sample)
# print("\n--- Polynomial Features (Degree 2) ---")
# for original, transformed in zip(X["study_hours"], X_poly_sample):
#     source, target = transformed # unpack the original and squared term
#     print(f"Original: {original}  -->  Transformed: {target}")
# print("="*60)
model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
model.fit(X, y)

new_student = pd.DataFrame({"study_hours": [3.5,2.5,7.5]})
predicted_score = model.predict(new_student)
for hours, score in zip(new_student["study_hours"], predicted_score):
    print(f"Predicted Exam Score for {hours} study hours: {score}/100")
print("="*60)
# merge both df
df= X.copy()
df["exam_score"] = y
print(df)
df = pd.concat([df, new_student.assign(exam_score=predicted_score)], ignore_index=True)
print("="*60)
print(df)

# merge new_student predictions

print("="*60)
# # visualize the merged dataset using scatter plot
import matplotlib.pyplot as plt
# plt.scatter(df["study_hours"], df["exam_score"], color="blue", label="Actual")
# plt.scatter(new_student["study_hours"], predicted_score, color="red", label="Predicted")
# plt.xlabel("Study Hours")
# plt.ylabel("Exam Score")
# plt.title("Polynomial Regression Predictions")
# plt.legend()
# plt.show()


x_smooth = pd.DataFrame({"study_hours": np.linspace(df["study_hours"].min(), df["study_hours"].max(), 300)})
y_smooth = model.predict(x_smooth)
# 2. Your original plot elements + the smooth line
plt.scatter(df["study_hours"], df["exam_score"], color="blue", label="Actual")
plt.scatter(new_student["study_hours"], predicted_score, color="red", label="Predicted")
# Plot the smooth bell line
plt.plot(x_smooth["study_hours"], y_smooth, color="green", linewidth=2, label="Fitted Curve (Bell)")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Polynomial Regression Predictions")
plt.legend()
plt.show()

