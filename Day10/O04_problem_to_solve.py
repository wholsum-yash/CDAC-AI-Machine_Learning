'''
The Challenge:
Imagine you are a cricket data analyst.
Your organization has collected performance statistics for various international cricket players (including matches played, total runs, wickets taken, and catches). 
However, sports scouts are tired of manually categorizing players.
'''

'''
Goal: Categorize cricket players based on their performance statistics and
      accurately predict their primary playing role: Batsman, Bowler, or All-Rounder.
'''

'''
Approach:
1. Clean and prepare raw data.
2. Feature Scaling
3. Train a Model : use 80% of the data for training
4. Evaluate the Model : use the remaining 20% of the data for testing
5. Predict
6. visualize
'''

raw_csv_data = """matches,runs,wickets,catches,dominant_hand,role
150,6500,4,45,Right,Batsman
120,4800,10,38,Left,Batsman
200,8500,12,60,Right,Batsman
90,3200,2,25,Right,Batsman
180,7200,8,50,Left,Batsman
110,4200,5,30,Right,Batsman
75,2100,0,18,Right,Batsman
140,5900,15,42,Right,Batsman
130,250,180,20,Right,Bowler
160,400,220,15,Right,Bowler
95,150,130,12,Left,Bowler
210,600,310,25,Right,Bowler
85,100,110,8,Right,Bowler
175,500,260,18,Left,Bowler
105,300,145,10,Right,Bowler
190,800,290,22,Right,Bowler
140,3200,120,35,Right,Allrounder
165,4100,150,40,Left,Allrounder
115,2500,95,28,Right,Allrounder
195,5100,185,55,Right,Allrounder
80,1800,70,20,Right,Allrounder
135,3100,110,32,Left,Allrounder
170,4500,140,48,Right,Allrounder
100,2200,80,22,Right,Allrounder
160,3900,135,36,Right,Allrounder
"""

# Convert the raw CSV data into a pandas DataFrame
import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.tree import plot_tree


# step 1 : load the data series into a DataFrame
df = pd.read_csv(StringIO(raw_csv_data))

# Display the first few rows of the DataFrame to verify the data
print(df.head())
print("The Target feature is 'role'\n")
print(df['role'].value_counts())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# step 2 : Data Preprocessing (encoding and squeezing/normalization/scaling)
 # encoder for role

role_encoder = LabelEncoder()
df['role_encoded'] = role_encoder.fit_transform(df['role'])

# encoder for dominant_hand
hand_encoder = LabelEncoder()
df['hand_encoded'] = hand_encoder.fit_transform(df['dominant_hand'])

# print("Encoder mapping for role:")
# for idx,label in enumerate(role_encoder.classes_):
#     print(f"{label} -> {idx}")

# print("Hand Encoder mapping:",list(zip(hand_encoder.classes_, range(len(hand_encoder.classes_)))))

# select feature columns (X) and target column (y)
feature_columns = ['matches', 'runs', 'wickets', 'catches', 'hand_encoded']
X = df[feature_columns]
y = df['role_encoded']

print(X[:5])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
print(X_scaled_df.head())

# step 3 : Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42,stratify=y)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("=" * 60)

# step 4 : Train a Model

rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, criterion = "gini")
rf_clf.fit(X_train, y_train)

# step 5 : Evaluate the Model
y_pred = rf_clf.predict(X_test)

# calculate the accuracy:
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred,target_names=role_encoder.classes_ ))

# predict for new player
new_player_raw = np.array([[130,3500,105,30,1]])  # 
new_player_scaled = scaler.transform(new_player_raw)

# redict role
predicted_code = rf_clf.predict(new_player_scaled)
predicted_role = role_encoder.inverse_transform(predicted_code)
print(f"Predicted role for the new player: {predicted_role[0]}")    

# step 6 : Feature Importance
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Feature Importance Bar Chart
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

axes[0].bar(range(len(feature_columns)), importances[indices], color='skyblue', edgecolor='black')
axes[0].set_xticks(range(len(feature_columns)))
axes[0].set_xticklabels([feature_columns[i] for i in indices], rotation=30)
axes[0].set_title("Random Forest - Feature Importances", fontsize=14)
axes[0].set_ylabel("Importance Score")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: Plot one of the Decision Trees inside the Forest
plot_tree(
    rf_clf.estimators_[0],
    feature_names=feature_columns,
    class_names=role_encoder.classes_,
    filled=True,
    rounded=True,
    ax=axes[1]
)
axes[1].set_title("Structure of Tree #1 inside the Random Forest", fontsize=14)

plt.tight_layout()
plt.show()

'''
Chronological Step-by-Step Workflow (The Roadmap)
Step 1: Load and Inspect the Raw Data
What to do: Read the raw CSV file into a Pandas DataFrame.

Why: Inspect the data types (text vs. numbers) and check for missing values or weird formatting.

Step 2: Data Preprocessing & Cleaning (The "Last Bit" Cleaning)
What to do:

Convert text categories like dominant_hand ("Right"/"Left") into numbers using LabelEncoder.

Convert the target labels (role: "Batsman", "Bowler", "All-Rounder") into numeric codes so the machine learning model can process them.

Why: Machine learning algorithms only understand numbers, not words.

Step 3: Feature Scaling (Squeezing the Values)
What to do: Use StandardScaler to transform features like runs (which can be 10,000+) and wickets (which might be 150) onto a uniform scale (mean = 0, variance = 1).

Why: It prevents large-number columns from dominating smaller-number columns during distance or impurity calculations.

Step 4: Split Data into Training and Testing Sets
What to do: Separate your features (X = matches, runs, wickets, etc.) from your target (y = role), and use train_test_split() to reserve 80% for training and 20% for testing.

Why: You must test your model on players it has never seen before to check if it has genuinely learned patterns rather than just memorizing the answers.

Step 5: Train the Random Forest Classifier
What to do: Initialize RandomForestClassifier(n_estimators=50) and .fit() it using your training data.

Why: The algorithm will build 50 different decision trees, introducing randomness into the data and features each time to ensure diversity.

Step 6: Evaluate Model Performance
What to do: Predict the roles for the test set and generate an accuracy_score and classification_report.

Why: To measure how well your "forest of trees" performs on unseen data.

Step 7: Make a Prediction on a New Player
What to do: Feed a custom player profile (e.g., someone with 150 matches, 4,500 runs, and 90 wickets) into your trained pipeline.

Why: To see the model in action answering real-world scouting questions.

Step 8: Visualize the Insights
What to do:

Plot a bar chart of Feature Importances to see whether runs or wickets drove the model's decisions.

Plot one of the individual decision trees using plot_tree().

Why: To make the "black box" of machine learning transparent and easy for your audience to interpret.
'''