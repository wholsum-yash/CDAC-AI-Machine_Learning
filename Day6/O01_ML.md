Machine Learning : is broadly categorized into several core paradigms based on how algorithms learn from data and interact with their environment.

### 1. Supervised Learning
The model is trained on a **labeled dataset**, meaning every input comes with the correct output. The algorithm learns a mapping function from inputs to outputs.
* **Key Tasks:** Classification (predicting discrete classes) and Regression (predicting continuous values).
* **Common Examples:** Spam detection, house price prediction, image recognition.

### 2. Unsupervised Learning
The model works with **unlabeled data**, exploring the data structure on its own to find hidden patterns, groupings, or representations.
* **Key Tasks:**  Clustering (grouping similar data points), Dimensionality Reduction (compressing features while retaining variance), and Association Rule Learning.
* **Common Examples:** Customer segmentation, anomaly detection, PCA/t-SNE for visualization.

### 3. Semi-Supervised Learning
A hybrid approach that uses a **small amount of labeled data** combined with a **large volume of unlabeled data**. This is especially useful when labeling data is expensive or time-consuming.
* **How it Works:** The model uses the labeled data to learn basic patterns and leverages the vast amount of unlabeled data to understand the underlying data manifold.
* **Common Examples:** web page classification, Speech recognition, medical imaging analysis.

### 4. Self-Supervised Learning
A specialized form of unsupervised learning where the model generates its own **pseudo-labels** from the input data itself, typically by masking parts of the input and predicting the missing pieces.  This is the foundation of modern Large Language Models (LLMs) and foundational vision models.
* **How it Works:** "Next-token prediction" or "masked image modeling."
* **Common Examples:** BERT, GPT series models, contrastive learning frameworks (like SimCLR). BERT = Bidirectional Encoder Representaion from trasformer


### 5. Reinforcement Learning (RL)
An agent learns to make decisions by **interacting with an environment** to achieve a goal. It takes actions, observes the state of the environment, and receives rewards or penalties, optimizing its policy through trial and error.
* **Key Components:** Agent, Environment, State, Action, Reward.
* **Common Examples:** Autonomous driving,game-playing AI (AlphaGo), robotics control.

### 6. Transfer Learning:
    Taking a pre-trained model (trained on a massive dataset) and fine-tuning it on a smaller, specific target dataset.

### 7. Active Learning:
    The algorithm proactively queries a human user (or oracle) to label the most informative data points, minimizing the labeling effort.



### 1. Supervised Learning
* **Classification:** : Target column requires categorical values (binary strings like `"Yes"`/`"No"`, or integer class IDs like `0, 1, 2`).
* *Example target column:* `["Yes", "No", "Yes", "No"]` or `[0, 1, 2, 0]`
* *APIs:* `sklearn.linear_model.LogisticRegression`, `sklearn.svm.SVC`, `xgboost.XGBClassifier`
    * **binary classification**
        * **Logistic Regression:**  `sklearn.linear_model.LogisticRegression`
        * **Linear Support Vector Machine:** `sklearn.svm.LinearSVC` or `sklearn.svm.SVC(kernel='linear')`
        * **Binary Decision Tree:** `sklearn.tree.DecisionTreeClassifier`
    * **multi-class classification**
        * **Multinomial Logistic Regression:** `sklearn.linear_model.LogisticRegression(multi_class='multinomial')`
        * **Multi-class SVM / Random Forest:** `sklearn.svm.SVC`, `sklearn.ensemble.RandomForestClassifier`
        * **Gradient Boosting (Multi-class):** `xgboost.XGBClassifier`, `lightgbm.LGBMClassifier`
    * **multi-label classification**
        * **Classifier Chains / Binary Relevance:** `sklearn.multioutput.ClassifierChain`, `sklearn.multiclass.OneVsRestClassifier`
    * **ordinal classification / ranking**
        * **Ordinal Regression:** `mord` library (e.g., `mord.LogisticIT`) or custom threshold wrappers
        * **Learning-to-Rank:** `xgboost.XGBRanker` or `lightgbm.LGBMRanker`

* **Regression:** :  Target column requires continuous numeric values (floats or integers representing quantities, prices, or measurements).

* **Ordinal Regression & Ranking:**
    Target column requires ordered categorical values or rank integers (e.g., star ratings 1 to 5, low/medium/high tiers).
* **Survival Analysis:**
    Target columns require a duration column (continuous time-to-event float) and an event indicator column (binary boolean or `0`/`1` flag).

### 2. Unsupervised Learning
### 3. Semi-Supervised Learning
### 4. Self-Supervised Learning
### 5. Reinforcement Learning (RL)
### 6. Transfer Learning:
### 7. Active Learning:
