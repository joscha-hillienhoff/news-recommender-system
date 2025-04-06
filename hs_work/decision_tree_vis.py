# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data_for_classifierv3.csv')

# Drop the 'idx' column as it is just an identifier and not a feature
df = df.drop('idx', axis=1)

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Split the data into features (X) and target (y)
X = df.drop('best_model', axis=1)
y = df['best_model']

# Scale the features (important for models like KNN and neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize a DataFrame to store all predictions
predictions_df = pd.DataFrame({'True_Label': y})

# --- Train and Evaluate Models on Entire Dataset ---

# 1. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_scaled, y)
y_pred_dt = dt.predict(X_scaled)
print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y, y_pred_dt))
print(classification_report(y, y_pred_dt))
predictions_df['DT_Prediction'] = y_pred_dt


# Visualize the Decision Tree using plot_tree
# plt.figure(figsize=(20, 10))  # Set figure size for readability
# plot_tree(dt, feature_names=X.columns, class_names=['1', '2', '3'], filled=True, rounded=True)
# plt.title("Decision Tree Visualization")
# plt.show()

# 2. Random Forest
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_scaled, y)
# y_pred_rf = rf.predict(X_scaled)
# print("\nRandom Forest Results:")
# print("Accuracy:", accuracy_score(y, y_pred_rf))
# print(classification_report(y, y_pred_rf))
# predictions_df['RF_Prediction'] = y_pred_rf

# # 3. Logistic Regression (multinomial for multi-class)
# lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=1000)
# lr.fit(X_scaled, y)
# y_pred_lr = lr.predict(X_scaled)
# print("\nLogistic Regression Results:")
# print("Accuracy:", accuracy_score(y, y_pred_lr))
# print(classification_report(y, y_pred_lr))
# predictions_df['LR_Prediction'] = y_pred_lr


# --- Output Decision Tree Rules as Text ---
def tree_to_code(tree, feature_names, class_names, node_index=0, indent=""):
    """Recursively generates text representation of the decision tree rules."""

    if tree.children_left[node_index] == tree.children_right[node_index]:  # Leaf node
        leaf_values = tree.value[node_index][0]
        predicted_class_index = np.argmax(leaf_values)
        print(f"{indent}Predict: {class_names[predicted_class_index]} (samples: {int(sum(leaf_values))})")
    else:
        feature = feature_names[tree.feature[node_index]]
        threshold = tree.threshold[node_index]

        print(f"{indent}If {feature} <= {threshold:.4f}:")
        tree_to_code(tree, feature_names, class_names, tree.children_left[node_index], indent + "  ")

        print(f"{indent}Else {feature} > {threshold:.4f}:")
        tree_to_code(tree, feature_names, class_names, tree.children_right[node_index], indent + "  ")

# Get feature names (before scaling, if you want original names)
feature_names = X.columns.tolist()
class_names = y.unique().astype(str).tolist() # Get unique class names as strings

print("\nDecision Tree Rules (Text Output):")
tree_to_code(dt.tree_, feature_names, class_names)