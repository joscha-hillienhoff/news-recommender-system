# Import necessary libraries
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras

# Load the data
df = pd.read_csv("data_for_classifierv3.csv")

# Drop the 'idx' column as it is just an identifier and not a feature
df = df.drop("idx", axis=1)

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Split the data into features (X) and target (y)
X = df.drop("best_model", axis=1)
y = df["best_model"]

# Scale the features (important for models like KNN and neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize a DataFrame to store all predictions
predictions_df = pd.DataFrame({"True_Label": y})

# --- Train and Evaluate Models on Entire Dataset ---

# 1. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_scaled, y)
y_pred_dt = dt.predict(X_scaled)
print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y, y_pred_dt))
print(classification_report(y, y_pred_dt))
predictions_df["DT_Prediction"] = y_pred_dt

# 2. Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)
y_pred_rf = rf.predict(X_scaled)
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y, y_pred_rf))
print(classification_report(y, y_pred_rf))
predictions_df["RF_Prediction"] = y_pred_rf

# 3. Logistic Regression (multinomial for multi-class)
lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42, max_iter=1000)
lr.fit(X_scaled, y)
y_pred_lr = lr.predict(X_scaled)
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y, y_pred_lr))
print(classification_report(y, y_pred_lr))
predictions_df["LR_Prediction"] = y_pred_lr

# 4. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)
y_pred_knn = knn.predict(X_scaled)
print("\nKNN Results:")
print("Accuracy:", accuracy_score(y, y_pred_knn))
print(classification_report(y, y_pred_knn))
predictions_df["KNN_Prediction"] = y_pred_knn

# 5. LightGBM (Boosting Model)
lgb_model = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=42)
lgb_model.fit(X_scaled, y)
y_pred_lgb = lgb_model.predict(X_scaled)
print("\nLightGBM Results:")
print("Accuracy:", accuracy_score(y, y_pred_lgb))
print(classification_report(y, y_pred_lgb))
predictions_df["LGB_Prediction"] = y_pred_lgb

# 6. Artificial Neural Network (ANN)
# One-hot encode the target for the neural network
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))

# Build the neural network model
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(X_scaled.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),  # 3 classes
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model on the entire dataset
model.fit(X_scaled, y_onehot, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

# Predict on the entire dataset
y_pred_nn = model.predict(X_scaled)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1) + 1  # Convert back to 1,2,3
print("\nNeural Network Results:")
print("Accuracy:", accuracy_score(y, y_pred_nn_classes))
print(classification_report(y, y_pred_nn_classes))
predictions_df["NN_Prediction"] = y_pred_nn_classes

# Save all predictions to a single CSV file
predictions_df.to_csv("all_classifier_predictions.csv", index=False)

print("========================")
print("Predictions for all classifiers have been saved to 'all_classifier_predictions.csv'.")
