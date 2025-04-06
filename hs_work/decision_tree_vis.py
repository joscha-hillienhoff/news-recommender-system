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
from codecarbon import EmissionsTracker  
from datetime import datetime
import os

df = pd.read_csv('data_for_classifierv2.csv')
df = df.drop('idx', axis=1)
print("Missing values in each column:")
print(df.isnull().sum())

X = df.drop('best_model', axis=1)
y = df['best_model']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

predictions_df = pd.DataFrame({'True_Label': y})

# --- Step 1: Setup Carbon Emissions Tracking ---
tracker = EmissionsTracker(project_name="classifier_training", output_dir="emissions", log_level="critical")
tracker.start()  # Start tracking emissions

# --- Train and Evaluate Models on Entire Dataset ---

# 1. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_scaled, y)
y_pred_dt = dt.predict(X_scaled)
print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y, y_pred_dt))
print(classification_report(y, y_pred_dt))
predictions_df['DT_Prediction'] = y_pred_dt

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

feature_names = X.columns.tolist()
class_names = y.unique().astype(str).tolist() 
# print("\nDecision Tree Rules (Text Output):")
# tree_to_code(dt.tree_, feature_names, class_names)


# --- Step 2: Output Carbon Emissions Report ---
emissions = tracker.stop()  # Stop tracking and get emissions
print(f"💡 Carbon emissions from this run: {emissions:.6f} kg CO2eq")

# Display detailed emissions information and write to txt
try:
    # Load latest emissions entry
    df_emissions = pd.read_csv("emissions/emissions.csv")
    emissions_data = df_emissions.iloc[-1]

    # Prepare values
    duration_hr = emissions_data['duration'] / 3600
    energy_kwh = emissions_data['energy_consumed']
    cpu_power = emissions_data['cpu_power']
    gpu_power = (
        f"{emissions_data['gpu_power']:.2f} W"
        if 'gpu_power' in emissions_data and not pd.isna(emissions_data['gpu_power'])
        else "Not available"
    )
    country = emissions_data['country_name'] if 'country_name' in emissions_data else "Not available"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print to console
    print(f"\nDetailed emissions data:")
    print(f"- Duration: {duration_hr:.2f} hours")
    print(f"- Energy consumed: {energy_kwh:.4f} kWh")
    print(f"- CPU Power: {cpu_power:.2f} W")
    print(f"- GPU Power: {gpu_power}")
    print(f"- Country: {country}")

    # Create structured report text
    report = f"""\
Emissions Report – {timestamp}
====================================
Total Emissions:     {emissions:.6f} kg CO2eq
Duration:            {duration_hr:.2f} hours
Energy Consumed:     {energy_kwh:.4f} kWh
CPU Power:           {cpu_power:.2f} W
GPU Power:           {gpu_power}
Country:             {country}
====================================
"""

    # Ensure output directory exists
    os.makedirs("emissions", exist_ok=True)

    # Save to .txt file

except Exception as e:
    print(f"\n❗ Could not load detailed emissions data: {str(e)}")

print("========================")
print("Predictions saved to 'all_classifier_predictions.csv'.")
print("Emissions report saved to 'emissions/emissions_report_classifier_training.txt'.")