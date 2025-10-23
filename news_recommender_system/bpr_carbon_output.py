from datetime import datetime
import gc
import json
import os

from codecarbon import EmissionsTracker  # For carbon emissions tracking
import cornac
import numpy as np
import pandas as pd
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.utils.constants import SEED
from recommenders.utils.timer import Timer
from sklearn.metrics import roc_auc_score

# Set hyperparameters
NUM_FACTORS = 200
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
LAMBDA_REG = 0.001
TOP_K = 10
PRED_BATCH_SIZE = 10000


# Define the evaluation metrics from the scoring script
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0


# Load the MIND behavior data
train_behav = pd.read_csv("MINDsmall_train/behaviors.tsv", sep="\t", header=None)
train_behav.columns = ["Impression ID", "User ID", "Time", "History", "Impressions"]
train_behav["Impressions"] = train_behav["Impressions"].apply(lambda x: x.split(" "))
train_behav["Clicked"] = [[] for _ in range(len(train_behav))]
for i in range(len(train_behav["Impressions"])):
    lis = train_behav["Impressions"][i]
    len_lis = len(lis)
    for j in range(len_lis):
        curr = lis[j]
        news_article = curr[:-2]
        is_viewed = curr[-1]
        lis[j] = news_article
        if is_viewed == "1":
            train_behav.loc[i, "Clicked"].append(news_article)
behavior = train_behav
behavior["Time"] = pd.to_datetime(behavior["Time"])
behavior_exploded = behavior.explode("Clicked")
interactions = behavior_exploded[["User ID", "Clicked", "Time"]].copy()
interactions["rating"] = 1  # Implicit feedback: clicked items get rating=1
interactions.rename(columns={"User ID": "userID", "Clicked": "itemID"}, inplace=True)

# Split data into train and test sets based on the 90th percentile of Time
test_time_th = interactions["Time"].quantile(0.9)
train = interactions[interactions["Time"] < test_time_th].copy()
test = interactions[interactions["Time"] >= test_time_th].copy()

# Get unique items for prediction filtering
unique_items = interactions["itemID"].unique()
print(f"Total unique items: {len(unique_items)}")

# Create Cornac Dataset from training data
train_set = cornac.data.Dataset.from_uir(
    train[["userID", "itemID", "rating"]].itertuples(index=False), seed=SEED
)

# Initialize the BPR model
bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lambda_reg=LAMBDA_REG,
    verbose=True,
    seed=SEED,
)

# --- Step 1: Setup Carbon Emissions Tracking ---
tracker = EmissionsTracker(
    project_name="bpr_recommendation", output_dir="emissions", log_level="critical"
)
tracker.start()  # Start tracking emissions

# Train the model
with Timer() as t:
    bpr.fit(train_set)
print(f"Took {t} seconds for training.")

# Get user-item pairs from test set
test_user_items = test[["userID", "itemID"]].copy()


# Custom prediction function that handles matrix operations correctly
def custom_predict_for_user_batch(model, user_ids, test_df, user_map, item_map):
    """Generate predictions for specific user-item pairs"""
    results = []

    # Create batches of users
    batch_size = PRED_BATCH_SIZE  # Adjust based on memory constraints
    n_batches = (len(user_ids) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx+1}/{n_batches}")

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(user_ids))
        batch_users = user_ids[start_idx:end_idx]
        batch_results = []

        # Process one user at a time to avoid memory issues
        for user_id in batch_users:
            # Skip users not in the training set
            if user_id not in user_map:
                continue

            user_idx = user_map.get(user_id)

            # Get items this user interacted with in test set
            user_test_items = test_df[test_df["userID"] == user_id]["itemID"].values

            # Skip if no test items
            if len(user_test_items) == 0:
                continue

            # Map items to internal indices and filter out those not in training
            valid_item_indices = []
            valid_original_items = []

            for item_id in user_test_items:
                if item_id in item_map:
                    valid_item_indices.append(item_map[item_id])
                    valid_original_items.append(item_id)

            if not valid_item_indices:
                continue

            # Score one item at a time to avoid shape mismatch
            item_scores = []
            for idx in valid_item_indices:
                # Model expects a single item index as an integer
                score = model.score(user_idx, idx)
                item_scores.append(score)

            # Create predictions for this user
            for j, item_id in enumerate(valid_original_items):
                batch_results.append((user_id, item_id, float(item_scores[j])))

        results.extend(batch_results)

        # Force garbage collection after each batch
        gc.collect()

    return pd.DataFrame(results, columns=["userID", "itemID", "prediction"])


# Get user and item mappings from the model
print("Getting user and item mappings...")
user_map = {u: i for i, u in enumerate(train_set.uid_map.keys())}
item_map = {i: j for j, i in enumerate(train_set.iid_map.keys())}

print(f"User map size: {len(user_map)}, Item map size: {len(item_map)}")

# Generate predictions in a memory-efficient way
with Timer() as t:
    unique_test_users = test_user_items["userID"].unique()
    print(f"Generating predictions for {len(unique_test_users)} users")

    # Get predictions only for test user-item pairs
    predictions = custom_predict_for_user_batch(
        bpr, unique_test_users, test_user_items, user_map, item_map
    )

    # print(f"Took {t} seconds for prediction. Final predictions shape: {predictions.shape}")

# Prepare test data for evaluation
test_eval = test.copy()
test_eval = test_eval[["userID", "itemID", "rating"]]

# Evaluate the model using ranking metrics from recommenders library
print("Evaluating predictions with recommenders metrics...")
eval_map = map_at_k(test_eval, predictions, col_prediction="prediction", k=TOP_K)
eval_ndcg = ndcg_at_k(test_eval, predictions, col_prediction="prediction", k=TOP_K)
eval_precision = precision_at_k(test_eval, predictions, col_prediction="prediction", k=TOP_K)
eval_recall = recall_at_k(test_eval, predictions, col_prediction="prediction", k=TOP_K)

# Print evaluation results from recommenders library
print("\n=== Recommenders Library Metrics ===")
print(
    "MAP@10:\t%f" % eval_map,
    "NDCG@10:\t%f" % eval_ndcg,
    "Precision@10:\t%f" % eval_precision,
    "Recall@10:\t%f" % eval_recall,
    sep="\n",
)

# --- Step 2: Output Carbon Emissions Report ---
emissions = tracker.stop()  # Stop tracking and get emissions
print(f"💡 Carbon emissions from this run: {emissions:.6f} kg CO2eq")

# Display detailed emissions information and write to txt
try:
    # Load latest emissions entry
    df_emissions = pd.read_csv("emissions/emissions.csv")
    emissions_data = df_emissions.iloc[-1]

    # Prepare values
    duration_hr = emissions_data["duration"] / 3600
    energy_kwh = emissions_data["energy_consumed"]
    cpu_power = emissions_data["cpu_power"]
    gpu_power = (
        f"{emissions_data['gpu_power']:.2f} W"
        if "gpu_power" in emissions_data and not pd.isna(emissions_data["gpu_power"])
        else "Not available"
    )
    country = (
        emissions_data["country_name"] if "country_name" in emissions_data else "Not available"
    )

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
📄 Emissions Report – {timestamp}
====================================
🌱 Total Emissions:     {emissions:.6f} kg CO2eq

🕒 Duration:            {duration_hr:.2f} hours
⚡ Energy Consumed:     {energy_kwh:.4f} kWh
🧠 CPU Power:           {cpu_power:.2f} W
🎮 GPU Power:           {gpu_power}

🌍 Country:             {country}
====================================
"""

    # Ensure output directory exists
    os.makedirs("emissions", exist_ok=True)


except Exception as e:
    print(f"\n❗ Could not load detailed emissions data: {str(e)}")
