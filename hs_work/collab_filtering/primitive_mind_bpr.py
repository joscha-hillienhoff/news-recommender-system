import pandas as pd
import cornac
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
import numpy as np
import gc
import json
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
    gains = 2 ** y_true - 1
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
behavior['Time'] = pd.to_datetime(behavior['Time'])
behavior_exploded = behavior.explode('Clicked')
interactions = behavior_exploded[['User ID', 'Clicked', 'Time']].copy()
interactions['rating'] = 1  # Implicit feedback: clicked items get rating=1
interactions.rename(columns={'User ID': 'userID', 'Clicked': 'itemID'}, inplace=True)

# Split data into train and test sets based on the 90th percentile of Time
test_time_th = interactions['Time'].quantile(0.9)
train = interactions[interactions['Time'] < test_time_th].copy()
test = interactions[interactions['Time'] >= test_time_th].copy()

# Get unique items for prediction filtering
unique_items = interactions['itemID'].unique()
print(f"Total unique items: {len(unique_items)}")

# Create Cornac Dataset from training data
train_set = cornac.data.Dataset.from_uir(
    train[['userID', 'itemID', 'rating']].itertuples(index=False),
    seed=SEED
)

# Initialize the BPR model
bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lambda_reg=LAMBDA_REG,
    verbose=True,
    seed=SEED
)

# Train the model
with Timer() as t:
    bpr.fit(train_set)
print(f"Took {t} seconds for training.")

# Get user-item pairs from test set
test_user_items = test[['userID', 'itemID']].copy()

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
            user_test_items = test_df[test_df['userID'] == user_id]['itemID'].values
            
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
        
    return pd.DataFrame(results, columns=['userID', 'itemID', 'prediction'])

# Get user and item mappings from the model
print("Getting user and item mappings...")
user_map = {u: i for i, u in enumerate(train_set.uid_map.keys())}
item_map = {i: j for j, i in enumerate(train_set.iid_map.keys())}

print(f"User map size: {len(user_map)}, Item map size: {len(item_map)}")

# Generate predictions in a memory-efficient way
with Timer() as t:
    unique_test_users = test_user_items['userID'].unique()
    print(f"Generating predictions for {len(unique_test_users)} users")
    
    # Get predictions only for test user-item pairs
    predictions = custom_predict_for_user_batch(
        bpr, 
        unique_test_users, 
        test_user_items,
        user_map,
        item_map
    )
    
    print(f"Took seconds for prediction. Final predictions shape: {predictions.shape}")

# Prepare test data for evaluation
test_eval = test.copy()
test_eval = test_eval[['userID', 'itemID', 'rating']]

# Evaluate the model using ranking metrics from recommenders library
print("Evaluating predictions with recommenders metrics...")
eval_map = map_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)

# Print evaluation results from recommenders library
print("\n=== Recommenders Library Metrics ===")
print("MAP@10:\t%f" % eval_map,
      "NDCG@10:\t%f" % eval_ndcg,
      "Precision@10:\t%f" % eval_precision,
      "Recall@10:\t%f" % eval_recall, sep='\n')

# Function to generate the prediction file in the required format for evaluation
def generate_cf_prediction_file(model, input_file="MINDsmall_train/behaviors.tsv", output_file="prediction_cf.txt", 
                                truth_file="MINDsmall_train/truth_val.txt"):
    """
    Generates a prediction file in the required format using the trained CF model.
    Also evaluates the predictions using the scoring metrics.
    
    Each line in the output file will be:
    <ImpressionID> <JSON list of ranks>
    """
    # Load the validation behaviors file
    behaviors_val = pd.read_csv(input_file, sep="\t", header=None)
    behaviors_val.columns = ["ImpressionId", "User ID", "Time", "History", "Impressions"]
    
    # Split the "Impressions" column into a list
    behaviors_val["ImpressionList"] = behaviors_val["Impressions"].apply(lambda x: x.split(" "))
    
    # Clean the news IDs (removing the click suffix, similar to training)
    behaviors_val["CleanedImpressions"] = behaviors_val["ImpressionList"].apply(lambda lst: [s[:-2] for s in lst])
    
    # Extract the labels (1 for clicked, 0 for not clicked)
    behaviors_val["Labels"] = behaviors_val["ImpressionList"].apply(lambda lst: [int(s[-1]) for s in lst])
    
    # Get the user and item mappings from the CF model (internal IDs)
    user_map = model.train_set.uid_map
    item_map = model.train_set.iid_map

    # Lists to store metrics for each impression
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    with open(output_file, "w") as f:
        # Iterate over each impression in the validation file
        for idx, row in behaviors_val.iterrows():
            imp_id = row["ImpressionId"]
            user_id = row["User ID"]
            impression_news = row["CleanedImpressions"]
            labels = row["Labels"]
            
            # Skip impressions with no labels (for consistency with the scoring script)
            if not labels:
                continue
            
            # Compute predicted score for each news in the impression
            scores = []
            for news in impression_news:
                # Only compute if the user and news are in the CF model's training mappings
                if user_id in user_map and news in item_map:
                    u_idx = user_map[user_id]
                    i_idx = item_map[news]
                    score = model.score(u_idx, i_idx)
                else:
                    score = 0  # Default score if unknown
                scores.append(score)
            
            # Rank the news articles by predicted score (higher is better)
            # This gives a new list where the best news is first
            sorted_indices = np.argsort(scores)[::-1]
            ranked_news = [impression_news[i] for i in sorted_indices]
            
            # For the output, we need a list of ranks in the original order of impression_news.
            # For each news in the original order, its rank is the index in the ranked_news list + 1.
            ranks = [ranked_news.index(news) + 1 for news in impression_news]
            
            # Write the impression id and JSON list of ranks to the file
            f.write(f"{imp_id} {json.dumps(ranks)}\n")
            
            # Calculate metrics for this impression
            y_true = np.array(labels, dtype='float32')
            y_score = [1.0/rank for rank in ranks]  # Convert ranks to scores (higher is better)
            
            # Calculate and store metrics
            auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0
            mrr = mrr_score(y_true, y_score)
            ndcg5 = ndcg_score(y_true, y_score, 5)
            ndcg10 = ndcg_score(y_true, y_score, 10)
            
            aucs.append(auc)
            mrrs.append(mrr)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)
    
    print(f"✅ Prediction file '{output_file}' created.")
    
    # Calculate and print average metrics
    avg_auc = np.mean(aucs)
    avg_mrr = np.mean(mrrs)
    avg_ndcg5 = np.mean(ndcg5s)
    avg_ndcg10 = np.mean(ndcg10s)
    
    print("\n=== MIND Evaluation Metrics ===")
    print(f"AUC: {avg_auc:.4f}")
    print(f"MRR: {avg_mrr:.4f}")
    print(f"nDCG@5: {avg_ndcg5:.4f}")
    print(f"nDCG@10: {avg_ndcg10:.4f}")
    
    # Write metrics to an output file
    with open("scores_cf.txt", "w") as f:
        f.write(f"AUC:{avg_auc:.4f}\nMRR:{avg_mrr:.4f}\nnDCG@5:{avg_ndcg5:.4f}\nnDCG@10:{avg_ndcg10:.4f}")
    
    return avg_auc, avg_mrr, avg_ndcg5, avg_ndcg10

# Generate predictions and evaluate using the MIND metrics
auc, mrr, ndcg5, ndcg10 = generate_cf_prediction_file(
    bpr, 
    input_file="MINDsmall_train/valiation_behaviors.tsv", 
    output_file="prediction_cf.txt"
)

print("\n=== Final MIND Evaluation Metrics Summary ===")
print(f"AUC: {auc:.4f}")
print(f"MRR: {mrr:.4f}")
print(f"nDCG@5: {ndcg5:.4f}")
print(f"nDCG@10: {ndcg10:.4f}")