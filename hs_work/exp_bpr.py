# import pandas as pd
# import ast
# import cornac
# from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
# from recommenders.models.cornac.cornac_utils import predict_ranking
# from recommenders.utils.timer import Timer
# from recommenders.utils.constants import SEED
# import numpy as np
# import gc

# # Set hyperparameters
# NUM_FACTORS = 200
# NUM_EPOCHS = 100
# LEARNING_RATE = 0.01
# LAMBDA_REG = 0.001
# TOP_K = 10
# PRED_BATCH_SIZE = 10000

# # Load the MIND behavior data
# train_behav = pd.read_csv("MINDsmall_train/behaviors.tsv", sep="\t", header=None)
# train_behav.columns = ["Impression ID", "User ID", "Time", "History", "Impressions"]
# train_behav["Impressions"] = train_behav["Impressions"].apply(lambda x: x.split(" "))
# train_behav["Clicked"] = [[] for _ in range(len(train_behav))]
# for i in range(len(train_behav["Impressions"])):
#     lis = train_behav["Impressions"][i]
#     len_lis = len(lis)
#     for j in range(len_lis):
#         curr = lis[j]
#         news_article = curr[:-2]
#         is_viewed = curr[-1]
#         lis[j] = news_article
#         if is_viewed == "1":
#             train_behav.loc[i, "Clicked"].append(news_article)
# behavior = train_behav
# behavior['Time'] = pd.to_datetime(behavior['Time'])
# behavior_exploded = behavior.explode('Clicked')
# interactions = behavior_exploded[['User ID', 'Clicked', 'Time']].copy()
# interactions['rating'] = 1  # Implicit feedback: clicked items get rating=1
# interactions.rename(columns={'User ID': 'userID', 'Clicked': 'itemID'}, inplace=True)

# # Split data into train and test sets based on the 90th percentile of Time
# # valid_behav = pd.read_csv("MINDsmall_train/valiation_behaviors.tsv", sep="\t", header=None)
# # valid_behav.columns = ["Impression ID", "User ID", "Time", "History", "Impressions"]
# # valid_behav["Impressions"] = valid_behav["Impressions"].apply(lambda x: x.split(" "))
# # valid_behav["Clicked"] = [[] for _ in range(len(valid_behav))]
# # for i in range(len(valid_behav["Impressions"])):
# #     lis = valid_behav["Impressions"][i]
# #     len_lis = len(lis)
# #     for j in range(len_lis):
# #         curr = lis[j]
# #         news_article = curr[:-2]
# #         is_viewed = curr[-1]
# #         lis[j] = news_article
# #         if is_viewed == "1":
# #             valid_behav.loc[i, "Clicked"].append(news_article)

# # validation = valid_behav
# # validation['Time'] = pd.to_datetime(validation['Time'])
# # behavior_exploded = validation.explode('Clicked')
# # interactions = behavior_exploded[['User ID', 'Clicked', 'Time']].copy()
# # interactions['rating'] = 1  # Implicit feedback: clicked items get rating=1
# # interactions.rename(columns={'User ID': 'userID', 'Clicked': 'itemID'}, inplace=True)

# # # Split data into train and test sets based on the 90th percentile of Time
# test_time_th = interactions['Time'].quantile(0.9)
# train = interactions[interactions['Time'] < test_time_th].copy()
# test = interactions[interactions['Time'] >= test_time_th].copy()
# # train = behavior
# # test = validation

# # Get unique items for prediction filtering
# unique_items = interactions['itemID'].unique()
# print(f"Total unique items: {len(unique_items)}")

# # Create Cornac Dataset from training data
# train_set = cornac.data.Dataset.from_uir(
#     train[['userID', 'itemID', 'rating']].itertuples(index=False),
#     seed=SEED
# )

# # Initialize the BPR model
# bpr = cornac.models.BPR(
#     k=NUM_FACTORS,
#     max_iter=NUM_EPOCHS,
#     learning_rate=LEARNING_RATE,
#     lambda_reg=LAMBDA_REG,
#     verbose=True,
#     seed=SEED
# )

# # Train the model
# with Timer() as t:
#     bpr.fit(train_set)
# print(f"Took {t} seconds for training.")

# # Get user-item pairs from test set
# test_user_items = test[['userID', 'itemID']].copy()

# # Custom prediction function that handles matrix operations correctly
# def custom_predict_for_user_batch(model, user_ids, test_df, user_map, item_map):
#     """Generate predictions for specific user-item pairs"""
#     results = []
    
#     # Create batches of users
#     batch_size = PRED_BATCH_SIZE  # Adjust based on memory constraints
#     n_batches = (len(user_ids) + batch_size - 1) // batch_size
    
#     for batch_idx in range(n_batches):
#         if batch_idx % 10 == 0:
#             print(f"Processing batch {batch_idx+1}/{n_batches}")
            
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(user_ids))
#         batch_users = user_ids[start_idx:end_idx]
#         batch_results = []
        
#         # Process one user at a time to avoid memory issues
#         for user_id in batch_users:
#             # Skip users not in the training set
#             if user_id not in user_map:
#                 continue
                
#             user_idx = user_map.get(user_id)
            
#             # Get items this user interacted with in test set
#             user_test_items = test_df[test_df['userID'] == user_id]['itemID'].values
            
#             # Skip if no test items
#             if len(user_test_items) == 0:
#                 continue
                
#             # Map items to internal indices and filter out those not in training
#             valid_item_indices = []
#             valid_original_items = []
            
#             for item_id in user_test_items:
#                 if item_id in item_map:
#                     valid_item_indices.append(item_map[item_id])
#                     valid_original_items.append(item_id)
            
#             if not valid_item_indices:
#                 continue
            
#             # Score one item at a time to avoid shape mismatch
#             item_scores = []
#             for idx in valid_item_indices:
#                 # Model expects a single item index as an integer
#                 score = model.score(user_idx, idx)
#                 item_scores.append(score)
            
#             # Create predictions for this user
#             for j, item_id in enumerate(valid_original_items):
#                 batch_results.append((user_id, item_id, float(item_scores[j])))
        
#         results.extend(batch_results)
        
#         # Force garbage collection after each batch
#         gc.collect()
        
#     return pd.DataFrame(results, columns=['userID', 'itemID', 'prediction'])

# # Get user and item mappings from the model
# print("Getting user and item mappings...")
# user_map = {u: i for i, u in enumerate(train_set.uid_map.keys())}
# item_map = {i: j for j, i in enumerate(train_set.iid_map.keys())}

# print(f"User map size: {len(user_map)}, Item map size: {len(item_map)}")

# # Generate predictions in a memory-efficient way
# with Timer() as t:
#     unique_test_users = test_user_items['userID'].unique()
#     print(f"Generating predictions for {len(unique_test_users)} users")
    
#     # Get predictions only for test user-item pairs
#     predictions = custom_predict_for_user_batch(
#         bpr, 
#         unique_test_users, 
#         test_user_items,
#         user_map,
#         item_map
#     )
    
#     # print(f"Took {t} seconds for prediction. Final predictions shape: {predictions.shape}")
#     print(f"Took shit tonne of time for prediction. Final predictions shape: {predictions.shape}")

# # Prepare test data for evaluation
# test_eval = test.copy()
# test_eval = test_eval[['userID', 'itemID', 'rating']]

# # Evaluate the model using ranking metrics
# print("Evaluating predictions...")
# eval_map = map_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
# eval_ndcg = ndcg_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
# eval_precision = precision_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
# eval_recall = recall_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)

# # Print evaluation results
# print("MAP:\t%f" % eval_map,
#       "NDCG:\t%f" % eval_ndcg,
#       "Precision@K:\t%f" % eval_precision,
#       "Recall@K:\t%f" % eval_recall, sep='\n')

# import json
# import numpy as np
# import pandas as pd

# def generate_cf_prediction_file(model, input_file="MINDsmall_train/behaviors.tsv", output_file="prediction_cf.txt"):
#     """
#     Generates a prediction file in the required format using the trained CF model.
    
#     Each line in the output file will be:
#     <ImpressionID> <JSON list of ranks>
#     """
#     # Load the validation behaviors file
#     behaviors_val = pd.read_csv(input_file, sep="\t", header=None)
#     behaviors_val.columns = ["ImpressionId", "User ID", "Time", "History", "Impressions"]
    
#     # Split the "Impressions" column into a list
#     behaviors_val["ImpressionList"] = behaviors_val["Impressions"].apply(lambda x: x.split(" "))
    
#     # Clean the news IDs (removing the click suffix, similar to training)
#     behaviors_val["CleanedImpressions"] = behaviors_val["ImpressionList"].apply(lambda lst: [s[:-2] for s in lst])
    
#     # Get the user and item mappings from the CF model (internal IDs)
#     user_map = model.train_set.uid_map
#     item_map = model.train_set.iid_map

#     with open(output_file, "w") as f:
#         # Iterate over each impression in the validation file
#         for idx, row in behaviors_val.iterrows():
#             imp_id = row["ImpressionId"]
#             user_id = row["User ID"]
#             impression_news = row["CleanedImpressions"]
            
#             # Compute predicted score for each news in the impression
#             scores = []
#             for news in impression_news:
#                 # Only compute if the user and news are in the CF model's training mappings
#                 if user_id in user_map and news in item_map:
#                     u_idx = user_map[user_id]
#                     i_idx = item_map[news]
#                     score = model.score(u_idx, i_idx)
#                 else:
#                     score = 0  # Default score if unknown
#                 scores.append(score)
            
#             # Rank the news articles by predicted score (higher is better)
#             # This gives a new list where the best news is first
#             sorted_indices = np.argsort(scores)[::-1]
#             ranked_news = [impression_news[i] for i in sorted_indices]
            
#             # For the output, we need a list of ranks in the original order of impression_news.
#             # For each news in the original order, its rank is the index in the ranked_news list + 1.
#             ranks = [ranked_news.index(news) + 1 for news in impression_news]
            
#             # Write the impression id and JSON list of ranks to the file
#             f.write(f"{imp_id} {json.dumps(ranks)}\n")
    
#     print(f"✅ Prediction file '{output_file}' created.")

# # Example usage:
# generate_cf_prediction_file(bpr, input_file="MINDsmall_train/behaviors.tsv", output_file="prediction_cf.txt")



import sys, os, os.path
import pandas as pd
import ast
import cornac
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
import numpy as np
import gc
import json
from sklearn.metrics import roc_auc_score

# -------------------------------
# CONFIGURATION & HYPERPARAMETERS
# -------------------------------
NUM_FACTORS = 200
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
LAMBDA_REG = 0.001
TOP_K = 10
PRED_BATCH_SIZE = 10000

# --------------------------------
# LOAD AND PREPROCESS MIND BEHAVIOR DATA
# --------------------------------
train_behav = pd.read_csv("MINDsmall_train/behaviors.tsv", sep="\t", header=None)
train_behav.columns = ["Impression ID", "User ID", "Time", "History", "Impressions"]
train_behav["Impressions"] = train_behav["Impressions"].apply(lambda x: x.split(" "))

# Extract clicked items (implicit positive feedback)
train_behav["Clicked"] = [[] for _ in range(len(train_behav))]
for i in range(len(train_behav["Impressions"])):
    lis = train_behav["Impressions"][i]
    for j in range(len(lis)):
        curr = lis[j]
        news_article = curr[:-2]  # remove last two characters
        is_viewed = curr[-1]      # last character indicates click (1 means clicked)
        lis[j] = news_article
        if is_viewed == "1":
            train_behav.loc[i, "Clicked"].append(news_article)

# Use the exploded clicked interactions
behavior = train_behav
behavior['Time'] = pd.to_datetime(behavior['Time'])
behavior_exploded = behavior.explode('Clicked')
interactions = behavior_exploded[['User ID', 'Clicked', 'Time']].copy()
interactions['rating'] = 1  # all positive implicit feedback
interactions.rename(columns={'User ID': 'userID', 'Clicked': 'itemID'}, inplace=True)

# -------------------------------
# SPLIT DATA INTO TRAIN & TEST
# -------------------------------
test_time_th = interactions['Time'].quantile(0.9)
train = interactions[interactions['Time'] < test_time_th].copy()
test = interactions[interactions['Time'] >= test_time_th].copy()
print(f"Total unique items: {len(interactions['itemID'].unique())}")

# ----------------------------------
# CREATE CORNAC DATASET & TRAIN MODEL
# ----------------------------------
train_set = cornac.data.Dataset.from_uir(
    train[['userID', 'itemID', 'rating']].itertuples(index=False),
    seed=SEED
)

bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lambda_reg=LAMBDA_REG,
    verbose=True,
    seed=SEED
)

with Timer() as t:
    bpr.fit(train_set)
print(f"Took {t} seconds for training.")

# ---------------------------------------
# CUSTOM PREDICTION FUNCTION (for evaluation)
# ---------------------------------------
def custom_predict_for_user_batch(model, user_ids, test_df, user_map, item_map):
    """Generate predictions for specific user-item pairs"""
    results = []
    batch_size = PRED_BATCH_SIZE
    n_batches = (len(user_ids) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx+1}/{n_batches}")
            
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(user_ids))
        batch_users = user_ids[start_idx:end_idx]
        batch_results = []
        
        for user_id in batch_users:
            if user_id not in user_map:
                continue
                
            user_idx = user_map.get(user_id)
            user_test_items = test_df[test_df['userID'] == user_id]['itemID'].values
            if len(user_test_items) == 0:
                continue
                
            valid_item_indices = []
            valid_original_items = []
            
            for item_id in user_test_items:
                if item_id in item_map:
                    valid_item_indices.append(item_map[item_id])
                    valid_original_items.append(item_id)
            
            if not valid_item_indices:
                continue
            
            item_scores = []
            for idx in valid_item_indices:
                score = model.score(user_idx, idx)
                item_scores.append(score)
            
            for j, item_id in enumerate(valid_original_items):
                batch_results.append((user_id, item_id, float(item_scores[j])))
        results.extend(batch_results)
        gc.collect()
        
    return pd.DataFrame(results, columns=['userID', 'itemID', 'prediction'])

# Get mappings from CF model
print("Getting user and item mappings...")
user_map = {u: i for i, u in enumerate(train_set.uid_map.keys())}
item_map = {i: j for j, i in enumerate(train_set.iid_map.keys())}
print(f"User map size: {len(user_map)}, Item map size: {len(item_map)}")

# Generate predictions for test set pairs
test_user_items = test[['userID', 'itemID']].copy()
with Timer() as t:
    unique_test_users = test_user_items['userID'].unique()
    print(f"Generating predictions for {len(unique_test_users)} users")
    predictions = custom_predict_for_user_batch(bpr, unique_test_users, test_user_items, user_map, item_map)
    print(f"Took a long time for prediction. Final predictions shape: {predictions.shape}")

# Evaluate using Cornac evaluation functions (for ranking)
test_eval = test[['userID', 'itemID', 'rating']].copy()
print("Evaluating predictions with Cornac metrics...")
eval_map = map_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test_eval, predictions, col_prediction='prediction', k=TOP_K)
print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')

# -----------------------------------------------------
# FUNCTIONS FROM EVALUATION SCRIPT (for file-based eval)
# -----------------------------------------------------
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
    
def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def parse_line(l):
    parts = l.strip('\n').split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid line format: {l} (expected two parts)")
    impid, ranks = parts
    ranks = json.loads(ranks)
    return impid, ranks

def scoring(truth_f, sub_f):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    
    line_index = 1
    for lt in truth_f:
        ls = sub_f.readline()
        impid, labels = parse_line(lt)
        
        if labels == []:
            continue 
        if ls == '':
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index))       
        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                line_index,
                sub_impid,
                impid
            ))
        lt_len = float(len(labels))
        y_true =  np.array(labels, dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1. / rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(
                    line_index,
                    lt_len
                ))
            y_score.append(score_rslt)
        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        ndcg5 = ndcg_score(y_true, y_score, 5)
        ndcg10 = ndcg_score(y_true, y_score, 10)
        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
        line_index += 1
    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)

# --------------------------------------------------
# GENERATE CF PREDICTION FILE (FILE-BASED FORMAT)
# --------------------------------------------------
def generate_cf_prediction_file(model, input_file="MINDsmall_train/behaviors.tsv", output_file="prediction_cf.txt"):
    """
    Generates a prediction file in the required format using the trained CF model.
    Each line in the output file will be:
    <ImpressionID> <JSON list of ranks>
    """
    behaviors_val = pd.read_csv(input_file, sep="\t", header=None)
    behaviors_val.columns = ["ImpressionId", "User ID", "Time", "History", "Impressions"]
    behaviors_val["ImpressionList"] = behaviors_val["Impressions"].apply(lambda x: x.split(" "))
    behaviors_val["CleanedImpressions"] = behaviors_val["ImpressionList"].apply(lambda lst: [s[:-2] for s in lst])
    
    user_map = model.train_set.uid_map
    item_map = model.train_set.iid_map

    with open(output_file, "w") as f:
        for idx, row in behaviors_val.iterrows():
            imp_id = row["ImpressionId"]
            user_id = row["User ID"]
            impression_news = row["CleanedImpressions"]
            scores = []
            for news in impression_news:
                if user_id in user_map and news in item_map:
                    u_idx = user_map[user_id]
                    i_idx = item_map[news]
                    score = model.score(u_idx, i_idx)
                else:
                    score = 0
                scores.append(score)
            sorted_indices = np.argsort(scores)[::-1]
            ranked_news = [impression_news[i] for i in sorted_indices]
            ranks = [ranked_news.index(news) + 1 for news in impression_news]
            f.write(f"{imp_id} {json.dumps(ranks)}\n")
    print(f"✅ Prediction file '{output_file}' created.")

# Generate prediction file using CF model
generate_cf_prediction_file(bpr, input_file="MINDsmall_train/behaviors.tsv", output_file="prediction_gpt_cf.txt")


import csv

def convert_behaviors_to_truth(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        
        for row in reader:
            impression_id = row[0]  # First column is the impression ID
            impressions = row[-1].split()  # Last column contains impressions
            
            # Extract click indicators (0 or 1) from each news-click pair
            click_labels = [pair.split('-')[1] for pair in impressions]
            
            # Write to output file in required format
            outfile.write(f"{impression_id} {click_labels}\n")
    
    print(f"Truth file saved to {output_file}")

# Example usage
convert_behaviors_to_truth("MINDsmall_train/behaviors.tsv", "truth_val.txt")


# --------------------------------------------------
# PERFORM FILE-BASED EVALUATION USING THE EVALUATION SCRIPT FUNCTIONS
# --------------------------------------------------
# (Assume your truth file is in "input/ref/truth_val.txt" and prediction file is in "input/res/prediction_cf.txt")
truth_file_path = "truth_val.txt"
submission_file_path = "prediction_gpt_cf.txt"
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = os.path.join(output_dir, "scores_val_cf.txt")

try:
    with open(truth_file_path, 'r') as truth_f, open(submission_file_path, 'r') as sub_f:
        auc, mrr, ndcg5, ndcg10 = scoring(truth_f, sub_f)
    with open(output_filename, 'w') as output_file:
        output_file.write("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg5, ndcg10))
    print(f"✅ Evaluation scores saved to {output_filename}")
except Exception as e:
    print("Error during evaluation:", e)
