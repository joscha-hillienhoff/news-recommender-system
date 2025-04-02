# Import necessary libraries
import pandas as pd
import cornac
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

# Set hyperparameters
NUM_FACTORS = 200  # Latent factors (same as MovieLens BPR notebook)
NUM_EPOCHS = 100   # Number of training iterations
LEARNING_RATE = 0.01  # Learning rate for optimization
LAMBDA_REG = 0.001    # Regularization parameter
TOP_K = 10            # Top-K items for evaluation metrics

# Load the MIND behavior data
# Update the path to where your behaviour.csv is located
behavior = pd.read_csv("/kaggle/input/mind-processed-daniel/behaviour.csv")

# Add a 'rating' column with value 1 for all clicks (implicit feedback)
behavior['rating'] = 1

# Rename columns to match the BPR notebook's expected format
behavior.rename(columns={'userId': 'userID', 'click': 'itemID'}, inplace=True)

# Split into train and test sets based on time (90th percentile of epochhrs)
test_time_th = behavior['epochhrs'].quantile(0.9)
train = behavior[behavior['epochhrs'] < test_time_th].copy()
test = behavior[behavior['epochhrs'] >= test_time_th].copy()

# Create Cornac Dataset for training
# Cornac expects tuples of (userID, itemID, rating)
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

# Train the model and measure training time
with Timer() as t:
    bpr.fit(train_set)
print(f"Took {t} seconds for training.")

# Generate ranking predictions for all user-item pairs not in the training set
with Timer() as t:
    all_predictions = predict_ranking(
        bpr,
        train,
        usercol='userID',
        itemcol='itemID',
        remove_seen=True  # Exclude items the user has already seen in training
    )
print(f"Took {t} seconds for prediction.")

# Evaluate the model using ranking metrics
eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

# Print evaluation results
print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')