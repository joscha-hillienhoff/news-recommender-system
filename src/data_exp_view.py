import numpy as np
import pandas as pd

train_data = pd.read_csv("MINDsmall_train/behaviors.tsv", sep="\t", header=None)
val_data = pd.read_csv("MINDsmall_train/valiation_behaviors.tsv", sep="\t", header=None)

train_users = set(train_data[1])
val_users = set(val_data[1])

new_users = val_users - train_users

# print(len(train_users), len(val_users), len(new_users))

model_df = pd.read_csv("model_best.csv")

hybrid_data = pd.DataFrame(
    columns=[
        "idx",
        "time_imp",
        "imp_count",
        "hist_count",
        "imp_type",
        "hist_type",
        "user_in_train",
        "perc_itm_in_train",
    ]
)
hybrid_data["idx"] = model_df.iloc[:, 0]
hybrid_data["time_imp"] = val_data.iloc[:, 2].values
hybrid_data["time_imp"] = pd.to_datetime(hybrid_data["time_imp"], errors="coerce")
hybrid_data["time_imp"] = hybrid_data["time_imp"].dt.hour + (
    hybrid_data["time_imp"].dt.minute >= 30
).astype(int)
hybrid_data["time_imp"] = hybrid_data["time_imp"] % 24

hybrid_data["imp_count"] = val_data.iloc[:, 4].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)
hybrid_data["hist_count"] = val_data.iloc[:, 3].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)
hybrid_data["user_in_train"] = val_data.iloc[:, 1].apply(lambda x: 1 if x in train_users else 0)

history_lists = train_data.iloc[:, 3].dropna().apply(lambda x: str(x).split())
impression_lists = (
    train_data.iloc[:, 4].dropna().apply(lambda x: [imp.split("-")[0] for imp in str(x).split()])
)
# Flatten and combine both lists
all_news_ids = set()
for lst in history_lists:
    all_news_ids.update(lst)
for lst in impression_lists:
    all_news_ids.update(lst)

train_itms = list(all_news_ids)

history_lists = val_data.iloc[:, 3].dropna().apply(lambda x: str(x).split())
impression_lists = (
    val_data.iloc[:, 4].dropna().apply(lambda x: [imp.split("-")[0] for imp in str(x).split()])
)
# Flatten and combine both lists
all_news_ids = set()
for lst in history_lists:
    all_news_ids.update(lst)
for lst in impression_lists:
    all_news_ids.update(lst)

val_items = list(all_news_ids)

train_itms_set = set(train_itms)

# Calculate percentage for each row
percentages = list[float] = []
for imp_list in impression_lists:
    if not imp_list:  # avoid division by zero
        percentages.append(0)
        continue
    present = sum(1 for item in imp_list if item in train_itms_set)
    percentage = (present / len(imp_list)) * 100
    percentages.append(percentage)

# Assign to hybrid_data
hybrid_data["perc_itm_in_train"] = percentages
hybrid_data["best_model"] = model_df.iloc[:, 1].values
hybrid_data = hybrid_data.drop(columns=["imp_type", "hist_type", "user_in_train"])
print(hybrid_data.head())

hybrid_data.to_csv("data_for_classifierv3.csv", index=False)
