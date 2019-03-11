from urllib.request import urlopen
import pandas as pd
from user_base_cosine import user_based_cosine_predict
from user_base_pearson import user_based_pearson_predict
from item_base_cosine import item_based_cosine_predict
from item_base_adj_cosine import item_based_adj_cosine_predict
from customized_algorithm import customized_predict


# ---------- Main ----------
# import training and testing datasets
training_url = urlopen("http://www.cse.scu.edu/~yfang/coen272/train.txt")
training_table = pd.read_csv(training_url, sep="\s+", header=None)
testing5_url = urlopen("http://www.cse.scu.edu/~yfang/coen272/test5.txt")
testing5_table = pd.read_csv(testing5_url, sep="\s+", header=None)
testing10_url = urlopen("http://www.cse.scu.edu/~yfang/coen272/test10.txt")
testing10_table = pd.read_csv(testing10_url, sep="\s+", header=None)
testing20_url = urlopen("http://www.cse.scu.edu/~yfang/coen272/test20.txt")
testing20_table = pd.read_csv(testing20_url, sep="\s+", header=None)

# set global parameters for every prediction algorithm
amplify = 1
similarity_threshold = 0
inverse_frequency = True

# # ---------- User-based Collaborative Filtering (Cosine Similarity) ----------
# result5 = user_based_cosine_predict(training_table, testing5_table, amplify, similarity_threshold, inverse_frequency)
# result10 = user_based_cosine_predict(training_table, testing5_table, amplify, similarity_threshold, inverse_frequency)
# result20 = user_based_cosine_predict(training_table, testing5_table, amplify, similarity_threshold, inverse_frequency)
#
# # ---------- User-based Collaborative Filtering (Pearson Correlation) ----------
# result5 = user_based_pearson_predict(training_table, testing5_table, amplify, similarity_threshold, inverse_frequency)
# result10 = user_based_pearson_predict(training_table, testing5_table, amplify, similarity_threshold, inverse_frequency)
# result20 = user_based_pearson_predict(training_table, testing5_table, amplify, similarity_threshold, inverse_frequency)
#
# # ---------- Item-based Collaborative Filtering (Cosine Similarity) ----------
# result5 = item_based_cosine_predict(training_table, testing5_table, amplify, similarity_threshold, inverse_frequency)
# result10 = item_based_cosine_predict(training_table, testing10_table, amplify, similarity_threshold, inverse_frequency)
# result20 = item_based_cosine_predict(training_table, testing20_table, amplify, similarity_threshold, inverse_frequency)
#
# # ---------- Item-based Collaborative Filtering (Adjusted Cosine Similarity) ----------
# result5 = item_based_adj_cosine_predict(training_table, testing5_table, amplify, similarity_threshold, inverse_frequency)
# result10 = item_based_adj_cosine_predict(training_table, testing10_table, amplify, similarity_threshold, inverse_frequency)
# result20 = item_based_adj_cosine_predict(training_table, testing20_table, amplify, similarity_threshold, inverse_frequency)

# ------------------------- Customized Algorithm ------------------------------
result = customized_predict(training_table, testing5_table, testing10_table, testing20_table)
result5 = result["result5"]
result10 = result["result10"]
result20 = result["result20"]

# convert ratings to integers with value within 1 ~ 5
for record_id in range(result5.shape[0]):
    result5.at[record_id, 2] = round(result5.at[record_id, 2])
    if result5.at[record_id, 2] > 5:
        result5.at[record_id, 2] = 5
    elif result5.at[record_id, 2] < 1:
        result5.at[record_id, 2] = 1
for record_id in range(result10.shape[0]):
    result10.at[record_id, 2] = round(result10.at[record_id, 2])
    if result10.at[record_id, 2] > 5:
        result10.at[record_id, 2] = 5
    elif result10.at[record_id, 2] < 1:
        result10.at[record_id, 2] = 1
for record_id in range(result20.shape[0]):
    result20.at[record_id, 2] = round(result20.at[record_id, 2])
    if result20.at[record_id, 2] > 5:
        result20.at[record_id, 2] = 5
    elif result20.at[record_id, 2] < 1:
        result20.at[record_id, 2] = 1

# write prediction result to files
result5 = result5.astype(int)
result10 = result10.astype(int)
result20 = result20.astype(int)
result5.to_csv("result5.txt", sep=" ", header=None, index=None)
result10.to_csv("result10.txt", sep=" ", header=None, index=None)
result20.to_csv("result20.txt", sep=" ", header=None, index=None)
