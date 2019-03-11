import pandas as pd
import os
from random import randint
from user_base_cosine import user_based_cosine_predict
from user_base_pearson import user_based_pearson_predict
from item_base_cosine import item_based_cosine_predict
from item_base_adj_cosine import item_based_adj_cosine_predict
from customized_algorithm import customized_predict


# ------Separate origin training dataset into new training dataset and validation dataset for accuracy evaluation------
def validation_file_generator(absolute_path, user_count):
    training = pd.read_csv(absolute_path + "/data/train.txt", sep="\s+", header=None)
    test5 = open(absolute_path + "/validate/test5.txt", "w")
    test10 = open(absolute_path + "/validate/test10.txt", "w")
    test20 = open(absolute_path + "/validate/test20.txt", "w")
    validate5 = open(absolute_path + "/validate/validate5.txt", "w")
    validate10 = open(absolute_path + "/validate/validate10.txt", "w")
    validate20 = open(absolute_path + "/validate/validate20.txt", "w")
    new_user_id = training.shape[0] - user_count + 1
    while user_count > 0:
        user_id = randint(0, training.shape[0] - 1)
        while not rate_counter(training, user_id):
            user_id = randint(0, training.shape[0] - 1)
        movie_count = 0
        for movie_id in range(0, training.shape[1]):
            if training.at[user_id, movie_id] != 0:
                if movie_count < 5:
                    test5.write(str(new_user_id) + " " + str(movie_id + 1)
                                + " " + str(training.at[user_id, movie_id]) + "\n")
                    test10.write(str(new_user_id) + " " + str(movie_id + 1)
                                 + " " + str(training.at[user_id, movie_id]) + "\n")
                    test20.write(str(new_user_id) + " " + str(movie_id + 1)
                                 + " " + str(training.at[user_id, movie_id]) + "\n")
                elif movie_count < 10:
                    test5.write(str(new_user_id) + " " + str(movie_id + 1)
                                + " 0\n")
                    validate5.write(str(new_user_id) + " " + str(movie_id + 1)
                                    + " " + str(training.at[user_id, movie_id]) + "\n")
                    test10.write(str(new_user_id) + " " + str(movie_id + 1)
                                 + " " + str(training.at[user_id, movie_id]) + "\n")
                    test20.write(str(new_user_id) + " " + str(movie_id + 1)
                                 + " " + str(training.at[user_id, movie_id]) + "\n")
                elif movie_count < 20:
                    test5.write(str(new_user_id) + " " + str(movie_id + 1)
                                + " 0\n")
                    validate5.write(str(new_user_id) + " " + str(movie_id + 1)
                                    + " " + str(training.at[user_id, movie_id]) + "\n")
                    test10.write(str(new_user_id) + " " + str(movie_id + 1)
                                 + " 0\n")
                    validate10.write(str(new_user_id) + " " + str(movie_id + 1)
                                     + " " + str(training.at[user_id, movie_id]) + "\n")
                    test20.write(str(new_user_id) + " " + str(movie_id + 1)
                                 + " " + str(training.at[user_id, movie_id]) + "\n")
                else:
                    test5.write(str(new_user_id) + " " + str(movie_id + 1)
                                + " 0\n")
                    validate5.write(str(new_user_id) + " " + str(movie_id + 1)
                                    + " " + str(training.at[user_id, movie_id]) + "\n")
                    test10.write(str(new_user_id) + " " + str(movie_id + 1)
                                 + " 0\n")
                    validate10.write(str(new_user_id) + " " + str(movie_id + 1)
                                     + " " + str(training.at[user_id, movie_id]) + "\n")
                    test20.write(str(new_user_id) + " " + str(movie_id + 1)
                                 + " 0\n")
                    validate20.write(str(new_user_id) + " " + str(movie_id + 1)
                                     + " " + str(training.at[user_id, movie_id]) + "\n")
                movie_count += 1
        training = training.drop([user_id])
        training = training.reset_index(drop=True)
        user_count -= 1
        new_user_id += 1
    test5.close()
    test10.close()
    test20.close()
    validate5.close()
    validate10.close()
    validate20.close()
    training.to_csv(absolute_path + "/validate/train.txt", sep=" ", header=None, index=None)
    return training


# ---Use this function to ensure in validation files, picked user_id contains enough movie rating data----
def rate_counter(training, user_id):
    count = 0
    for movie_id in range(0, training.shape[1]):
        if training.at[user_id, movie_id] != 0:
            count += 1
            if count > 30:
                return True
    return False


# -------------------------------Calculate prediction error of validation---------------------------------
def predict_score(absolute_path, validate_path, result_path):
    validate_table = pd.read_csv(absolute_path + validate_path, sep="\s+", header=None)
    result_table = pd.read_csv(absolute_path + result_path, sep="\s+", header=None)
    rate_sum = 0
    for rate in range(0, validate_table.shape[0]):
        rate_sum += pow(validate_table.at[rate, 2] - result_table.at[rate, 2], 2)
    return pow(rate_sum / validate_table.shape[0], 0.5)


# ---------- Main ----------
# Set validation times, randomly generate validation set each time and take average of all predictions
validate_times = 1
uc_dic = {}
up_dic = {}
ic_dic = {}
ia_dic = {}
for validate_time in range(0, validate_times, 1):

    # generated a new group of training data and validation data
    path = os.path.abspath(os.path.dirname('cross_validation.py'))
    # training_table = validation_file_generator(path, 33)
    training_table = pd.read_csv(path + "/validate/train.txt", sep="\s+", header=None)
    testing5_table = pd.read_csv(path + "/validate/test5.txt", sep="\s+", header=None)
    testing10_table = pd.read_csv(path + "/validate/test10.txt", sep="\s+", header=None)
    testing20_table = pd.read_csv(path + "/validate/test20.txt", sep="\s+", header=None)

    # try each combination of parameters
    for similarity_threshold in ([0]):
        for amplify in ([1]):
            for inverse_frequency in ([False]):
                for method in (["ca"]):

                    print(str(validate_time) + " " + method + " " + str(similarity_threshold) + " " + str(amplify)
                          + " " + str(inverse_frequency))

                    if method == "uc":
                        # ---------- User-based Collaborative Filtering (Cosine Similarity) ----------
                        result5 = user_based_cosine_predict(training_table, testing5_table, amplify,
                                                            similarity_threshold, inverse_frequency)
                        result10 = user_based_cosine_predict(training_table, testing10_table, amplify,
                                                             similarity_threshold, inverse_frequency)
                        result20 = user_based_cosine_predict(training_table, testing20_table, amplify,
                                                             similarity_threshold, inverse_frequency)

                    elif method == "up":
                        # ---------- User-based Collaborative Filtering (Pearson Correlation) --------
                        result5 = user_based_pearson_predict(training_table, testing5_table, amplify,
                                                             similarity_threshold, inverse_frequency)
                        result10 = user_based_pearson_predict(training_table, testing10_table, amplify,
                                                              similarity_threshold, inverse_frequency)
                        result20 = user_based_pearson_predict(training_table, testing20_table, amplify,
                                                              similarity_threshold, inverse_frequency)

                    elif method == "ic":
                        # ---------- Item-based Collaborative Filtering (Cosine Similarity) ----------
                        result5 = item_based_cosine_predict(training_table, testing5_table, amplify,
                                                            similarity_threshold, inverse_frequency)
                        result10 = item_based_cosine_predict(training_table, testing10_table, amplify,
                                                             similarity_threshold, inverse_frequency)
                        result20 = item_based_cosine_predict(training_table, testing20_table, amplify,
                                                             similarity_threshold, inverse_frequency)

                    elif method == "ia":
                        # ---------- Item-based Collaborative Filtering (Pearson Similarity) ----------
                        result5 = item_based_adj_cosine_predict(training_table, testing5_table, amplify,
                                                                similarity_threshold, inverse_frequency)
                        result10 = item_based_adj_cosine_predict(training_table, testing10_table, amplify,
                                                                 similarity_threshold, inverse_frequency)
                        result20 = item_based_adj_cosine_predict(training_table, testing20_table, amplify,
                                                                 similarity_threshold, inverse_frequency)
                    else:
                        # ------------------------- Customized Algorithm ------------------------------
                        result = customized_predict(training_table, testing5_table, testing10_table, testing20_table)
                        result5 = result["result5"]
                        result10 = result["result10"]
                        result20 = result["result20"]

                    # Convert ratings to integers with value within 1 ~ 5
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

                    # Write prediction result to files
                    result5 = result5.astype(int)
                    result10 = result10.astype(int)
                    result20 = result20.astype(int)
                    result5.to_csv("result5.txt", sep=" ", header=None, index=None)
                    result10.to_csv("result10.txt", sep=" ", header=None, index=None)
                    result20.to_csv("result20.txt", sep=" ", header=None, index=None)

                    # Get score of prediction
                    score_5 = predict_score(path, "/validate/validate5.txt", "/result5.txt")
                    score_10 = predict_score(path, "/validate/validate10.txt", "/result10.txt")
                    score_20 = predict_score(path, "/validate/validate20.txt", "/result20.txt")
                    final_score = (score_5 + score_10 + score_20) / 3
                    score = [score_5, score_10, score_20, final_score]
                    print(score)

                    # Add all prediction results for averaging purpose
                    if method == "uc":
                        if (similarity_threshold, amplify, inverse_frequency) not in uc_dic:
                            uc_dic[similarity_threshold, amplify, inverse_frequency] = [0, 0, 0, 0]
                        for i in range(0, 4, 1):
                            uc_dic[similarity_threshold, amplify, inverse_frequency][i] += score[i]

                    elif method == "up":
                        if (similarity_threshold, amplify, inverse_frequency) not in up_dic:
                            up_dic[similarity_threshold, amplify, inverse_frequency] = [0, 0, 0, 0]
                        for i in range(0, 4, 1):
                            up_dic[similarity_threshold, amplify, inverse_frequency][i] += score[i]

                    elif method == "ic":
                        if (similarity_threshold, amplify, inverse_frequency) not in ic_dic:
                            ic_dic[similarity_threshold, amplify, inverse_frequency] = [0, 0, 0, 0]
                        for i in range(0, 4, 1):
                            ic_dic[similarity_threshold, amplify, inverse_frequency][i] += score[i]

                    elif method == "ia":
                        if (similarity_threshold, amplify, inverse_frequency) not in ia_dic:
                            ia_dic[similarity_threshold, amplify, inverse_frequency] = [0, 0, 0, 0]
                        for i in range(0, 4, 1):
                            ia_dic[similarity_threshold, amplify, inverse_frequency][i] += score[i]

# output prediction error to files
uc_result = open("uc_result.txt", "w")
up_result = open("up_result.txt", "w")
ic_result = open("ic_result.txt", "w")
ia_result = open("ia_result.txt", "w")
for key in ia_dic.keys():
    for attribute in key:
        uc_result.write(str(attribute) + " ")
        up_result.write(str(attribute) + " ")
        ic_result.write(str(attribute) + " ")
        ia_result.write(str(attribute) + " ")
    for i in range(0, 4, 1):
        uc_dic[key][i] /= validate_times
        uc_result.write(str(uc_dic[key][i]) + " ")
        up_dic[key][i] /= validate_times
        up_result.write(str(up_dic[key][i]) + " ")
        ic_dic[key][i] /= validate_times
        ic_result.write(str(ic_dic[key][i]) + " ")
        ia_dic[key][i] /= validate_times
        ia_result.write(str(ia_dic[key][i]) + " ")
    uc_result.write("\n")
    up_result.write("\n")
    ic_result.write("\n")
    ia_result.write("\n")
uc_result.close()
up_result.close()
ic_result.close()
ia_result.close()
