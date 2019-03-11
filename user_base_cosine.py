from math import log10
import pandas as pd


# ---------- User-based Collaborative Filtering (Cosine Similarity) ----------

# ---------- Calculate the average rating of each movie ----------
def calculate_movie_average(training_table):
    movie_average = {}
    for movie_id in range(0, training_table.shape[1]):
        average = 0
        count = 0
        for user_id in range(0, training_table.shape[0]):
            if training_table.at[user_id, movie_id] != 0:
                average += training_table.at[user_id, movie_id]
                count += 1
        if count != 0:
            average /= count
        movie_average[movie_id] = average
    return movie_average


# ---------- Calculate Inverse User Frequency ----------
def calculate_iuf(training_table):
    iuf = []
    for movie_id in range(0, training_table.shape[1]):
        count = 0
        for user_id in range(0, training_table.shape[0]):
            if training_table.at[user_id, movie_id] != 0:
                count += 1
        if count == 0:
            iuf.append(0)
        else:
            iuf.append(log10(training_table.shape[0] / count))
    return iuf


# ---------- Calculate Cosine similarity between users----------
def calculate_weight(training_table, user_rate, amplify, iuf):
    weights = {}
    max_weight = 0
    for user_id in range(0, training_table.shape[0]):
        numerator = 0
        denominator_left = 0
        denominator_right = 0
        for movie_id in user_rate:
            if training_table.at[user_id, movie_id - 1] != 0:
                numerator += training_table.at[user_id, movie_id - 1] * iuf[movie_id - 1] \
                             * user_rate[movie_id] * iuf[movie_id - 1]
                denominator_left += pow(training_table.at[user_id, movie_id - 1] * iuf[movie_id - 1], 2)
                denominator_right += pow(user_rate[movie_id] * iuf[movie_id - 1], 2)
        if numerator != 0 and denominator_left != 0 and denominator_right != 0:
            cur_weight = numerator / (pow(denominator_left, 0.5) * pow(denominator_right, 0.5))
            cur_weight = cur_weight * pow(abs(cur_weight), amplify - 1)
            weights[user_id] = cur_weight
            max_weight = max(max_weight, cur_weight)
    weights[-1] = max_weight
    return weights


# ---------- Predict ratings by similarity and known ratings ----------
def calculate_rate(training_table, weights, movie_id, movie_average, user_average, similarity_threshold):
    max_weights = weights[-1]
    total_weight = 0
    total_rate = 0
    for user_id in weights:
        if user_id != -1 and training_table.at[user_id, movie_id - 1] != 0 \
                and weights[user_id] >= max_weights * similarity_threshold:
            total_rate += training_table.at[user_id, movie_id - 1] * weights[user_id]
            total_weight += weights[user_id]
    if total_weight != 0:
        total_rate /= total_weight
    elif movie_average[movie_id - 1] != 0:
        total_rate = movie_average[movie_id - 1]
    else:
        total_rate = user_average
    return total_rate


# ---------- predict ratings of movies listed in test-table and output the result ----------
def user_based_cosine_predict(training_table, test_table, amplify, similarity_threshold, iuf_boolean):
    result = pd.DataFrame()
    pre_user_id = -1
    user_rate = {}
    weights = {}
    iuf = [1] * training_table.shape[1]
    if iuf_boolean:
        iuf = calculate_iuf(training_table)
    movie_average = calculate_movie_average(training_table)
    user_average = 0
    count = 0
    for user_id in range(0, test_table.shape[0]):
        if test_table.at[user_id, 0] == pre_user_id:
            if test_table.at[user_id, 2] == 0:  # movie need to be rated
                if test_table.at[user_id - 1, 2] != 0:  # only calculate weights when encounter new users
                    weights = calculate_weight(training_table, user_rate, amplify, iuf)
                final_rate = calculate_rate(training_table, weights, test_table.at[user_id, 1], movie_average,
                                            user_average / count, similarity_threshold)
                new_record = pd.Series([test_table.at[user_id, 0], test_table.at[user_id, 1], final_rate])
                result = result.append(new_record, ignore_index=True)
            else:   # movie already been rated
                user_rate[test_table.at[user_id, 1]] = test_table.at[user_id, 2]
                user_average += test_table.at[user_id, 2]
                count += 1
        else:   # new users
            pre_user_id = test_table.at[user_id, 0]
            user_rate.clear()
            user_rate[test_table.at[user_id, 1]] = test_table.at[user_id, 2]
            user_average = test_table.at[user_id, 2]
            count = 1
    return result
