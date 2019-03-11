from user_base_cosine import user_based_cosine_predict
from user_base_pearson import user_based_pearson_predict
from item_base_cosine import item_based_cosine_predict
from item_base_adj_cosine import item_based_adj_cosine_predict


# ---------- Customized Algorithm of combining user-based and item-based algorithms ----------
def customized_predict(training_table, testing5_table, testing10_table, testing20_table):

    # Get prediction of user-based and item-based algorithms
    result5uc = user_based_cosine_predict(training_table, testing5_table, 3, 0.6, False)
    result10uc = user_based_cosine_predict(training_table, testing10_table, 3, 0.6, False)
    result20uc = user_based_cosine_predict(training_table, testing20_table, 3, 0.6, False)

    result5up = user_based_pearson_predict(training_table, testing5_table, 1, 0, False)
    result10up = user_based_pearson_predict(training_table, testing10_table, 1, 0, False)
    result20up = user_based_pearson_predict(training_table, testing20_table, 1, 0, False)

    result5ic = item_based_cosine_predict(training_table, testing5_table, 1, 0, True)
    result10ic = item_based_cosine_predict(training_table, testing10_table, 1, 0, True)
    result20ic = item_based_cosine_predict(training_table, testing20_table, 1, 0, True)

    result5ia = item_based_adj_cosine_predict(training_table, testing5_table, 1, 0, False)
    result10ia = item_based_adj_cosine_predict(training_table, testing10_table, 1, 0, False)
    result20ia = item_based_adj_cosine_predict(training_table, testing20_table, 1, 0, False)

    # Sum prediction and take average
    result5 = result5uc
    result10 = result10uc
    result20 = result20uc
    for record_id in range(result5.shape[0]):
        result5.at[record_id, 2] += result5up.at[record_id, 2]
        result5.at[record_id, 2] /= 2
    for record_id in range(result10.shape[0]):
        result10.at[record_id, 2] += result10ic.at[record_id, 2] + result10ia.at[record_id, 2]
        result10.at[record_id, 2] /= 3
    for record_id in range(result20.shape[0]):
        result20.at[record_id, 2] += result20up.at[record_id, 2] + result20ic.at[record_id, 2] \
                                     + result20ia.at[record_id, 2]
        result20.at[record_id, 2] /= 4
    result = {"result5": result5, "result10": result10, "result20": result20}
    return result
