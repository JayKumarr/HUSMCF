import numpy as np
import csv

def dict_to_npmatrix(u2u_weight_dic):
    total_users = len(u2u_weight_dic)

    matrixx = np.zeros((total_users, total_users))

    user_index_mapping = []
    user_index_mapping.extend(u2u_weight_dic.keys())

    for user, user_vec in u2u_weight_dic.items():
        u = user_index_mapping.index(user)
        for u2, score in user_vec.items():
            v = user_index_mapping.index(u2)
            matrixx[u][v] = score

    return matrixx


def reading_dataset(file_path, ignore_first_column=True):
    # Executing through CSV
    user_id_column_index = 0
    item_id_column_index = 1
    rating_column_index = 2

    user_item_rating = {}
    with open(file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if (ignore_first_column):
                ignore_first_column = False
                continue
            user = row[user_id_column_index]
            item = row[item_id_column_index]
            rating = float(row[rating_column_index])
            if (user not in user_item_rating):
                user_item_rating[user] = {}
            user_item_rating[user][item] = rating

    return user_item_rating