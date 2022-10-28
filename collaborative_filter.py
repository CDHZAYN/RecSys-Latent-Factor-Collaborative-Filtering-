import numpy as np
import pandas as pd


def cf(matrix):
    this_matrix = matrix.copy()
    corr_result = np.zeros((len(this_matrix), len(this_matrix)))
    ranks = []
    for line in this_matrix:
        line -= np.full(len(line), np.mean(line))
    matrix_df = pd.DataFrame(this_matrix)
    for i in range(len(matrix_df)):
        for t in range(0, i):
            corr_result[i][t] = corr_result[t][i]
        for j in range(i, len(matrix_df)):
            if i == j:
                corr_result[i][j] = -1.0
            else:
                corr_result[i][j] = matrix_df.loc[i].corr(matrix_df.loc[j], method="pearson")
    for i in range(len(matrix_df)):
        ranks.append(np.argsort(-corr_result[i]).tolist())
    return ranks


def cf_item(matrix):
    simi_ranks = cf(matrix.T)
    # print(simi_ranks)
    rec_ranks = []
    for user in matrix:
        rec_rate = np.zeros(len(matrix[0]))
        user_ranks = np.argsort(-user)
        for item in user_ranks:
            if user[item] != user[user_ranks[0]]:
                break
            for simi_item in simi_ranks[item][0:3]:
                # print("user", user, "considering", item, "select", simi_item)
                if user[simi_item] == 0:
                    rec_rate[simi_item] += 1
        rec_ranks.append(np.argsort(-rec_rate).tolist()[:10])
    return rec_ranks


def cf_user(matrix):
    simi_ranks = cf(matrix)
    # print(simi_ranks)
    rec_ranks = []
    for user in range(len(matrix)):
        rec_rate = np.zeros(len(matrix[0]))
        for simi_user in matrix[simi_ranks[user][0:3]]:
            user_ranks = np.argsort(-simi_user)
            for item in user_ranks:
                if simi_user[item] != simi_user[user_ranks[0]]:
                    break
                # print("user", user, "similar to", simi_user, "select", item)
                if matrix[user][item] == 0:
                    rec_rate[item] += 1
        rec_ranks.append(np.argsort(-rec_rate).tolist()[:10])
    return rec_ranks

