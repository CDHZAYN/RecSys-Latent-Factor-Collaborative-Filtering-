import numpy as np
# from latent_factor import lf
from collaborative_filter import cf_item, cf_user

if __name__ == '__main__':

    # read files
    ratings = np.loadtxt("ml-latest-small/ratings.csv", delimiter=",", skiprows=1).astype(np.int64)
    movies = np.loadtxt("ml-latest-small/movies.csv", delimiter=",", skiprows=1, usecols=0, encoding="UTF-8").astype(
        np.int64)

    # create mapping for movie
    movie_rev_map = {movies[k]: k for k in range(len(movies))}
    input_matrix = np.zeros((np.amax(ratings[:, 0], axis=0), len(movies)))

    # map user to user - 1
    # map item(id) to item(line number)

    for rating in ratings:
        input_matrix[rating[0] - 1][movie_rev_map[rating[1]]] = rating[2]

    # input matrix mock
    # line: user
    # column: movie
    input_matrix = np.array([[4, 0, 2, 0, 1, 3, 4, 0, 0, 0],
                             [0, 0, 2, 3, 1, 1, 5, 0, 0, 0],
                             [4, 1, 2, 0, 1, 0, 0, 0, 0, 0],
                             [4, 1, 2, 5, 1, 5, 5, 0, 0, 0],
                             [3, 0, 5, 0, 2, 2, 0, 0, 0, 0],
                             [1, 0, 3, 0, 4, 2, 3, 0, 0, 0]]).astype(np.float64)

    rec_part = []
    # compute latent factor using gradient descent
    # rst_matrix_lf = lf(input_matrix)
    # rec_part.append(rst_matrix_lf)
    # print("latent factor done!")

    # compute item_item collaborative_filter
    rst_matrix_cfi = cf_item(input_matrix)
    rec_part.append(rst_matrix_cfi)
    print("collaborative factor for item done!")

    rst_matrix_cfu = cf_user(input_matrix)
    rec_part.append(rst_matrix_cfu)
    print("collaborative factor for user done!")

    score = [5, 4, 3, 2, 2, 1, 1, 1, 1, 1]
    portion = [1, 1]
    rec_result = []
    for user in range(len(rst_matrix_cfu)):
        rec_per_user = np.zeros(len(input_matrix[0]))
        for method in range(0, 3):
            for item_rank in range(0, 10):
                rec_per_user[rec_part[method][user][item_rank]] += score[item_rank]*portion[method]
        rec_result.append(np.argsort(-rec_per_user).tolist()[:3])

    np.savetxt('output.csv', rec_result, delimiter=', ', fmt='%f')