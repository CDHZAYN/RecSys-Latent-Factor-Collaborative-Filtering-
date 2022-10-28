import numpy as np


# 核心算法
def gd(input_matrix, degree=5, loop_range=50000, step_width=0.0002, lambda2=0.002):
    # 基本维度参数定义
    user_num = len(input_matrix)
    item_num = len(input_matrix[0])

    # user_latent、item_latent初始值，随机生成
    user_latent = np.random.rand(user_num, degree)
    item_latent = np.random.rand(item_num, degree)
    item_latent = item_latent.T

    output_matrix = 0

    # 开始迭代
    for step in range(loop_range):
        # 对所有的用户u、物品i做遍历，对应的特征向量Pu，Qi梯度下降
        for u in range(user_num):
            for i in range(item_num):
                # 对于每一个大于0的评分，求出预测的评分误差
                if input_matrix[u][i] > 0:
                    loss = np.dot(user_latent[u, :], item_latent[:, i]) - input_matrix[u][i]

                    # 带入公式，按照梯度下降算法更新当前的Pu与Qi
                    for k in range(degree):
                        user_latent[u][k] = user_latent[u][k] - step_width * (2 * loss * item_latent[k][i] + 2 * lambda2 * user_latent[u][k])
                        item_latent[k][i] = item_latent[k][i] - step_width * (2 * loss * user_latent[u][k] + 2 * lambda2 * item_latent[k][i])

        # u、i遍历完成，所有的特征向量更新完成，可以得到P、item_latent，可以计算预测评分矩阵
        output_matrix = np.dot(user_latent, item_latent)

        # 计算当前损失函数
        cost = 0
        for u in range(user_num):
            for i in range(item_num):
                if input_matrix[u][i] > 0:
                    cost += (np.dot(user_latent[u, :], item_latent[:, i]) - input_matrix[u][i]) ** 2
                    # 加上正则化项
                    for k in range(degree):
                        cost += lambda2 * (user_latent[u][k] ** 2 + item_latent[k][i] ** 2)

        if cost < 0.001:
            break

    return output_matrix


def lf(matrix):
    output_matrix = gd(matrix)
    output_matrix = np.subtract(output_matrix, matrix)
    rec_ranks = []
    for user in output_matrix:
        rec_ranks.append(np.argsort(-user).tolist()[:10])
    return rec_ranks
