import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats
import math
from mpl_toolkits.mplot3d import Axes3D

COLOR_LIST = ['red', 'blue', 'pink', 'green']

config = {
    'K': 4,
    'dim': 2,
    'n': 50,

    'miu': np.array([[2, 2], [9,9], [9,1], [2, 7]]),
    'sigma': np.array([[[2, 0], [0, 2]],
                       [[3, 0], [0, 3]],
                       [[5, 0], [0, 5]],
                       [[1, 0], [0, 1]]])
}



def get_data(k, dim, n, miu, sigma):
    data = np.zeros((k, n, dim))
    for i in range(k):
        random.seed(2)
        data[i] = np.random.multivariate_normal(miu[i], sigma[i], n)
    data1 = np.zeros((k * n, dim))
    for i in range(k):
        data1[i * n:(i + 1) * n] = data[i]
    return data1


def select_initial_center(data, k, n, dim):
    center = np.zeros((k, dim))
    flags = np.zeros((n, 1))
    for i in range(k):
        # 此处flags避免选择同一初始点
        while True:
            a = np.random.randint(0, n)
            if flags[a] == 0:
                center[i, :] = data[a, :]
                flags[a] = 1
                break
    return center


def new_center(data, k, n, dim, classes):
    new_center = np.zeros((k, dim))
    num = np.zeros(k)
    for i in range(n):
        if classes[i, dim] < k:
            c = int(classes[i, dim])
            new_center[c, :] = new_center[c, :] + classes[i, :dim]
            num[c] += 1
    for i in range(k):
        if num[i] != 0:
            new_center[i, :] /= num[i]
    return new_center


def kmeans(data, k, n, dim):
    classes = np.zeros((n, dim + 1))
    classes[:, 0:dim] = data
    center = select_initial_center(data, k, n, dim)
    while True:
        distance = np.zeros(k)
        for i in range(n):
            for j in range(k):
                distance[j] = np.linalg.norm(data[i, :] - center[j, :])
            arg = np.argmin(distance)
            classes[i, dim] = arg
        p = 0

        newcenter = new_center(data, k, n, dim, classes)
        for i in range(k):
            distance_bias = np.linalg.norm(newcenter[i, :] - center[i, :])
            if distance_bias < 1e-15:
                p += 1
                print('class%d聚类成功' % p)
        print('\n')
        if p == k:
            break
        else:
            center = newcenter
    return classes, center


def k_means(data):
    classes, center = kmeans(data, config['K'], np.size(data, axis=0), config['dim'])
    dim = config['dim']
    n = np.size(data, axis=0)
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for i in range(n):
        if 0 <= i < config['n']:
            plt.scatter(data[i, 0], data[i, 1], c='blue',
                        marker='o')
        elif config['n'] <= i < 2 * config['n']:
            plt.scatter(data[i, 0], data[i, 1], c='red',
                        marker=',')
        elif 2 * config['n'] <= i < 3 * config['n']:
            plt.scatter(data[i, 0], data[i, 1], c='pink',
                        marker='o')
        else:
            plt.scatter(data[i, 0], data[i, 1], c='green',
                        marker=',')
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    for i in range(n):
        if classes[i, dim] == 0:
            if a == 0:
                plt.scatter(classes[i, 0], classes[i, 1], c='blue',
                            marker='o', label='class1')
                a = 1
            if a == 1:
                plt.scatter(classes[i, 0], classes[i, 1], c='blue',
                            marker='o')
        elif classes[i, dim] == 1:
            if b == 0:
                plt.scatter(classes[i, 0], classes[i, 1], c='red',
                            marker=',', label='class2')
                b = 1
            if b == 1:
                plt.scatter(classes[i, 0], classes[i, 1], c='red',
                            marker=',')
        elif classes[i, dim] == 2:
            if c == 0:
                plt.scatter(classes[i, 0], classes[i, 1], c='pink',
                            marker='o', label='class3')
                c = 1
            if c == 1:
                plt.scatter(classes[i, 0], classes[i, 1], c='pink',
                            marker='o')
        elif classes[i, dim] == 3:
            if d == 0:
                plt.scatter(classes[i, 0], classes[i, 1], c='green',
                            marker=',', label='class4')
                d = 1
            if d == 1:
                plt.scatter(classes[i, 0], classes[i, 1], c='green',
                            marker=',')
        elif classes[i, dim] == 4:
            if e == 0:
                plt.scatter(classes[i, 0], classes[i, 1], c='orange',
                            marker='o', label='class5')
                e = 1
            if e == 1:
                plt.scatter(classes[i, 0], classes[i, 1], c='orange',
                            marker='o')

    for i in range(config['K']):
        if i == 0:
            plt.scatter(center[i, 0], center[i, 1], c='black',
                        marker='*', label='center')
        else:
            plt.scatter(center[i, 0], center[i, 1], c='black',
                        marker='*')
    plt.legend(loc='best')
    plt.show()


def EM(data, miu, sigma, n, K, alpha, dim):
    gamma = np.zeros((n, K))
    for j in range(n):
        mixture_model = 0
        for k in range(K):
            mixture_model += alpha[k] * scipy.stats.multivariate_normal.pdf(data[j],
                                                                            miu[k], sigma[k])
        for k in range(K):
            gamma[j][k] = alpha[k] * scipy.stats.multivariate_normal.pdf(data[j],
                                                                         miu[k], sigma[k]) / mixture_model

    miu1 = np.zeros((K, dim))
    sigma1 = np.zeros((K, dim, dim))
    alpha1 = np.zeros(K)
    for k in range(K):
        Nk = 0
        for j in range(n):
            Nk += gamma[j][k]
        miu2 = np.zeros(dim)
        for j in range(n):
            miu2 += gamma[j][k] * data[j]
        miu1[k] = miu2 / Nk

        sigma2 = np.zeros(dim)
        for j in range(n):
            sigma2 += (data[j] - miu[k]) ** 2 * gamma[j][k]
        sigma3 = np.eye(dim)
        sigma3[0, 0] = sigma2[0]
        sigma3[1, 1] = sigma2[1]
        sigma1[k] = sigma3 / Nk

        alpha1[k] = Nk / n
    return miu1, sigma1, alpha1, gamma


def log_likelihood(data, miu, sigma, alpha, n, K):
    it = 0
    for j in range(n):
        temp = 0
        for k in range(K):
            temp += alpha[k] * scipy.stats.multivariate_normal.pdf(data[j], mean=miu[k],
                                                                   cov=sigma[k])
        it += math.log(temp)
    return it


def show_gmm(data, miu, sigma, classes, real_ce):
    fig = plt.figure()
    ax = plt.subplot()
    K = 4
    N = np.size(data, 0)
    for i in range(N):
        plt.scatter(data[i, 0], data[i, 1], marker='.',
                    color=COLOR_LIST[int(classes[i])])
    for j in range(K):
        plt.scatter(miu[j, 0], miu[j, 1], c='black',
                    marker='*')
    plt.show()


def EM_show(data):
    classe, center = kmeans(data, config['K'], np.size(data, axis=0), config['dim'])
    n = np.size(data, axis=0)
    k = config['K']
    dim = config['dim']
    miu = np.zeros((k, dim))
    for i in range(k):
        for j in range(dim):
            miu[i][j] = center[i][j]
    sigma = np.array([[[1, 0], [0, 1]]] * k)
    alpha = np.array([1 / config['K']] * config['K'])
    epoch = 100
    eps = 1e-2
    classes = np.zeros(n)
    for i in range(epoch):
        old_loss = log_likelihood(data, miu, sigma, alpha, n, k)
        miu, sigma, alpha, gamma = EM(data, miu, sigma, n, k, alpha, dim)
        new_loss = log_likelihood(data, miu, sigma, alpha, n, k)
        print(i, new_loss - old_loss)

        if abs(new_loss - old_loss) < eps:
            argmaxs = np.argmax(gamma, axis=1)
            for j in range(n):
                classes[j] = argmaxs[j]
            show_gmm(data, miu, sigma, classes, False)
            break



def show_UCI(data, classes):
    fig = plt.figure()
    ax = Axes3D(fig)
    K = 4
    N = np.size(data, 0)
    for i in range(N):
        ax.scatter(data[i, 0], data[i, 1], data[i, 2], color=COLOR_LIST[int(classes[i])])
    plt.show()


def UCI():
    data = np.loadtxt('uci_data/iris.data', dtype=str, delimiter=',')
    a = data[:, 0:3]
    b = np.array(a, dtype=np.float32)

    N = np.size(b, axis=0)
    K = 4
    dim = np.size(b, axis=1)
    classe, center = kmeans(b, K, N, dim)
    miu = np.array([[0] * dim] * K)
    sigma = np.array([np.eye(dim)] * K)
    temp = np.array([[0] * dim] * K)
    temp_counts = np.array([0] * K)
    for i in range(N):
        c = int(classe[i, -1])
        temp_counts[c] += 1
        for j in range(dim):
            temp[c, j] += classe[i, j]
    for i in range(K):
        for j in range(dim):
            miu[i, j] = temp[i, j] / temp_counts[i]
    for i in range(K):
        for j in range(dim):
            for k in range(N):
                if classe[i, -1] == i:
                    sigma[i, j, j] += pow((classe[i, j] - miu[i, j]), 2)
            sigma[i, j, j] /= temp_counts[i]
    alpha = np.array([1 / K] * K)
    epoch = 100
    eps = 1e-4
    classes = np.zeros(N)
    for i in range(epoch):
        old_loss = log_likelihood(b, miu, sigma, alpha, N, K)
        miu, sigma, alpha, gamma = EM(b, miu, sigma, N, K, alpha, dim)
        new_loss = log_likelihood(b, miu, sigma, alpha, N, K)
        argmaxs = np.argmax(gamma, axis=1)
        for j in range(N):
            classes[j] = argmaxs[j]
        if abs(new_loss - old_loss) < eps:
            show_UCI(b, classes)
            break


data = get_data(config['K'], config['dim'], config['n'], config['miu'], config['sigma'])
k_means(data)
EM_show(data)
UCI()
