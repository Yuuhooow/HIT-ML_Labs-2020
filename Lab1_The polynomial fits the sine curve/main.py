import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def sin_func(x):
    return np.sin(2 * np.pi * x)


def data(size):
    x = np.linspace(0, 1, num=size)
    return x.reshape(size, 1)


def create_data(size):
    x = data(size)
    np.random.seed(50)
    y = sin_func(x) + np.random.normal(scale=0.1, size=x.shape)  # 加入噪声
    return x, y

# plt.scatter(x_train,y_train,facecolor="none",edgecolor="green",s=30,label="training data")
# plt.plot(x_test,y_test,c="b",label="sin(2πx)")
# plt.xlabel("x",size=20)
# plt.ylabel("y",size=20)
# plt.legend()
# plt.show()

def create_matrix10(d):  # 构造x的10个样本的d阶多项式的矩阵
    X = np.empty([10, d + 1])
    for row in range(0, 10):
        X[row, d] = 1
        for col in range(d):
            # a = np.logspace(1, 10, base=x_train[row])
            X[row, col] = np.power(x_train[row],col+1)
    return X

def create_matrix100(d):  # 构造x的100个样本的d阶多项式的矩阵
    X = np.empty([100, d + 1])
    for row in range(0, 100):
        X[row, d] = 1
        for col in range(d):
            # a = np.logspace(1, 100, base=x_train[row])
            X[row, col] = np.power(x_test[row],col+1)
    return X

def f1(x_train,y_train,x_test,y_test,d):  #无惩罚项的最小二乘法求解
    X = create_matrix10(d)
    X1 = create_matrix100(d)
    y = y_train.reshape((10,1))
    S1 = np.dot(X.T,X)
    S2 = np.dot(X.T,y)
    w = np.linalg.solve(S1,S2)
    Y = np.dot(X1,w)

    plt.scatter(x_train,y_train,facecolor="none",edgecolor="green",s=30,label="training data")
    plt.plot(x_test,y_test,c="b",label="sin(2πx)")
    plt.plot(x_test,Y,c="red",label="Without penalty term")
    plt.xlabel("x",size=20)
    plt.ylabel("y",size=20)
    plt.title("无惩罚项的最小二乘法图像拟合")
    plt.legend()
    plt.show()

def f2(x_train,y_train,x_test,y_test,d,lemda):  #带惩罚项的最小二乘法求解，惩罚系数为lemda
    X = create_matrix10(d)
    X1 = create_matrix100(d)
    y = y_train.reshape((10,1))
    S1 = np.dot(X.T,X)+10*lemda*np.eye(d+1)
    S2 = np.dot(X.T,y)
    w = np.linalg.solve(S1,S2)
    Y = np.dot(X1,w)

    plt.scatter(x_train,y_train,facecolor="none",edgecolor="green",s=30,label="training data")
    plt.plot(x_test,y_test,c="b",label="sin(2πx)")
    plt.plot(x_test,Y,c="red",label="With penalty term")
    plt.xlabel("x",size=20)
    plt.ylabel("y",size=20)
    plt.title("惩罚系数为{}，阶数为{}的最小二乘法图像拟合".format(str(lemda),str(d)))
    plt.legend()
    plt.show()

def g(x_train,y_train,x_test,y_test,d,a):   #梯度下降法,d为阶数，a为学习率
    X = create_matrix10(d)
    X1 = create_matrix100(d)
    y = y_train.reshape((10, 1))
    w = np.ones([d+1,1])
    E = 0
    for num in range(1,20000000):
        E1=E
        S1 = np.dot(X,w)-y
        S2 = 1/10*np.dot(X.T,S1)
        S3 = -S1
        E = np.dot(S3.T,S3)
        w = w - a*S2
        if (abs(E1-E) < 1e-7):
            break
    Y = np.dot(X1,w)

    plt.scatter(x_train,y_train,facecolor="none",edgecolor="green",s=30,label="training data")
    plt.plot(x_test,y_test,c="b",label="sin(2πx)")
    plt.plot(x_test,Y,c="red",label="Gradient descent method")
    plt.xlabel("x",size=20)
    plt.ylabel("y",size=20)
    plt.title("学习率为{}，阶数为{}的梯度下降法图像拟合".format(str(a),str(d)))
    plt.legend()
    plt.show()

def h(x_train,y_train,x_test,y_test,d):    #共轭梯度法，d为阶数
    X = create_matrix10(d)
    X1 = create_matrix100(d)
    y = y_train.reshape((10, 1))
    A = np.dot(X.T,X)
    b = np.dot(X.T,y)
    w = np.zeros([d+1,1])
    r = b - np.dot(A,w)
    p = r
    for k in range(0,d):
        S1 = np.dot(r.T,p)
        S2 = np.dot(A,p)
        S3 = np.dot(S2.T,p)
        alpha = S1/S3     #优化步长
        w = w + alpha*p
        r = b - np.dot(A,w)
        S4 = np.dot(A,p)
        S5 = np.dot(r.T,S4)
        S6 = np.dot(p.T,S4)
        beta = -S5/S6
        p = r + beta*p    #优化方向

    Y = np.dot(X1,w)

    plt.scatter(x_train,y_train,facecolor="none",edgecolor="green",s=30,label="training data")
    plt.plot(x_test,y_test,c="b",label="sin(2πx)")
    plt.plot(x_test,Y,c="red",label="Conjugate gradient method")
    plt.xlabel("x",size=20)
    plt.ylabel("y",size=20)
    plt.title("阶数为{}的共轭梯度法图像拟合".format(str(d)))
    plt.legend()
    plt.show()


#主函数界面
x_train, y_train = create_data(10)
x_test = data(100)
y_test = sin_func(x_test)

f1(x_train,y_train,x_test,y_test,10)    #无惩罚项的最小二乘法
f2(x_train,y_train,x_test,y_test,5,0.00002)  #带惩罚项的最小二乘法求解，惩罚系数为lemda
g(x_train,y_train,x_test,y_test,7,0.05)    #梯度下降法,d为阶数，a为学习率
h(x_train,y_train,x_test,y_test,9)    #共轭梯度法，d为阶数