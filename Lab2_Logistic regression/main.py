import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def sig(x): #sigmoid函数
    if x < -10:
        return 0
    else:
        return 1 / (1 + np.exp(-x))

def cal_loss1(X,Y,W):  #损失函数,X.shape[0]即为数据量
    l = 0
    for i in range(X.shape[0]):
        wx = np.dot(X[i],W)
        l = -Y[i]*wx + np.log(1+np.exp(wx))
    loss = l
    return loss

# def cal_loss2(X,Y,W,N,lemda):  #带惩罚的损失函数，有误需改正
#     l = 0
#     for i in range(X.shape[0]):
#         wx = np.dot(X[i], W)
#         l = -Y[i] * wx + np.log(1 + np.exp(wx))
#     loss = l / X.shape[0] + lemda/(2*N) * np.dot(W.T,W)
#     return loss

def cal_gradient1(X,Y,n,a):  #无惩罚项的的梯度下降法求W，a为学习率,n为迭代次数
    W = np.zeros((X.shape[1],1))
    loss = 0
    dw = np.zeros((W.shape[0],1))
    L = np.zeros((n,1))
    for num in range(n):
        loss0 = loss
        for j in range(W.shape[0]):
            for i in range(X.shape[0]):
                dw[j] += X[i][j] * (-Y[i] + sig(np.dot(X[i], W)))  #???
            dw[j] = dw[j]/X.shape[0]
        W = W - a * dw
        # print(loss)
        # print(W)
        loss = cal_loss1(X, Y, W)
        L[num] = loss
        if (abs(loss - loss0) < 1e-10):
            break
    x = np.arange(0,num)
    y = L[x]
    plt.plot(x, y, c="blue",label="loss")
    plt.xlabel("n", size=15)
    plt.ylabel("L", size=15)
    plt.title("无正则项的损失函数")
    plt.legend()
    plt.show()
    return W

def cal_gradient2(X,Y,n,a,lemda):  #有惩罚项的的梯度下降法求W，a为学习率,n为迭代次数,lemda为正则项系数
    W = np.zeros((X.shape[1],1))
    loss = 0
    dw = np.zeros((W.shape[0],1))
    L = np.zeros((n,1))
    for num in range(n):
        loss0 = loss
        for j in range(W.shape[0]):
            for i in range(X.shape[0]):
                dw[j] = dw[j] + X[i][j] * (-Y[i] + sig(np.dot(X[i], W)))
            dw[j] = (dw[j]+lemda*W[j])/X.shape[0] + lemda*dw[j]
        W = W - a * dw
        # print(loss)
        # print(W)
        loss = cal_loss1(X, Y, W) + lemda/(2*X.shape[0]) * np.dot(W.T,W)
        L[num] = loss
        if (abs(loss - loss0) < 1e-10):
            break
    x = np.arange(0,num)
    y = L[x]
    plt.plot(x, y, c="blue",label="loss")
    plt.xlabel("n", size=15)
    plt.ylabel("L", size=15)
    plt.title("带正则项的损失函数")
    plt.legend()
    plt.show()
    return W




#计算匹配率
def cal_ratio(X,W):
    c = 0  #c为匹配成功的样本数
    a = int(X.shape[0]/2)
    for i in range(a):
        if (np.dot(X[i],W) < 0 or np.dot(X[i],W) == 0):
            c += 1
    for i in range(a,X.shape[0]):
        if (np.dot(X[i],W) > 0):
            c += 1
    return c/X.shape[0]




#自己的数据和算法
def my_logistic(N,n,a,lemda):  #N为总数据量，假设0，1数据量相同，各有N/2数据,n为迭代次数，a为学习率,lemda为正则系数
    cov = [[1, 0], [0, 1]]
    # cov = [[2,1], [1,2]]
    mean1 = (1,1)
    mean2 = (3,3)
    size = int(N/2)
    np.random.seed(5)
    train0 = np.random.multivariate_normal(mean1, cov, size)  # y为0
    train1 = np.random.multivariate_normal(mean2, cov, size)  # y为1
    train = np.vstack((train0,train1))
    Train = np.insert(train,2,values=1,axis=1)    #训练集阵，N*3，第3列为1，2维
    y0 = np.zeros((size,1))
    y1 = np.ones((size,1))
    Y = np.vstack((y0,y1))    #特征值矩阵

    #不带惩罚项的梯度下降法
    W1 = cal_gradient1(Train,Y,n,a)
    loss1 = cal_loss1(Train,Y,W1)
    ratio1 = cal_ratio(Train,W1)    #计算匹配率
    print("Matching ratio 1 = ", ratio1)

    x = Train[:,0]
    y1 = (-W1[0]*x-W1[2])/W1[1]
    plt.scatter(train0[:,0],train0[:,1],facecolor = "none",edgecolor = "blue",s = 30,label = "training data 0")
    plt.scatter(train1[:,0],train1[:,1],facecolor = "none",edgecolor= "red",s = 30,label = "training data 1")
    plt.plot(x,y1,c="green",label="Decision Boundary 1")
    plt.title("无正则项，数据量为{}，迭代次数为{}，学习率为{}，匹配率为{}".format(str(N), str(n),str(a),str(ratio1)))
    plt.xlabel("x1", size=15)
    plt.ylabel("x2", size=15)
    plt.legend()
    plt.show()

    #带惩罚项的梯度下降法
    W2 = cal_gradient2(Train,Y,n,a,lemda)
    loss2 = cal_loss1(Train,Y,W2)
    ratio2 = cal_ratio(Train,W2)
    print("Matching ratio 2 = ",ratio2)

    y2 = (-W2[0] * x - W2[2]) / W2[1]
    plt.scatter(train0[:, 0], train0[:, 1], facecolor="none", edgecolor="blue", s=30, label="training data 0")
    plt.scatter(train1[:, 0], train1[:, 1], facecolor="none", edgecolor="red", s=30, label="training data 1")
    plt.plot(x, y2, c="green", label="Decision Boundary 2")
    plt.title("有正则项，数据量{}，迭代次数{}，学习率{}，惩罚项系数{}，匹配率{}".format(str(N), str(n), str(a), str(lemda),str(ratio2)))
    plt.xlabel("x1", size=15)
    plt.ylabel("x2", size=15)
    plt.legend()
    plt.show()


#UCI实际数据测试
def UCI_logistic(n,a,lemda):    #n为迭代次数，a为学习率,lemda为正则系数
    # data = np.loadtxt("cmc.data",delimiter=',')   #对该数据进行测试时，dw手动修改loss以避免溢出
    data = np.loadtxt("data_banknote_authentication.txt", delimiter=',')
    x = data[:,:data.shape[1]-1]
    X = np.insert(x,x.shape[1],values=1,axis=1)
    Y = data[:,data.shape[1]-1]

    #无正则项
    W_UCI1 = cal_gradient1(X, Y, n, a)
    loss_UCI1 = cal_loss1(X, Y, W_UCI1)
    ratio_UCI1 = cal_ratio(X, W_UCI1)
    print("Matching ratio UCI 1 = ", ratio_UCI1)

    #有正则项
    W_UCI2 = cal_gradient2(X,Y,n,a,lemda)
    loss_UCI2 = cal_loss1(X,Y,W_UCI2)
    ratio_UCI2= cal_ratio(X,W_UCI2)
    print("Matching ratio UCI 2 = ", ratio_UCI2)




#主函数界面
my_logistic(500,200,0.1,0.01)
UCI_logistic(200,0.01,0.01)
