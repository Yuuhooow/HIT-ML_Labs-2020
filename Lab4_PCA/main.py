import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


#data：1行为一个特征，1列为一个样本
def PCA(data,k):    #代表被压缩到k维
    mean_data = np.mean(data,axis=0)    #被压缩到1行
    mean_removed = data - mean_data    #去均值
    cov_mean_removed = np.cov(mean_removed)
    eigvals,eigvects = np.linalg.eig(cov_mean_removed)    #特征值和特征向量
    eigvals_Loc = np.argsort(eigvals)
    eigvals_Loc_max_k = eigvals_Loc[:-(k+1):-1]    #返回k个最大特征值下标
    eigvects_max_k = eigvects[:,eigvals_Loc_max_k]     #返回k个最大特征值对应的特征向量
    lower_data = np.dot(eigvects_max_k.T,mean_removed)     #降维后的数据集
    re_data = np.dot(eigvects_max_k,lower_data)+mean_data     #重构原data
    return lower_data,re_data

#二维绘图
def draw0(data):
    rows, cols = data.shape
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(cols):
        ax.scatter(data[0,i],data[1,i],color='red')
    ax.set_title('2D')
    plt.show()
    return

#三维绘图
def draw1(data):
    rows, cols = data.shape
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(cols):
        ax.scatter(data[0,i],data[1,i],data[2,i],color='red')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D')
    plt.show()
    return

#计算峰值信噪比PSNR
def cal_PSNR(data1,data2):
    raws = data1.shape[0]
    cols = data1.shape[1]
    noise = data2 - data1
    sum = 0
    for i in range(raws):
        for j in range(cols):
            sum += np.abs(noise[i][j])
    MSE = sum/(raws*cols)
    PSNR = 20 * np.log10(255/np.sqrt(MSE))
    return np.round(PSNR,2)

# #计算PSNR
# def cal_psnr(img1, img2):
#     mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
#     if mse < 1.0e-10:
#         return 100
#     PIXEL_MAX = 1
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#自己生成数据进行降维测试
def my_PCA():
    mean = (5, 1, 10)
    cov = [[5, 0, 0], [0, 0.01, 0], [0, 0, 6]]
    size = 100
    np.random.seed(0)
    data = np.random.multivariate_normal(mean, cov, size)
    data = data.T
    draw1(data)
    lower_data, re_data = PCA(data, 2)  # 三维降到二维
    draw0(lower_data)
    return

def photo_PCA():
    k = 1
    pra = 1 #图片缩放比例

    img = cv2.imread('1.JPG')
    # img = cv2.imread('2.JPG')
    # img = cv2.imread('3.JPG')
    rows = img.shape[0]
    cols = img.shape[1]
    img_resize = cv2.resize(img,(int(pra*cols),int(pra*rows)))     #经测试，cols和rows的顺序应该是这样,缩放图片
    img_gray = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)     #转换为单通道灰度图
    print(img_gray.shape[0],img_gray.shape[1])
    # cv2.imshow("Original", img_gray)
    # cv2.waitKey(0)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original')
    plt.show()

    Rows = img_gray.shape[0]
    Cols = img_gray.shape[1]
    data = img_gray
    print(data)       #打印原灰度图片矩阵

    lower_data,re_data = PCA(data,k)
    re_data = np.real(re_data)

    #只能取整数，但仍为float64型
    for i in range(Rows):
        for j in range(Cols):
            re_data[i][j] = int(re_data[i][j])
    #特殊方法转换
    re_data = re_data.astype(int)
    print(re_data)

    PSNR = cal_PSNR(data, re_data)
    print()
    print("信噪比：", PSNR)

    plt.imshow(re_data,cmap='gray')
    plt.title("k={},PSNR={}".format(str(k),str(PSNR)))
    plt.show()
    #调cv库有问题
    # cv2.imshow("Restored", re_data)
    # cv2.waitKey(0)

    return


# my_PCA()    #我写的是3维-2维
photo_PCA()