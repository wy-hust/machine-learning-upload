import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

theta0 = 0
theta1 = 0
theta2 = 0

#theta0 = 4.161851 theta1 = 4.022459 theta2 = -3.769521


# 读取文件内容


def openfile(road):
    data = np.loadtxt(open(road), delimiter=",", skiprows=1)
    return data

# 拟合的函数计算


def hypothesis(x1, x2):
    global theta0
    global theta1
    global theta2
    y = theta0 + theta1*x1 + theta2 * x2
    return y


# square  h(θ)-y


def Jtheta(x1, x2, y):
    ans = 0
    for i in range(len(x1)):
        ans = ans + ((hypothesis(x1[i], x2[i])-y[i])
                     * (hypothesis(x1[i], x2[i])-y[i]))
    return ans/2


def sum(x1, x2, y):
    summary0 = 0
    summary1 = 0
    summary2 = 0
    for i in range(len(x1)):
        summary0 = summary0 + (hypothesis(x1[i], x2[i])-y[i])*1
        summary1 = summary1 + (hypothesis(x1[i], x2[i])-y[i])*x1[i]
        summary2 = summary2 + (hypothesis(x1[i], x2[i])-y[i])*x2[i]
    return summary0, summary1, summary2


def descent0(x1, x2, y):
    global theta0
    alpha = 0.00001
    sum0, sum1, sum2 = sum(x1, x2, y)
    theta0 = theta0 - alpha*sum0


def descent1(x1, x2, y):
    global theta1
    alpha = 0.00000009
    sum0, sum1, sum2 = sum(x1, x2, y)
    theta1 = theta1 - alpha*sum1


def descent2(x1, x2, y):
    global theta2
    alpha = 0.0000005
    sum0, sum1, sum2 = sum(x1, x2, y)
    theta2 = theta2 - alpha*sum2


if __name__ == "__main__":
    road = 'E:\Machine learning\线性回归\Dataset\mlm.csv'
    data = openfile(road)
    x1 = data[:, 0]
    x2 = data[:, 1]
    y = data[:, 2]
    k = 0
    count = 0
    Jtheta1 = Jtheta(x1, x2, y)
    descent0(x1, x2, y)
    descent1(x1, x2, y)
    descent2(x1, x2, y)
    while Jtheta(x1, x2, y)-Jtheta1 < -0.01:

        Jtheta1 = Jtheta(x1, x2, y)
        descent0(x1, x2, y)
        descent1(x1, x2, y)
        descent2(x1, x2, y)
        count += 1
        print("theta0 = %f theta1 = %f theta2 = %f" %(theta0, theta1, theta2))
        print("Jtheta = %lf" % Jtheta(x1, x2, y))
        print("循环次数：%d" % count)
        
    all_piancha = 0
    for i in range(1000):
        #print("hy = %f,y = %f ,偏差率为 %f " % (hypothesis(x1[i], x2[i]), y[i], (hypothesis(x1[i], x2[i])-y[i])/y[i]))
        all_piancha = all_piancha + abs((hypothesis(x1[i], x2[i])-y[i])/y[i])

    print("平均偏差率为：%f" % (all_piancha/1000))

    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, x2, y)

    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('X2', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X1', fontdict={'size': 15, 'color': 'red'})

    X, Y = np.meshgrid(x1, x2)
    Z = theta1*X + theta2*Y + theta0

    surf = ax.plot_surface(X, Y, Z)
    plt.show()
