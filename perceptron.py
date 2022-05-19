import numpy as np
import matplotlib.pyplot as plt
#matplotlib.pyplot是使matplotlib像MATLAB一样工作的命令样式函数的集合。每个pyplot功能都会对图形进行一些更改：例如，创建图形，在图形中创建绘图区域，在绘图区域中绘制一些线条，用标签装饰绘图等。



#随机生成W
#np.random.seed(10)
#np.random.seed(15)
np.random.seed(20)
w0 = np.random.uniform(-0.25,0.25)
w1 = np.random.uniform(-1,1)
w2 = np.random.uniform(-1,1)
w = [w0,w1,w2]
print(w)

#绘制期望图


#100 point
points = np.random.uniform(-1,1,200)
one = np.ones((100,1))

points = points.reshape(-1,2)
v = np.append(one, points, axis = 1)

#1000个点
#points = np.random.uniform(-1,1,2000)
#one = np.ones((1000,1))

#points = points.reshape(-1,2)
#v = np.append(one, points, axis = 1)

s1 = []
s2 = []
y1= []
for i in range(len(v)):
    if np.dot(v[i],w) >=0:
        s1.append(list(points[i]))
        y1.append(1)
    else:
        s2.append(list(points[i]))
        y1.append(0)

plt.plot(np.dot(s1,[1,0]), np.dot(s1,[0,1]), '.', color='red');
plt.plot(np.dot(s2,[1,0]), np.dot(s2,[0,1]), 'D', color='blue');
x = np.linspace(-1,1,100)
y = -w0/w2 - w1/w2*x
plt.plot(x, y, '-r')
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


#初始输入：training parameter η = 1；计数number of Missclassification=0随机初始化一个W'=uniformly at random on [−1, 1]

# eta = 1 condition
time = 20
e = 1
w0_m = np.random.uniform(-1, 1)
w1_m = np.random.uniform(-1, 1)
w2_m = np.random.uniform(-1, 1)
w_m = [w0_m, w1_m, w2_m]
mis_label1 = []
count = 0
for i in range(len(s1)):
    if ((w0_m + w1_m * s1[i][0] + w2_m * s1[i][1]) < 0):
        count += 1;
for i in range(len(s2)):
    if ((w0_m + w1_m * s2[i][0] + w2_m * s2[i][1]) >= 0):
        count += 1;
mis_label1.append(count)


def step(a, b, c, x):
    if ((a + b * x[1] + c * x[2]) >= 0):
        return 1
    else:
        return 0


for j in range(time):
    # training
    for i in range(len(y1)):
        count = 0
        Y = step(w0_m, w1_m, w2_m, v[i])
        w0_m = w0_m + e * (y1[i] - Y)
        w1_m = w1_m + e * v[i][1] * (y1[i] - Y)
        w2_m = w2_m + e * v[i][2] * (y1[i] - Y)
    for k in range(len(s1)):
        if ((w0_m + w1_m * s1[k][0] + w2_m * s1[k][1]) < 0):
            count += 1;
    for k in range(len(s2)):
        if ((w0_m + w1_m * s2[k][0] + w2_m * s2[k][1]) >= 0):
            count += 1;
    mis_label1.append(count)
w11 = [w0_m, w1_m, w2_m]

# eta = 10 condition
e = 10
w0_m = w_m[0]
w1_m = w_m[1]
w2_m = w_m[2]
mis_label2 = []

for i in range(len(s1)):
    if ((w0_m + w1_m * s1[i][0] + w2_m * s1[i][1]) < 0):
        count += 1;
for i in range(len(s2)):
    if ((w0_m + w1_m * s2[i][0] + w2_m * s2[i][1]) >= 0):
        count += 1;
mis_label2.append(count)


def step(a, b, c, x):
    if ((a + b * x[1] + c * x[2]) >= 0):
        return 1
    else:
        return 0


for j in range(time):
    # training
    for i in range(len(y1)):
        count = 0
        Y = step(w0_m, w1_m, w2_m, v[i])
        w0_m = w0_m + e * (y1[i] - Y)
        w1_m = w1_m + e * v[i][1] * (y1[i] - Y)
        w2_m = w2_m + e * v[i][2] * (y1[i] - Y)
    for k in range(len(s1)):
        if ((w0_m + w1_m * s1[k][0] + w2_m * s1[k][1]) < 0):
            count += 1;
    for k in range(len(s2)):
        if ((w0_m + w1_m * s2[k][0] + w2_m * s2[k][1]) >= 0):
            count += 1;
    mis_label2.append(count)
w22 = [w0_m, w1_m, w2_m]

# eta = 0.1 condition
e = 0.1
w0_m = w_m[0]
w1_m = w_m[1]
w2_m = w_m[2]
mis_label3 = []

for i in range(len(s1)):
    if ((w0_m + w1_m * s1[i][0] + w2_m * s1[i][1]) < 0):
        count += 1;
for i in range(len(s2)):
    if ((w0_m + w1_m * s2[i][0] + w2_m * s2[i][1]) >= 0):
        count += 1;
mis_label3.append(count)


def step(a, b, c, x):
    if ((a + b * x[1] + c * x[2]) >= 0):
        return 1
    else:
        return 0


for j in range(time):
    # training
    for i in range(len(y1)):
        count = 0
        Y = step(w0_m, w1_m, w2_m, v[i])
        w0_m = w0_m + e * (y1[i] - Y)
        w1_m = w1_m + e * v[i][1] * (y1[i] - Y)
        w2_m = w2_m + e * v[i][2] * (y1[i] - Y)
    for k in range(len(s1)):
        if ((w0_m + w1_m * s1[k][0] + w2_m * s1[k][1]) < 0):
            count += 1;
    for k in range(len(s2)):
        if ((w0_m + w1_m * s2[k][0] + w2_m * s2[k][1]) >= 0):
            count += 1;
    mis_label3.append(count)
w33 = [w0_m, w1_m, w2_m]

plt.plot(mis_label3, color='red', label='eta = 0.1')
plt.plot(mis_label1, color='orange', label='eta = 1')
plt.plot(mis_label2, color='black', label='eta = 10')
plt.xlabel('epoch times')
plt.ylabel('mislabeled point')
plt.legend()
plt.show()
print("reference  w0=", w[0],'s1=',w[1],'w2=',w[2])
print("η=0.1 w0=", w33[0], 's1=', w33[1], 'w2=', w33[2])
print("η=1  w0=", w11[0], 's1=', w11[1], 'w2=', w11[2])
print("η=10 w0=", w22[0], 's1=', w22[1], 'w2=', w22[2])





