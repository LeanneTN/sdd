#深度学习结果
#画对比条形图（测试集上不同模型各指标对比）、折线图（验证集上不同模型各指标随训练轮数的变化）
#我们之后应该要按照自己用到的算法调整
import json
from pylab import *
import matplotlib
import matplotlib.pyplot as plt

#各模型的不同评价指标结果文件
# filename1 = '..\\log\\test.json'
filename2 = '..\\log\\attention_simple_relu.json'
filename3 = '..\\log\\attention_simple_sigmoid.json'
filename4 = '..\\log\\baseline.json'
filename5 = '..\\log\\vae.json'

def readfile():
    data = []
    with open(filename2) as f2:
        f2 = json.load(f2)
    data.append(f2)
    with open(filename3) as f3:
        f3 = json.load(f3)
    data.append(f3)
    with open(filename4) as f4:
        f4 = json.load(f4)
    data.append(f4)
    with open(filename5) as f5:
        f5 = json.load(f5)
    data.append(f5)
    return data

#折线图里用（绘制20个折点，共学习1000轮，所以每50个数绘一个点）
#返回折点值数组
def average(data):
    y = []
    sum1 = 0
    num = 0
    for i in data:
        #循环第40~49次的数据平均数放入y数组，重新计数往下循环
        if 40 <= num <= 49:
            sum1 += i
        num += 1
        if num == 50:
            y.append(sum1/10)
            num = 0
            sum1 = 0
    return y

#折线图（不同模型各指标随训练轮数的变化）
def draw_val(data, type):
    plt.figure(figsize=(8, 6))
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    x = range(0, 20)

    #y：四种模型的“type”指标的值处理后的折点值数组
    y1 = average(data[0]['val'][type])
    print(type,y1)
    y2 = average(data[1]['val'][type])
    y3 = average(data[2]['val'][type])
    y4 = data[3]['val'][type] #这个就训练20代，无需处理
    print(type, y4)

    plt.plot(x, y1, marker='o', ms=5, label="AttentionSimpleRelu")
    plt.plot(x, y2, marker='v', ms=5, label="AttentionSimpleSigmoid")
    plt.plot(x, y3, marker='s', ms=5, label="Baseline")
    plt.plot(x, y4, marker='x', ms=5, label="VAE")
    plt.xticks(rotation=45)  #rotation代表lable显示的旋转角度
    plt.xlabel("Epoch(round)")  #x轴的label

    if type == 'acc':
        yl = 'Accuracy'
    elif type == 'precision':
        yl = 'Precision'
    elif type == 'recall':
        yl = 'Recall'
    else:
        print("ZB吧？")

    plt.ylabel("Validation "+yl)  #y轴的label显示评价指标名称

    plt.legend(loc="center right", prop={'family': 'Times New Roman', 'size': 14})
    xmajorLocator = plt.MultipleLocator(5)  # 将x主刻度标签设置为20的倍数
    xmajorFormatter = plt.FormatStrFormatter('%5d')  # 设置x轴标签文本的格式
    xminorLocator = plt.MultipleLocator(1)  # 将x轴次刻度标签设置为5的倍数
    ymajorLocator = plt.MultipleLocator(0.1)  # 将y轴主刻度标签设置为0.5的倍数
    ymajorFormatter = plt.FormatStrFormatter('%1.3f')  # 设置y轴标签文本的格式
    yminorLocator = plt.MultipleLocator(0.025)  # 将此y轴次刻度标签设置为0.1的倍数
    ax = subplot(111)  # 注意:一般都在ax中设置,不再plot中设置
    # 设置主刻度标签的位置,标签文本的格式
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    # 显示次刻度标签的位置,没有标签文本
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.grid(True, which='major', ls='--')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major', ls='--')  # y坐标轴的网格使用次刻度
    ax.yaxis.grid(True, which='minor', ls='--')  # y坐标轴的网格使用次刻度

    plt.savefig("img/deep_val_"+type+".jpg")  #保存图片
    plt.show()

#条形图（测试集上不同模型各指标对比）
def draw_test(data, type):
    plt.figure(figsize=(8, 9))
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    x = ["AttentionSimpleRelu", "AttentionSimpleSigmoid", "Baseline", "VAE"]

    y1 = data[0]['test'][type]
    y2 = data[1]['test'][type]
    y3 = data[2]['test'][type]
    y4 = data[3]['test'][type]

    y = [y1, y2, y3, y4]
    plt.bar(x, y, color=['#67a3cc', '#ff7f0e', '#2ca02c', '#db4344'])  #颜色可以适当改改
    for a, b in zip(x, y):  #利用zip函数将两个列表(list)组成字典(dict)
        plt.text(a, b, '%.5f' % b, ha='center', va='center', fontsize=10)
    plt.xticks(rotation=45)
    plt.xlabel("Epoch(round)")

    if type == 'acc':
        yl = 'Accuracy'
    elif type == 'precision':
        yl = 'Precision'
    elif type == 'recall':
        yl = 'Recall'
    elif type == 'f1':
        yl = 'F1-Score'
    else:
        print("ZB吧？")

    plt.ylabel("Test " + yl)
    ymajorLocator = plt.MultipleLocator(0.05)  # 将y轴主刻度标签设置为0.5的倍数
    ymajorFormatter = plt.FormatStrFormatter('%1.3f')  # 设置y轴标签文本的格式
    yminorLocator = plt.MultipleLocator(0.025)  # 将此y轴次刻度标签设置为0.1的倍数
    ax = subplot(111)  # 注意:一般都在ax中设置,不再plot中设置
    # 设置主刻度标签的位置,标签文本的格式
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    # 显示次刻度标签的位置,没有标签文本
    ax.yaxis.set_minor_locator(yminorLocator)

    plt.savefig("img/deep_test_" + type + ".jpg")
    plt.show()


data = readfile()  #data为四个json结果组成的对象数组
#折线图（验证集上不同模型各指标随训练轮数的变化）
draw_val(data, 'acc')
draw_val(data, 'precision')
draw_val(data, 'recall')
#条形图（测试集上不同模型各指标对比）
draw_test(data, 'acc')
draw_test(data, 'precision')
draw_test(data, 'recall')
draw_test(data, 'f1')
