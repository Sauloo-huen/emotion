from matplotlib import pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification,RobertaTokenizer


def print1():
    x1 = np.linspace(1, 10, 20)
    print(x1)
    y1 = x1 * x1 + 2
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    axes.plot(x1, y1, 'r')
    plt.show()

def print2():
    import numpy as np
    import matplotlib.pyplot as plt
    import os  # 导入os库

    x = np.linspace(0, 10, 30)  # 产生0-10之间30个元素的等差数列
    noise = np.random.randn(30)  # 产生30个标准正态分布的元素
    y1 = 110
    y2 = 355
    y3 = 530
    # plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置字体为SimHei显示中文\n",
    plt.rc('font', size=14)  # 设置图中字号大小\n",

    plt.figure(figsize=(6, 4))  # 设置画布\n",
    plt.bar([0, 1, 2], [np.sum(y1), np.sum(y2), np.sum(y3)], width=0.5)  # 绘制柱状图\n",
    plt.title('parameter')  # 添加标题\n",
    labels = ['Bert', 'Roberta-large', 'wudao']
    plt.xlabel('moedl')  # 添加横轴标签\n",
    plt.ylabel('parameter')  # 添加纵轴标签\n",
    plt.xticks(range(3), labels)  # 横轴刻度与标签对准\n",

    path = '/common-data/new_build/xiaoying.huang/emotion-main/img'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + 'scatter.jpg')  # 保存图片
    plt.savefig(path + 'plot.jpg')  # 保存图片\n",
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # model = RobertaForSequenceClassification.from_pretrained('/common-data/new_build/xiaoying.huang/emotion-main/roberta-large')
    # print(count_parameters(model))
    print2()

if __name__ == '__main__':
    main()
