import torch
from torch.utils.data import DataLoader, Dataset
import re
import pickle
# emotion_dict = {"none":0, "like":1, }

class NlpccDataset(Dataset):
    def __init__(self):
        self.path1 = r'D:\Python_projection\bilibili\nlpcc\Nlpcc2013Train.tsv'
        self.path2 = r'D:\Python_projection\bilibili\nlpcc\Nlpcc2014Train.tsv'
        self.lines1 = open(self.path1, encoding='utf-8').readlines() # readlines 返回列表
        self.lines2 = open(self.path2, encoding='utf-8').readlines()
        self.lines = self.lines1[1:] + self.lines2[1:]
        self.labels = []
        self.contents = []
        # filters = ['\t', '\n', '#', '$', '%', '@', '//']
        for line in self.lines:
            line = line.strip('\n')
            label_str, content = line.split(',', maxsplit=1)
            if label_str == 'none':
                label = 0
            elif label_str == 'like':
                label = 1
            elif label_str == 'disgust':
                label = 2
            elif label_str == 'surprise':
                label = 3
            elif label_str == 'happiness':
                label = 4
            elif label_str == 'fear':
                label = 5
            elif label_str == 'sadness':
                label = 6
            elif label_str == 'anger':
                label = 7
            else:
                print('有未分类标签')
            self.labels.append(label)
            self.contents.append(content)
        # 要删掉部分的none 再构建训练集以及测试集
        # label_0: 39484
        # label_1: 6145
        # label_2: 4437
        # label_3: 1211
        # label_4: 3920
        # label_5: 447
        # label_6: 3625
        # label_7: 2856
        # 62125
        i = j = 0
        while j < 20000:
            if self.labels[i] == 0:
                self.labels.pop(i)
                self.contents.pop(i)
                j += 1
            else:
                i += 1
        # label_0: 19484
        # label_1: 6145
        # label_2: 4437
        # label_3: 1211
        # label_4: 3920
        # label_5: 447
        # label_6: 3625
        # label_7: 2856
        # 42125
        total = list(zip(self.contents, self.labels))
        # print(total[0:10])
        # print(type(self.contents)) # list
        # print(type(self.contents[0])) # str
        test = []
        train = []
        for i in range(len(self.contents)):
            if i%10 == 0:
                test.append(total[i]) # [(content,label), (content,label)...]
            else:
                train.append(total[i])
        # print(len(train))
        # print(len(test))

        f_test = open(r'D:\Python_projection\bilibili\nlpcc\Test.txt', 'w', encoding='utf-8')
        f_train = open(r'D:\Python_projection\bilibili\nlpcc\Train.txt', 'w', encoding='utf-8')
        # f_test.write(test)
        for i, val in enumerate(train):
            f_train.write(str(train[i][1])+'\t'+train[i][0]+'\n') # 不能放元组
        for i, val in enumerate(test):
            f_test.write(str(test[i][1])+'\t'+test[i][0]+'\n')
        # pickle.dump(train, f_train)
        # pickle.dump(test, f_test)
        f_train.close()
        f_test.close()



    def __getitem__(self, idx):
        return self.content[idx], self.labels[idx]

    def __len__(self):
        return len(self.contents)

def change(path, path_save):
    label_dict = {'0':'无感','1':'振奋','2':'厌恶','3':'惊喜','4':'开心','5':'害怕','6':'难过','7':'生气'}
    with open(path, 'r', encoding='utf-8') as f:
        with open(path_save, 'w', encoding='utf-8') as w:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                label_digit, text = line.split('\t')
                if label_digit in label_dict:
                    label = label_dict[label_digit]
                else:
                    print('错误')
                w.write(label+'\t'+text+'\n')

def dev_data():
    test = open('Test2.txt','w',encoding='utf-8')
    dev = open('Dev.txt', 'w', encoding='utf-8')
    with open('Test1.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            line = line.strip('\n')
            if i%2 == 0:
                test.write(line + '\n')
            else:
                dev.write(line + '\n')
            i += 1



if __name__ == '__main__':
    # nlpcc = NlpccDataset()
    path = 'Test.txt'
    path_save = 'Test1.txt'
    # change(path,path_save)
    dev_data()
