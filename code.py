#!/usr/bin/env python
# coding: utf-8

# 

# In[24]:


import jieba
import re
import random
import numpy as np

Data_path = "dataset2"
Topics = ["射雕英雄传", "神雕侠侣", "倚天屠龙记", "天龙八部"]

N = 200 # 段落数，做训练集
D = 500 # 每段的单词数


# In[25]:


def data_preprocess_sample(topic):
    """
    输入主题（小说），预处理数据，从中进行文章抽样，然后分词。
    每个小说抽 60 段，前 50 段作为训练集，后 10 段作为测试集。
    """

    data = open("%s/%s.txt" % (Data_path, topic), "r", encoding="utf-8")
    data = data.read()


    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        data = data.replace(a, '')

    data_sample = re.findall(r'.{%d}' % D, data)
    data_sample = random.sample(data_sample, N//len(Topics) + 10)
    data_sample = [jieba.lcut(d) for d in data_sample]

    return data_sample[:-10], data_sample[-10:]
    


# 对四本小说进行预处理，各取出 50+10 段素材，每段长度 500 字。（但是由于编码问题，分词后有不少词语变成方格乱码，实际可用字数不足 500，不过影响不大，也能正常聚类）

# In[26]:


training_txt0, testing_txt0 = data_preprocess_sample(Topics[0])
training_txt1, testing_txt1 = data_preprocess_sample(Topics[1])
training_txt2, testing_txt2 = data_preprocess_sample(Topics[2])
training_txt3, testing_txt3 = data_preprocess_sample(Topics[3])

training_txt = training_txt0 + training_txt1 + training_txt2 + training_txt3
testing_txt = testing_txt0 + testing_txt1 + testing_txt2 + testing_txt3

print(len(training_txt))
print(len(testing_txt))


# 以下是训练过程。
# 先给每一个 Document 的每一个词随机分配一个 Topic，然后就可以统计一些量：
# - 每个 Document 中，属于每一个 Topic 的词各有多少，记作 $Freq(Topic_j|Document_i)$
# - 每个 Document 中，有多少个 Word，记作 $Count(Document_i)$
# 
# 这两个量可以计算出 Document 中，每个 Topic 的概率分布。即：
# $ Freq(Topic_j|Document_i) / Count(Document_i) = P(Topic_j|Document_i)$
# 
# - 每个 Topic 中，每个 Word 的频数有多少，记作 $Freq(Word_k|Topic_j)$
# - 每个 Topic 中，也多少个 Word，记作 $Count(Topic_j)$
# 
# 这两个量可以计算出 Topic 中，每个 Word 的概率分布。即：
# $ Freq(Word_k|Topic_j) / Count(Topic_j) = P(Word_k|Topic_j)$
# 
# ------
# 
# 接下来我们使用上面两个概率分布来反推。假设 Document 中的每一个 Word 是由最大概率来生成的，即：
# $ P(Word_k|Document_i) = P(Word_k|Topic_j) \times P(Topic_j|Document_i)$
# 
# 由此，我们就可以挑出概率最大的 Topic，给每一个词重新分配一个 Topic，然后重复上述步骤，直到无法找到概率更大的新 Topic。

# In[27]:


Topic_All = []  # 200 个文章，每个文章中的 500 个词是什么 topic
Topic_count = {}    # 4 个 topic，每个 topic 当中有多少个词
# 4 个 topic，每个 topic 中每个词的词频
Topic0_fre = {}; Topic1_fre = {}; Topic2_fre = {}; Topic3_fre = {}
Doc_fre = []  # 200 个文章，每个文章中，4 个 topic 的词各有多少
Doc_count = []  # 200 个文章，每个文章中有多少个词（500）


i = 0

for data in training_txt:

    topic = []      # 500 个词，每个词初始化一个 topic
    docfre = {}

    for word in data:
        a = random.randint(0, len(Topics)-1)
        topic.append(a)

        if '\u4e00' <= word <= '\u9fa5':
            Topic_count[a] = Topic_count.get(a, 0) + 1  # 统计每个 topic 总词数
            docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的词频
            if i < 50:
                Topic0_fre[word] = Topic0_fre.get(word, 0) + 1
            elif i < 100:
                Topic1_fre[word] = Topic1_fre.get(word, 0) + 1
            elif i < 150:
                Topic2_fre[word] = Topic2_fre.get(word, 0) + 1
            else:
                Topic3_fre[word] = Topic3_fre.get(word, 0) + 1
        
        #else:
            #print(word)
    
    Topic_All.append(topic)

    docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
    Doc_fre.append(docfre)
    Doc_count.append(sum(docfre))  # [500, 500, ...] 可惜很多乱码无法识别，实际不到 500

    i = i + 1

Topic_count = list(dict(sorted(Topic_count.items(), key=lambda x: x[0], reverse=False)).values())
Doc_fre = np.array(Doc_fre)  # 转为array方便后续计算
Topic_count = np.array(Topic_count)  # 转为array方便后续计算
Doc_count = np.array(Doc_count)  # 转为array方便后续计算
print(Doc_fre.shape)
print(Topic_count)
print(Doc_count.shape)


# In[28]:


Doc_pro = []  # 每个topic被选中的概率
Doc_pronew = []  # 记录每次迭代后每个topic被选中的新概率

for i in range(len(training_txt)):

    doc = np.divide(Doc_fre[i], Doc_count[i])
    Doc_pro.append(doc)

Doc_pro = np.array(Doc_pro)
print(Doc_pro)


# In[29]:


stop = 0  # 迭代停止标志
loopcount = 1  # 迭代次数
while stop == 0:
    i = 0   # 文章数目
    for data in training_txt:
        top = Topic_All[i]
        for w in range(len(data)):
            word = data[w]
            pro = []
            topfre = []
            if '\u4e00' <= word <= '\u9fa5':
                #for j in range(len(Topics)):
                #    exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 读取该词语在每个topic中出现的频数
                topfre.append(Topic0_fre.get(word, 0))
                topfre.append(Topic1_fre.get(word, 0))
                topfre.append(Topic2_fre.get(word, 0))
                topfre.append(Topic3_fre.get(word, 0))
                
                pro = Doc_pro[i] * topfre / Topic_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                Doc_fre[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                Doc_fre[i][m] += 1
                Topic_count[top[w]] -= 1  # 更新每个topic的总词数
                Topic_count[m] += 1
                exec('Topic{}_fre[word] = Topic{}_fre.get(word, 0) - 1'.format(top[w], top[w]))  # 更新每个topic该词的频数
                exec('Topic{}_fre[word] = Topic{}_fre.get(word, 0) + 1'.format(m, m))
                top[w] = m
        Topic_All[i] = top
        i += 1
    print(Doc_fre, 'new')
    print(Topic_count, 'new')
    if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
        for i in range(len(training_txt)):
            doc = np.divide(Doc_fre[i], Doc_count[i])
            Doc_pronew.append(doc)
        Doc_pronew = np.array(Doc_pronew)
    else:
        for i in range(len(training_txt)):
            doc = np.divide(Doc_fre[i], Doc_count[i])
            Doc_pronew[i] = doc
    print(Doc_pro)
    print(Doc_pronew)
    if (Doc_pronew == Doc_pro).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为模型已经训练完毕
        stop = 1
    else:
        Doc_pro = Doc_pronew.copy()
    loopcount += 1
print(Doc_pronew)  # 输出最终训练的到的每篇文章选中各个topic的概率
print(loopcount)  # 输出迭代次数
print(Topic_count)
print('模型训练完毕！')


# In[30]:


print(Doc_pronew.shape)
print(np.argmax(Doc_pronew, 1)[:50]) 
print(np.argmax(Doc_pronew, 1)[50:100]) 
print(np.argmax(Doc_pronew, 1)[100:150]) 
print(np.argmax(Doc_pronew, 1)[150:200]) 
# 训练结果非常神奇，能够完美分类


# 模型实际训练出了不同 Topic 中每一个 Word 的词频以及总词数，也就是 Topic 中，Word 的概率分布，在测试中直接调用就行。

# In[31]:


Doc_count_test = []  # 每篇文章中有多少个词
Doc_fre_test = []  # 每篇文章有多少各个topic的词
Topic_All_test = []  # 每篇文章中的每个词来自哪个topic
i = 0
for data in testing_txt:
    topic = []
    docfre = {}
    for word in data:
        a = random.randint(0, len(Topics) - 1)  # 为每个单词赋予一个随机初始topic
        topic.append(a)
        if '\u4e00' <= word <= '\u9fa5':
            docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的词频
    Topic_All_test.append(topic)
    docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
    Doc_fre_test.append(docfre)
    Doc_count_test.append(sum(docfre))  # 统计每篇文章的总词数
    i += 1
# print(Topic_All[0])
Doc_fre_test = np.array(Doc_fre_test)
Doc_count_test = np.array(Doc_count_test)
print(Doc_fre_test)
print(Doc_count_test)


Doc_pro_test = []  # 每个topic被选中的概率
Doc_pronew_test = []  # 记录每次迭代后每个topic被选中的新概率
for i in range(len(testing_txt)):
    doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
    Doc_pro_test.append(doc)
Doc_pro_test = np.array(Doc_pro_test)
print(Doc_pro_test)


stop = 0  # 迭代停止标志
loopcount = 1  # 迭代次数
while stop == 0:
    i = 0
    for data in testing_txt:
        top = Topic_All_test[i]
        for w in range(len(data)):
            word = data[w]
            pro = []
            topfre = []
            if '\u4e00' <= word <= '\u9fa5':
                # for j in range(len(data_txt)):
                # exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 读取该词语在每个topic中出现的频数
                topfre.append(Topic0_fre.get(word, 0))
                topfre.append(Topic1_fre.get(word, 0))
                topfre.append(Topic2_fre.get(word, 0))
                topfre.append(Topic3_fre.get(word, 0))
                
                pro = Doc_pro_test[i] * topfre / Topic_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                Doc_fre_test[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                Doc_fre_test[i][m] += 1
                top[w] = m
        Topic_All_test[i] = top
        i += 1
    print(Doc_fre_test, 'new')
    if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
        for i in range(len(testing_txt)):
            doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
            Doc_pronew_test.append(doc)
        Doc_pronew_test = np.array(Doc_pronew_test)
    else:
        for i in range(len(testing_txt)):
            doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
            Doc_pronew_test[i] = doc
    print(Doc_pro_test)
    print(Doc_pronew_test)
    if (Doc_pronew_test == Doc_pro_test).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为训练集已分类完毕
        stop = 1
    else:
        Doc_pro_test = Doc_pronew_test.copy()
    loopcount += 1
print(Doc_pronew)
print(Doc_pronew_test)
print(loopcount)
print('测试集测试完毕！')


# In[33]:


result = []
for k in range(len(testing_txt)):
    pro = []
    for i in range(len(Topics)):
        dis = 0
        for j in range(len(Topics)):
            dis += (Doc_pro[i][j] - Doc_pro_test[k][j])**2  # 计算欧式距离
        pro.append(dis)
    m = pro.index(min(pro))
    print(pro)
    result.append(m)
#print(files)
print(result[:10])
print(result[10:20])
print(result[20:30])
print(result[30:])
# 测试分类成功率相当感人
# 可以考虑用测试得到的概率分布去训练一个 full-connect 模型试试，能不能很好地拟合出一个分类模型


# In[37]:


print(Doc_pronew_test.shape)


# In[45]:


label = [0 for i in range(10)] + [1 for i in range(10)] + [2 for i in range(10)] + [3 for i in range(10)]
train_index = random.sample(range(40), 30)
test_index = [i for i in range(40) if i not in train_index]

print(train_index)
print(test_index)

labels = np.zeros(Doc_pronew_test.shape)
for i in range(len(label)):
    labels[i, label[i]] = 1

print(labels)


# In[65]:


train_data = Doc_pronew_test[train_index, :]
train_label = labels[train_index]
test_data = Doc_pronew_test[test_index, :]
test_label = labels[test_index]

import torch
import torch.nn as nn
from torch.autograd import Variable

train_data = torch.from_numpy(train_data.astype(np.float32))
train_label = torch.from_numpy(train_label.astype(np.float32))
test_data = torch.from_numpy(test_data.astype(np.float32))
test_label = torch.from_numpy(test_label.astype(np.float32))

print(train_label.shape)
print(test_label.shape)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)

        return x


# In[66]:


model = Classifier()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1,
                             weight_decay=1e-5)
criterion = nn.MSELoss()

for i in range(200):
    total_loss = 0

    optimizer.zero_grad()

    output = model(train_data)
    loss = criterion(output, train_label)

    loss.backward()
    optimizer.step()

    total_loss += loss.data

    if i % 10 == 0:
        print('epoch [{}/{}], loss:{:.4f}'
            .format(i, 10, total_loss))


torch.save(model.state_dict(), 'classifier.pth')


# In[74]:


output_test = model(test_data)
print(output_test)
print(test_label)

print(np.argmax(output_test.detach().numpy(), 1))
print(np.argmax(test_label.numpy()), 1)

loss_test = criterion(output_test, test_label)

print(loss_test)

# 正确率还算凑或，比直接的用距离来衡量好多了

