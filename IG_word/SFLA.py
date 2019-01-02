# -*- coding:utf-8 -*-
from IG_word.tf_idf_sfla import *
from IG_word.KNN_classify import *


def words():  # 读取降维后的词语
    words = []
    with open('reduction_words.txt', encoding='gb18030', errors='ignore') as f:
        line = f.readline()
        while line:
            words.append(line[:-1])
            line = f.readline()
    return words


# 使用蛙跳算法智能优化特征二次选择
def SFLA_init():
    vocabList = words()  # 使用信息熵降维后的词典
    frogNum = 20  # 蛙群规模N=20一共20个青蛙
    L = 10  # 族群内进化次数L
    T = 10  # 总迭代次数T
    Dmax = 45  # 最大移动步长Dmax
    memeplexNum = 5  # 族群数量5
    myarray = np.random.randint(0, 2, (20, len(vocabList)))  # myarray初始化青蛙的特征项0或1
    Xg = myarray[int(max_fitness(frogNum, myarray))]  # 找到分类最好的那只青蛙
    return myarray, Xg, vocabList, frogNum, L, T, Dmax, memeplexNum


def SFLA_memeplex():
    myarray, Xg, vocabList, frogNum, L, T, Dmax, memeplexNum = SFLA_init()
    for k in range(frogNum):  # 将青蛙划分到种群中，一共5个种群，每个族群4只青蛙
        # step2 划分子群
        M = []
        for i in range(memeplexNum):  # 族群进行初始化为空
            M.append([])
        for i in range(frogNum):  # 将青蛙加入到族群中
            M[i % memeplexNum].append(myarray[i])
    return M

def max_fitness(frogNum, myarray):
    accuracy_all = []
    for i in range(frogNum):
        accuracy_all.append((i + 1, second_reduction(myarray[i])))
    accuracy_all = np.array(accuracy_all)
    accuracy_all = accuracy_all[np.lexsort(-accuracy_all.T)]
    return accuracy_all[0][0]  # 是5*4的矩阵


def KNN_classify(second_redution):  # 对每一只青蛙（词库）KNN分类
    train_vec_List_sfla, idf_array_sfla, label_sfla = tf_idf_sfla(second_redution)  # 计算训练集的tf_idf
    test_vec_List_sfla, test_label_sfla = test_tf_idf_sfla(second_redution)  # 计算测试集的tf_idf
    k = 25
    prediction_sfla = []
    for x in range(len(test_vec_List_sfla)):
        neighbors_sfla = getNeighbors(train_vec_List_sfla, test_vec_List_sfla[x], k, label_sfla)
        result_sfla = getResponse(neighbors_sfla)
        prediction_sfla.append(result_sfla)
        # print("prediction=" + repr(result_sfla) + "  actual=" + repr(test_label_sfla[x]))#预测值和真实值
    accuracy_sfla = getAccurcy(test_label_sfla, prediction_sfla)  # 正确率%
    return accuracy_sfla


def second_reduction(frog_word):
    frog_vocablist = []
    vocabList = np.array(words())
    for i in range(len(frog_word)):
        if frog_word[i] == 1:
            frog_vocablist.append(vocabList[i])
    accuracy = KNN_classify(frog_vocablist)
    return accuracy


if __name__ == '__main__':
    print(SFLA_init())
