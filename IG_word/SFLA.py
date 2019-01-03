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
    Xg, Xb = max_min_fitness(frogNum, myarray)  # 找到分类最好的那只青蛙
    return myarray, myarray[Xg], vocabList, frogNum, L, T, Dmax, memeplexNum  # 返回随机生成的青蛙序列，最大的那只青蛙，词典


# 青蛙个数，族群内进化次数L，总迭代次数T，最大移动步长Dmax


def SFLA_memeplex():  # 族群内进化
    myarray, Xg, vocabList, frogNum, L, T, Dmax, memeplexNum = SFLA_init()
    for k in range(frogNum):  # 将青蛙划分到种群中，一共5个种群，每个族群4只青蛙
        # step2 划分子群
        M = []
        for p in range(memeplexNum):  # 族群进行初始化为空
            M.append([])
        for p in range(frogNum):  # 将青蛙加入到族群中
            M[p % memeplexNum].append(myarray[p])
    for y in range(memeplexNum):  # 输出每个族群每只青蛙的准确率
        for x in range(4):
            print(second_reduction(M[y][x]))
    for i in range(memeplexNum):  # 每个族群内部进化
        for t in range(L):  # 族群内进化次数L，每个族群进化10次
            r1 = np.random.randint(0, 100)  # 生成随机数r1
            r2 = np.random.randint(0, 100)  # 生成随机数r2
            Xb_value, Xw_value = max_min_fitness(4, M[i])  # 每个族群中最好和最差的青蛙的位置
            Xb = M[i][Xb_value]  # 记录下族群内最好的青蛙
            Xw = M[i][Xw_value]  # 记录下族群内最坏的青蛙
            Xw_new = frog_evolution(Xb, Xw, r1, r2)  # 青蛙进化，Xb是最好的青蛙，Xw是最坏的青蛙，r1和r2是随机数
            if second_reduction(Xw_new) > second_reduction(Xw):  # 将产生的新的青蛙和旧的青蛙比较适应度函数
                M[i][Xw_value] = Xw_new  # 若新青蛙好则用新青蛙代替
            else:
                # 否则用全局最优代替局部最优再进行优化
                Xw_new = frog_evolution(Xg, Xw, r1, r2)  # 全局最优进行优化
                if second_reduction(Xw_new) > second_reduction(Xw):  # 假如效果好则使用全局最优后选出的青蛙
                    M[i][Xw_value] = Xw_new  # 新青蛙代替
                else:
                    M[i][Xw_value] = np.random.randint(0, 2, len(vocabList))  # 若效果还是不好则随机产生
            for l in range(memeplexNum):
                for h in range(4):
                    print("L=", t, second_reduction(M[l][h]))
    return M


def frog_evolution(Xb, Xw, r1, r2):
    G = np.zeros(len(Xb))  # G产生一个零向量
    xb=Xb
    xw=Xw
    Xb_Xw = np.zeros(len(xb))  # Xb_Xw产生一个零向量
    Xw_Xb = np.zeros(len(xb))  # Xw_Xb产生一个零向量
    for j in range(len(xb)):  # 将最好和最差的青蛙共同为1的位置保存下来，也将G的相同位置赋值为1，将Xb和Xw
        if xb[j] == xw[j]:
            G[j] = 1
    a = 0
    b = 0
    for k in range(len(xb)):
        if xb[k] == 1 and xw[k] == 0:
            Xb_Xw[k] = 1
            a = a + 1
    a = int(a * r1)  # Xb_Xw需要保留几个1,其余清为0
    for q in range(len(xb)):
        if xw[q] == 1 and xb[q] == 0:
            Xw_Xb[q] = 1
            b = b + 1
    b = int(b * r2)  # Xw_Xb需要保留几个1,其余清为0
    o = 0
    for w in range(len(xb)):  # Xb_Xw保留1,其余清为0
        if Xb_Xw[w] == 1 and o <= a:
            o = o + 1
        else:
            Xb_Xw[w] = 0
    z = 0
    for w in range(len(xb)):  # Xw_Xb保留1,其余清为0
        if Xw_Xb[w] == 1 and z <= b:
            z = z + 1
        else:
            Xw_Xb[w] = 0
    Xw_new = Xw_Xb + Xb_Xw + G  # 新的青蛙产生
    return Xw_new


def max_min_fitness(frogNum, myarray):
    accuracy_all = []
    for i in range(frogNum):
        accuracy_all.append((i, second_reduction(myarray[i])))
    accuracy_all = np.array(accuracy_all)
    accuracy_all = accuracy_all[np.lexsort(-accuracy_all.T)]
    return int(accuracy_all[0][0]), int(accuracy_all[frogNum - 1][0])  # 找到适应度最好和最差的那只青蛙


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
    print(SFLA_memeplex())
