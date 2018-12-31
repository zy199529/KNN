# -*- coding:utf-8 -*-
from IG_word.IG_reduction import *
from IG_word.word_process import *
from IG_word.TF_IDF import *


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k, label):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        # print(trainingSet[x])
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((label[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccurcy(test_label, predictions):
    correct = 0
    for x in range(len(test_label)):
        if test_label[x] == predictions[x]:
            correct += 1
    return (correct / float(len(test_label))) * 100.0


def testbagOfWord2Vec(vocabList, inputSet):  # 词袋模型，统计概率的
    tmp = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            tmp[vocabList.index(word)] += 1  # 当前文档有这个词条，则根据词典位置获取其位置并赋值为1
    return tmp


if __name__ == '__main__':
    # 测试文本，使用KNN分类
    vocabList = reduction_words()  # 降维后的词典
    # testname = '交通_41.TXT'
    # testList = create_fenci(testname)
    # testVec = testbagOfWord2Vec(vocabList, testList)
    train_vec_List, idf_array, label = tf_idf()
    test_vec_List, test_label = test_tf_idf()
    # test_vec = np.array(get_l_tf(np.array(testVec)) * idf_array)
    k = 25
    prediction = []
    for x in range(len(test_vec_List)):
        neighbors = getNeighbors(train_vec_List, test_vec_List[x], k, label)
        result = getResponse(neighbors)
        prediction.append(result)
        print("prediction=" + repr(result) + "actual" + repr(test_label[x]))
    accuracy = getAccurcy(test_vec_List, prediction)
    print(neighbors)
