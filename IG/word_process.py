# -*- coding:utf-8 -*-
import collections
import os
import jieba
import numpy as np
np.set_printoptions(threshold=np.nan)


def create_fenci(filename):
    # step2 读取文本，预处理，分词，以及每个词的单词个数
    raw_word_list = []
    sentence_list = []
    with open(filename, encoding='gb18030', errors='ignore') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            if len(line) > 0:  # 如果句子非空
                raw_words = list(jieba.cut(line, cut_all=False))
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words and word not in ['qingkan520', 'www', 'com', 'http']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()

    word_count = collections.Counter(raw_word_list)

    # print('文本中总共有{n1}个单词,不重复单词数{n2},选取前30000个单词进入词典'
    #     .format(n1=len(raw_word_list), n2=len(word_count)))
    # 返回每个文本的词频，自己相应的词语
    return raw_word_list


def eachFile(filepath):
    file = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        file.append(child)
    return file


if __name__ == '__main__':
    # 统计停用词个数
    stop_words = []
    with open('stop_words.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))
    # 读取文件中的文件名称
    filePathC = "text"
    file_list = eachFile(filePathC)
    print("文件名", file_list)
    # word_list存放词典
    word_list = []
    # 存放每个文本的标签即所属的类
    label = []
    # class_df_list 每个单词在每个类中出现的文本数
    class_df_list = np.zeros(10)
    for article in file_list:
        tmp = []
        tmp = create_fenci(article)
        word_list.append(tmp)
        if '电脑' in article:
            label.append(1)
            class_df_list[0] += 1
        elif '环境' in article:
            label.append(2)
            class_df_list[1] += 1
        elif '交通' in article:
            label.append(3)
            class_df_list[2] += 1
        elif '教育' in article:
            label.append(4)
            class_df_list[3] += 1
        elif '经济' in article:
            label.append(5)
            class_df_list[4] += 1
        elif '军事' in article:
            label.append(6)
            class_df_list[5] += 1
        elif '体育' in article:
            label.append(7)
            class_df_list[6] += 1
        elif '医药' in article:
            label.append(8)
            class_df_list[7] += 1
        elif '艺术' in article:
            label.append(9)
            class_df_list[8] += 1
        elif '政治' in article:
            label.append(10)
            class_df_list[9] += 1
    print("标签", label)
    print("每个单词在类中出现的文本数", class_df_list)
    print("总词库", word_list)