# !usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from pylab import *
from sklearn import svm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd

train_dir = './data/train.csv'
test_dir = './data/test.csv'
result_dir = './data/sampleSubmission.csv'
final_dir = './data/submit.csv'
# train_file_dir = './data/pretrain/train.txt'
# test_file_dir = './data/pretrain/test.txt'

def read_train_file(train_dir):
    train_label = []
    train_text = []
    ham_num = 0
    with open(train_dir,'r',encoding='utf-8') as f:
        data = csv.reader(f)
        for line in data:
            if line[0] == 'ham' and ham_num < 500:   # positive >>> negative ,so just use the 500th datas
                ham_num += 1
                train_label.append(1)
                train_text.append(line[1].strip())
            if line[0] == 'spam':
                train_label.append(0)
                train_text.append(line[1].strip())
    assert len(train_text) == len(train_label)  # if the boolean is false,it will be AssertError
    print('训练集： %d个数据,其中 %d 正样本，%d 负样本' % (len(train_label), ham_num, len(train_label) - ham_num))
    return train_text,train_label


def read_test_file(test_dir,result_dir):
    test_label = []
    test_text = []
    ham_num = 0
    with open(test_dir,'r',encoding='utf-8') as f:
        data = csv.reader(f)
        for index,line in enumerate(data):
            if index == 0 :
                pass
            else:
                test_text.append(line[1].strip())

    with open(result_dir,'r',encoding='utf-8') as f:
        data = csv.reader(f)
        for line in data :
            if line[1] == 'ham':
                ham_num += 1
                test_label.append(1)
            if line[1] == 'spam' :
                test_label.append(0)
    print('测试集： %d个数据，其中%d 个正例，%d 个反例' % (len(test_text),ham_num,len(test_text)-ham_num))
    return test_text,test_label

def build_features(train_text,test_text,n):
    def pca(data,n):
        pca = PCA(n_components=n, whiten=False, svd_solver='auto', random_state=None)
        return pca.fit_transform(data)

    data = train_text + test_text
    tfidfvectorizer = TfidfVectorizer()
    vectors = tfidfvectorizer.fit_transform(data).toarray()
    vectors = pca(vectors, n)
    # # or use below method
    # vectors = TfidfTransformer().fit_transform(count_vectors).toarray()
    # vectors = pca(vectors,n)
    return vectors[:len(train_text)],vectors[len(train_text):]


def SVM_Classifier(train_data, train_label):
    # train
    clf = svm.SVC()
    weight = np.array([100 if label == 0 else 1  for label in train_label])
    clf.fit(train_data, train_label, sample_weight= weight)  # sample_weight = weight
    # clf.fit(train_data,train_label)
    print('train finish')
    # predict
    train_pred = clf.predict(train_data)
    prec = sum([train_pred == train_label]) / len(train_pred)
    print('训练数据集准确度： %f' % float(prec))
    return clf


def Bayes_train(train_data,train_label):
    clf = GaussianNB()
    clf.fit(train_data,train_label)
    print('train finish')
    # predict
    train_pred = clf.predict(train_data)
    prec = sum([train_pred == train_label])/len(train_data)
    print('训练数据集准确度： %f' % float(prec))
    return clf


def pred(clf, test_data, test_label):
    def get_test_id(test_dir):
        test_id = []
        with open(test_dir,'r',encoding='utf-8') as f:
            data = csv.reader(f)
            for index,line in enumerate(data):
                if index > 0:
                    test_id.append(line[0])
        return test_id

    def write_result(test_pred,test_id,final_dir):
        dataframe = pd.DataFrame({'pred_label':test_pred,'test_id': test_id })
        dataframe.to_csv(final_dir, index=False, sep=',')
        # with open(final_dir,'a',newline='',encoding='utf-8') as f:
        #     csv_writer = csv.writer(f,dialect='excel')
        #     csv_writer.writerow(['test_id','pred_label'])

        print('测试结果写入完成')

    test_pred = clf.predict(test_data)
    prec = sum([test_pred == test_label])/len(test_label)
    print('测试数据集准确度： %f' % float(prec))
    test_id = get_test_id(test_dir)
    write_result(test_pred,test_id,final_dir)


if __name__ == '__main__':
    train_text, train_label = read_train_file(train_dir)
    test_text, test_label = read_test_file(test_dir,result_dir)
    train_vec, test_vec = build_features(train_text, test_text, 50)
    clf = SVM_Classifier(train_vec, train_label)  # acuuracy: 100%
    pred(clf, test_vec, test_label)
    # plt.title('SVM-LinearSVC')
    # plt.xlabel('pca dimensionality')
    # plt.ylabel('accuracy')
    # x = linspace(10,90,9)
    # y1 = []
    # y2 = []
    # for dim in arange(10,100,10):
    #     print('PCA 维度为 %d' % dim)
    #     train_vec, test_vec = build_features(train_text,test_text,dim)
    #     clf = SVM_Classifier(train_vec,train_label)    # acuuracy: 100%
    #     # clf = Bayes_train(train_vec,train_label)   # accuracy: 20.36%
    #     pred(clf,test_vec,test_label)
    #     y1.append(y1_1)
    #     y2.append(y2_2)
    # plt.plot(x,y1,'r')
    # plt.plot(x,y2,'g')
    # plt.show()



#
# def getDataSet(file_path):
#     dataset = csv.reader(open(file_path, 'r', encoding='utf-8'))
#     Label = []
#     Text = []
#     print(dataset)
#     print(type(dataset))
#     for index, data in enumerate(dataset):
#         if index == 0:
#             continue
#         else:
#             # Label += [data[0]]
#             # Text += [data[1]]
#             Label.append(data[0])
#             Text.append(data[1])
#     return Label,Text
#
#
# def token(train_Text,test_Text):
#     train_words = []
#     test_words = []
#
#     # list_stopWords = list(set(stopwords.words('english')))
#     # print(list_stopWords)
#     # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#     # train_sentences = tokenizer.tokenize(train_Text)
#
#     for text in train_Text:
#         list_words = WordPunctTokenizer().tokenize(text)
#         # train_list_guoliWords = []
#         #
#         # for word in list_words:
#         #     if word not in list_stopWords:
#         #         train_list_guoliWords.append(word)
#         # train_words.append(train_list_guoliWords)
#         train_words.append(list_words)
#
#     for text in test_Text:
#         list_words = WordPunctTokenizer().tokenize(text)
#         test_words.append(list_words)
#
#
#
#     with open(train_file_dir, 'w') as f:
#         for word_list in train_words:
#             # print(type(word))  # the type of word is list
#             for word in word_list:
#                 try:
#                     f.write(word+" ")
#                 except UnicodeEncodeError:
#                     pass
#             f.write('\n')
#
#
#     with open(test_file_dir, 'w') as f:
#         for word_list in test_words:
#             for word in word_list:
#                 try:
#                     f.write(word+" ")
#                 except UnicodeEncodeError:
#                     pass
#             f.write('\n')
#
#
# # TF-IDF 统计词频权重 得到词向量，并进行PCA降维
# def tfIdfSum():
#     # train_text = []
#     with open(train_file_dir,'r+') as f:
#         # for line in f:
#         #     train_text.append(line)
#         train_text = f.readlines()
#     # print(train_text)
#     # print(type(train_text))   # type : list
#
#     with open(test_file_dir, 'r+') as f:
#         test_text = f.readlines()
#
#     # text = ['I am OK','Tanks a lot ,baby','baby,I love you you' ,'You are my my honor']
#     vectorizer = CountVectorizer()
#     transformer = TfidfTransformer()
#     vector_words = vectorizer.fit_transform(train_text+test_text)
#     tfidf_words = transformer.fit_transform(vector_words).toarray()
#
#     vector_train_words = tfidf_words[:len(train_text)]
#     vec_train_words = tfidf_words[len(train_text):]
#     # print("tf-idf词向量： \n",weight_train)
#
#     # for i in range(100):
#     #     print(u"-------这里输出第", i, u"个文本的词语tf-idf权重------")
#     #     for j in range(len(word)):
#     #         if weight_train[i][j] !=0 :
#     #             print(word[j],weight_train[i][j])
#
#
#     vec_train_label = []
#     for x in train_Label:
#         if x=='spam' :
#             vec_train_label.append(1)
#         else:
#             vec_train_label.append(0)
#     print(vec_train_label)
#
#     vec_test_label = []
#     for x in sub_Label:
#         if x=='spam' :
#             vec_test_label.append(1)
#         else:
#             vec_test_label.append(0)
#     print(vec_test_label)
#
#     return vector_train_words,vec_train_label,vec_train_words,vec_test_label

#
#
# def pcaJiangwei(n):
#     pca = PCA(n_components=n, whiten=False, svd_solver='auto', random_state=None)
#     pca.fit(weight_train)
#     weight_train_word = pca.transform(weight_train)
#     pca.fit(weight_test)
#     weight_test_word = pca.transform(weight_test)
#     return weight_train_word,weight_test_word
#
# def bayesClassifier(n):
#     weight_train_word,weight_test_word = pcaJiangwei(n)
#     # print(weight_train_word,weight_test_word)
#     clf = GaussianNB().fit(weight_train_word, weight_train_label)
#     y_train_pred = clf.predict(weight_train_word)  # param:  X:[[]], Y:[]
#     right_num = (y_train_pred == weight_train_label).sum()
#     train_accuracy =  float(right_num / len(y_train_pred))
#     print('训练集上正确率： ',train_accuracy)
#
#     y_test_pred = clf.predict(weight_test_word)
#     right_num2 = (y_test_pred == weight_test_label).sum()
#     test_accuracy = float(right_num2 / len(y_test_pred))
#     print('测试集上正确率： ', test_accuracy)
#     return train_accuracy,test_accuracy
#
#
# def SVMClassifier(n):
#     weight_train_word, weight_test_word = pcaJiangwei(n)
#     clf =svm.SVC()
#     clf.fit(weight_train_word, weight_train_label)
#     y_train_pred = clf.predict(weight_train_word)
#     right_num = (y_train_pred == weight_train_label).sum()
#     train_accuracy = float(right_num/len(weight_train_label))
#     print('训练集上正确率：',train_accuracy)
#
#     y_test_pred = clf.predict(weight_test_word)
#     right_num2 = (y_test_pred == weight_test_label).sum()
#     test_accuracy = float(right_num2/len(weight_test_label))
#     print('测试集上正确率：',test_accuracy)
#     return train_accuracy,test_accuracy
#



# if __name__ == '__main__':
    # # import sys
    # # print(sys.getdefaultencoding()) # 默认为utf-8
    #
    # # get dataset
    # train_Label,train_Text = getDataSet(train_dir)
    # test_Id,test_Text = getDataSet(test_dir)
    # sub_Id,sub_Label = getDataSet(result_dir)
    # # tokenize
    # token(train_Text,test_Text)
    # # get tfidf word embedding
    # weight_train, weight_train_label,weight_test,weight_test_label = tfIdfSum()
    # # use bayes model to classifier
    # plt.title('bayes classifier')
    # plt.xlabel('PCA n_components')
    # plt.ylabel('accuracy')
    # x = linspace(0, 250, 12)
    # y1 = []
    # y2 = []
    # for i in range(10,250,20):
    #     print('pca维度为： ',i)
    #     train_y ,test_y = SVMClassifier(i)
    #     y1.append(train_y)
    #     y2.append(test_y)
    # figure()
    # plot(x,y1,'r')
    # plot(x,y2,'g')
    # plt.show()





    # x = linspace(0, 5, 10)
    # y = x ** 2
    #
    # figure()
    # plot(x, y, 'r')
    # xlabel('x')
    # ylabel('y')
    # title('title')
    # plt.show()





    # A = []
    # A += ['String']
    # A += 'Str'
    # A.append('st')
    # A.append(['str'])
    # print(A)






