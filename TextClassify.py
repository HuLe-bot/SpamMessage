import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import numpy as np

train_file = './data/train.csv'
test_data_file = './data/test.csv'
test_label_file = './data/sampleSubmission.csv'

def read_train_file(file):
    '''
    :param file: file path
    :return: dict{sentences:label}
    '''
    data = []
    label = []
    with open(file, 'r', encoding='utf8') as f:
        ham = 0
        content = csv.reader(f)
        for line in content:
            if line[0] == 'ham' and ham < 500:
                ham += 1
                label.append(1)
                data.append(line[1].strip())
            if line[0] == 'spam':
                label.append(0)
                data.append(line[1].strip())
    assert len(data) == len(label)
    print('%d train data, %d 正样本，%d 负样本' % (len(data), sum(label), len(data)-sum(label)))
    return data, np.array(label)

def read_test_file(data_file, label_file):
    data = []
    label = []
    with open(label_file, 'r', encoding='utf8') as f:
        content = csv.reader(f)
        for row in content:
            if row[1] == 'ham':
                label.append(1)
            if row[1] == 'spam':
                label.append(0)

    with open(data_file, 'r', encoding='utf8') as f:
        content = csv.reader(f)
        for i, row in enumerate(content):
            if i > 0:
                data.append(row[1].strip())

    print('%d test data, %d 正样本，%d 负样本' % (len(data), sum(label), len(data) - sum(label)))
    return data, np.array(label)


def build_features(train_data, test_data, n):

    def pca(data, n):
        pca = PCA(n_components=n, whiten=False, svd_solver='auto', random_state=None)
        return pca.fit_transform(data)

    # sentence2vector
    data = train_data + test_data
    tfidfvectorizer = TfidfVectorizer()
    vectors = tfidfvectorizer.fit_transform(data).toarray()
    vectors = pca(vectors, n)

    return vectors[:len(train_data)], vectors[len(train_data):]


def SVM_train(train_data, train_label):
    # train
    clf = svm.SVC()
    data_weight = np.array([100 if label == 0 else 1 for label in train_label])
    clf.fit(train_data, train_label, sample_weight=data_weight)
    print('finish train')
    train_pred = clf.predict(train_data)
    prec = np.sum(train_pred == train_label)/len(train_pred)
    print('训练数据集准确率为%f' % float(prec))
    return clf

def Bayes_train(train_data,train_label):
    clf = GaussianNB()
    data_weight = np.array([100 if label == 0 else 1 for label in train_label])
    clf.fit(train_data, train_label, sample_weight=data_weight)
    print('finish train')
    train_pred = clf.predict(train_data)
    prec = np.sum(train_pred == train_label) / len(train_pred)
    print('训练数据集准确率为%f' % float(prec))
    return clf


def pred(clf, test_data, test_label):
    pred = clf.predict(test_data)
    prec = np.sum(test_label == pred) / len(test_label)
    print('测试数据集准确率为%f' % float(prec))


if __name__ == '__main__':
    train_data, train_label = read_train_file(train_file)
    test_data, test_label = read_test_file(test_data_file, test_label_file)
    for dim in range(10, 100, 10):
        print('PCA 维度为 %d' % dim)
        train_vec, test_vec = build_features(train_data, test_data, dim)
        model = SVM_train(train_vec, train_label)
        # modl = Bayes_train(train_vec,train_label)
        pred(model, test_vec, test_label)
