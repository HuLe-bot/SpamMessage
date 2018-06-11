# SpamMessage
## lintcode AI train task
directory -data: includes thress files: train_data.cv  test_data.csv and the submitSample.csv

spamMess.py and textClassify.py deal with the same problem: use sklearn.naive_bayse end sklearn.svm to classify the sapm message and the ham message.
steps:
1. read train_data.csv to get train_text and train_label
2. read test_fata.csv and submitSmaple.csv to get test_text and test_label
3. Bayes_train or SVM_train: 
use TF-IDF sentence2vec to build feature,and use pca to deal with the data sparseness probelm, finally train the model
in the train data: ham Message number >>> spam Message nuber,so i apply assign the different weights to ham and spam meassage or discarded some train ham message ,then fit(X,y,weight)   
4. predict: SVM >>> Naive Bayes
5. reviews: accuracy 
