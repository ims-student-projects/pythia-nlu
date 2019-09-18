import sys 
sys.path.append(sys.path[0] + '/../')
from corpus.corpus_base import Corpus
from feature_extract.feature_base import Feature

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

class SVM:
    def __init__(self, train_corpus, test_corpus):

        self.corpus_tr = train_corpus
        self.corpus_ts = test_corpus
        self.__predicted = []

        self.setup()
        self.train_data = self.feature_tr
        self.train_targets = self.all_targ_tr
        self.test_data = self.feature_ts
        self.test_targets = self.all_targ_ts

    def train(self, my_verbose=False):
        # --------- transform targets ----------- # 
        le = preprocessing.LabelEncoder()
        
        # ---------- make train test split --------- #
        #X_train, X_test, y_train, y_test = train_test_split(self.data, self.__targets, test_size=0.3)
        parameters = {'kernel':('poly', 'rbf'), 'C':[1, 10], 'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]}
        svc = svm.SVC(probability=True)
        model = GridSearchCV(svc, parameters, cv=5)
        # ---------- BUILD SVM MODEL ------------ #
        # clf = svm.SVC(  C = 10,
        #                 kernel = 'poly', 
        #                 verbose = my_verbose,
        #                 gamma = 0.00000000000000001,
        #                 probability=True) 

        # ---------- Train the model using the training sets ------------ #
        model.fit(self.train_data, le.fit_transform(self.train_targets))

        # ----------- Predict the response for test dataset ------------ #
        y_pred = model.predict(self.test_data)
        
        # ---------- Retain probabilities ------------------------------- #
        self.__predicted = model.predict_proba(self.test_data)
        
        # print(self.__predicted)
        # self.classes = model.classes_
        # print(self.classes)

        __prbList = list()

        for p in self.__predicted:
            probs = {}
            for i in range(len(p)):
                probs[le.inverse_transform([i])[0]] = p[i]
            __prbList.append(probs)

        # __prbList = [dict(t) for t in {tuple(d.items()) for d in __prbList}]
        
        # print(__prbList)

        # print(len(__prbList))
        # print()
        # print(__prbList) 

        print('CORPUS_TS size: ', self.corpus_ts.get_size())        
        
        for i, j in zip(__prbList, self.corpus_ts):
            j.set_intent_probabilities(i)

        print('----------- Intent probabilities set is complete ---------------')

        # ---------- MODEL ACCURACY ----------- #  
        
        print("Accuracy:",metrics.accuracy_score(le.fit_transform(self.test_targets), y_pred))

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Precision:",metrics.precision_score(le.fit_transform(self.test_targets), y_pred, average='micro'))

        # # Model Recall: what percentage of positive tuples are labelled as such?
        print("Recall:",metrics.recall_score(le.fit_transform(self.test_targets), y_pred, average='micro'))

        print("\n")
        print(classification_report(le.fit_transform(self.test_targets), y_pred))


        print('TRAIN DATA')
        #for i in range(7):
        #    print(i,le.fit_transform(self.train_targets).count(i))    

        print('TEST DATA')
        print(y_pred)
        print()
        print(le.fit_transform(self.test_targets))

        # predictions_test = le.inverse_transform(y_pred)
        # print(predictions_test)

        # 'AddToPlaylist' 0 'BookRestaurant' 1 'get weather' 2 'PlayMusic'3 
        # 'RateBook' 4 'SearchCreativeWork' 5 'SearchScreeningEvent' 6 

    def setup(self):

        # ---- grab all utterances ----
        self.all_sent_tr = list()
        for inst in self.corpus_tr:
            self.all_sent_tr.append(inst.get_utterance())

        # ---- grab all utterances ----
        self.all_sent_ts = list()
        for inst in self.corpus_ts:
            self.all_sent_ts.append(inst.get_utterance())

        self.all_sent_combined = self.all_sent_tr + self.all_sent_ts
        self.vocab = set()
        for sent in self.all_sent_combined:
            self.vocab.update(sent.split())

        # ---- train feature ----
        feat = Feature(self.vocab)

        self.feature_tr = feat.create_tfidf(self.all_sent_tr)

        # ---- test feature ---- 
        self.feature_ts = feat.create_tfidf(self.all_sent_ts)

        # print(feature_ts.create_tfidf(all_sent_ts))

        # ---- grab all train targets ----
        self.all_targ_tr = list()
        for inst in self.corpus_tr:
            self.all_targ_tr.append(inst.get_gold_intent())

        # print(all_targ_tr)

        # ---- grab all test targets ----
        self.all_targ_ts = list()
        for inst in self.corpus_ts:
            self.all_targ_ts.append(inst.get_gold_intent())


        # --------------------------------------- #

        # all_data = feature_tr.append(feature_ts)
        # all_targ = all_targ_tr + all_targ_ts

# ------------- enter all data and all target as input to SVM ----------------- #

if __name__ == '__main__':
    pass
    # tr = Corpus(70,'train')
    # ts = Corpus(20, 'test')
    # baseline_svm = SVM(tr,ts)
    # baseline_svm.setup()
    # baseline_svm.train()