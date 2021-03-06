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
        
        # ---------- BUILD SVM MODEL ------------ #
        model = svm.SVC(  C = 1,
                        kernel = 'poly', 
                        verbose = my_verbose,
                        gamma = 1,
                        probability=True) 

        # ---------- Train the model using the training sets ------------ #
        model.fit(self.train_data, le.fit_transform(self.train_targets))

        # ----------- Predict the response for test dataset ------------ #
        y_pred = model.predict(self.test_data)
        
        # ---------- Retain probabilities ------------------------------- #
        self.__predicted = model.predict_proba(self.test_data)

        __prbList = list()

        for p in self.__predicted:
            probs = {}
            for i in range(len(p)):
                probs[le.inverse_transform([i])[0]] = p[i]
            __prbList.append(probs)

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
            

        print('TEST DATA')
        print(y_pred)
        print()
        print(le.fit_transform(self.test_targets))

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

if __name__ == '__main__':
    pass
    tr = Corpus(9,'train')
    ts = Corpus(2, 'test')
    baseline_svm = SVM(tr,ts)
    baseline_svm.setup()
    print(baseline_svm.feature_tr)
    print(len(baseline_svm.feature_tr[0]))
    baseline_svm.train()