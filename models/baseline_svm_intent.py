from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report


class SVM:
    def __init__(self, train_data, train_targets, test_data, test_targets):
        self.train_data = train_data
        self.train_targets = train_targets
        self.test_data = test_data
        self.test_targets = test_targets

    def train(self):
        
        # --------- transform targets ----------- # 
        le = preprocessing.LabelEncoder()
        
        # ---------- make train test split --------- #
        #X_train, X_test, y_train, y_test = train_test_split(self.data, self.__targets, test_size=0.3)

        # ---------- BUILD SVM MODEL ------------ #
        clf = svm.SVC(  C = 10,
                        kernel = 'poly', 
                        verbose = True,
                        gamma = 0.00000000000000001,
                        probability=True) # Linear Kernel

        # ---------- Train the model using the training sets ------------ #
        clf.fit(self.train_data, le.fit_transform(self.train_targets))

        # ----------- Predict the response for test dataset ------------ #
        y_pred = clf.predict(self.test_data)
        
        # ---------- MODEL ACCURACY ----------- #  
        print()
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