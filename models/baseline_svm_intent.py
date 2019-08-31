from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics


class SVM:
    def __init__(self, data, targets):
        self.data = data # assuming that the things are coming in fine 
        self.targets = targets

    def train(self):
        
        # --------- transform targets ----------- # 
        le = preprocessing.LabelEncoder()
        self.__targets = self.targets  
        self.__targets = le.fit_transform(self.__targets)
        
        # ---------- make train test split --------- #
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.__targets, test_size=0.3,random_state=109)


        # ---------- BUILD SVM MODEL ------------ #
        clf = svm.SVC(kernel='linear') # Linear Kernel

        # ---------- Train the model using the training sets ------------ #
        clf.fit(X_train, y_train)
        
        # ----------- Predict the response for test dataset ------------ #
        y_pred = clf.predict(X_test)
        
        # ---------- MODEL ACCURACY ----------- #  
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))

        # # Model Recall: what percentage of positive tuples are labelled as such?
        # print("Recall:",metrics.recall_score(y_test, y_pred))