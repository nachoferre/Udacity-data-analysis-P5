import sys
import pickle

from sklearn.decomposition import NMF
from sklearn.feature_selection import SelectKBest, chi2

sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

class Final_project:


    def __init__(self):
        self.data_file = "final_project_dataset.pkl"
        ### Task 1: Select what features you'll use.
        ### features_list is a list of strings, each of which is a feature name.
        ### The first feature must be "poi".
        self.features_list = ['poi', 'salary', 'total_payments', 'bonus', 'total_stock_value', 'expenses',
                 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
        # We have chosen a big deal of features so we can work with the pca

        # In this variable we will load the whole dataset we are going to work with
        self.labels = ""
        self.features = ""

    def load_dataset(self):
        with open(self.data_file, "r") as data_file:
            self.data_dict = pickle.load(data_file)

    def outliers(self):
        # For the outliers we have to understand that the outliers give us important information about the differences
        # between all the employees. However we have to take out the errors, like one with TOTAL being treated as another employee
        self.data_dict.pop("TOTAL", 0)

    def main(self):
        self.load_dataset()
        ### Task 2: Remove outliers
        self.outliers()

        data = featureFormat(self.data_dict, self.features_list, sort_keys=True)
        self.labels, self.features = targetFeatureSplit(data)

        ### Task 4: Try a varity of classifiers
        features_train, features_test, labels_train, labels_test = \
            train_test_split(self.features, self.labels, test_size=0.3, random_state=42)
        estimators = [('reduce_dim', PCA()), ('classify', SVC())]
        pipe = Pipeline(estimators)
        param_grid = [
            {
                'reduce_dim': [PCA(iterated_power=7), NMF()],
                'reduce_dim__k': [2, 4, 8],
                'classify__C': [0.01, 1, 5, 10],
                'classify__kernel': ['linear', 'rbf']
            },
            {
                'reduce_dim': [SelectKBest(chi2)],
                'reduce_dim__k': [2, 4, 8],
                'classify__C': [0.01, 1, 5, 10],
                'classify__kernel': ['linear', 'rbf']
            },
        ]
        clf = GridSearchCV(pipe, param_grid, cv=3, n_jobs=4)
        clf.fit(self.features, self.labels)
        print "aaa"


if __name__ == '__main__':
    final_project = Final_project()
    final_project.main()
