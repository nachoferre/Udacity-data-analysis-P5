import sys
import pickle

sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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

    def save_estimator(self, filename, estimator):
        with open(filename+".pkl", "w") as clf_outfile:
            pickle.dump(estimator.best_estimator_, clf_outfile)
        with open(filename+"_results.txt", "w") as results:
            results.write("best score: " + estimator.best_score_)
            results.write("best params: " + estimator.best_estimator_)

    def main(self):
        self.load_dataset()
        ### Task 2: Remove outliers
        self.outliers()

        data = featureFormat(self.data_dict, self.features_list, sort_keys=True)
        self.labels, self.features = targetFeatureSplit(data)

        ### Task 4: Try a varity of classifiers
        features_train, features_test, labels_train, labels_test = \
            train_test_split(self.features, self.labels, test_size=0.3, random_state=42)

        estimators = [('reduce_dim', PCA()), ('classify', DecisionTreeClassifier())]
        pipe = Pipeline(estimators)
        param_grid = [
            {
                'classify__max_features': "auto"
            },

        ]
        clf = GridSearchCV(pipe, param_grid, n_jobs=3, verbose=10)
        clf.fit(self.features, self.labels)
        self.save_estimator("tree_estimator", clf)

        estimators = [('reduce_dim', PCA()), ('classify', SVC())]
        pipe = Pipeline(estimators)
        param_grid = [
            {
                'classify__C': [0.01, 1],
                'classify__kernel': ['linear']
                #'classify__C': [0.01, 1, 5, 10],
                #'classify__kernel': ['linear', 'rbf']
            },

        ]
        clf = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        clf.fit(self.features, self.labels)
        self.save_estimator("vector_estimator", clf)

        estimators = [('reduce_dim', PCA()), ('classify', GaussianNB())]
        pipe = Pipeline(estimators)
        param_grid = [{}, ]
        clf = GridSearchCV(pipe, param_grid, n_jobs=3, verbose=10)
        clf.fit(self.features, self.labels)
        self.save_estimator("gaussian_estimator", clf)

        estimators = [('reduce_dim', PCA()), ('classify', LinearRegression())]
        pipe = Pipeline(estimators)
        param_grid = [{}, ]
        clf = GridSearchCV(pipe, param_grid, n_jobs=3, verbose=10)
        clf.fit(self.features, self.labels)
        self.save_estimator("linear_estimator", clf)

        estimators = [('reduce_dim', PCA()), ('classify', MLPClassifier())]
        pipe = Pipeline(estimators)
        param_grid = [
            {
                'classify__hidden_layer_sizes': [10, 50, 100, 150, 200],
                'activation': ['logistic', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.00001, 0.0001, 0.001, 0.01],
                'max_iter': [150, 200, 250],
                'verbose': True
             },

        ]
        clf = GridSearchCV(pipe, param_grid, n_jobs=3, verbose=10)
        clf.fit(self.features, self.labels)
        self.save_estimator("neural_estimator", clf)


        print "aaa"


if __name__ == '__main__':
    final_project = Final_project()
    final_project.main()
