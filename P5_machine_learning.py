import sys
import pickle
import json
import numpy as np


sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from tester import test_classifier

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
            pickle.dump(estimator.best_estimator_._final_estimator, clf_outfile)
        with open(filename+"_best_result.txt", "w") as results:
            results.write("best score: " + str(estimator.best_score_))
        with open(filename+"_results.json", "w") as results:
            delete = ""
            aux = estimator.cv_results_
            test = np.array([1])
            testma = np.ma.array([1])
            for elem in aux:
                if type(aux[elem]) == type(test):
                    aux[elem] = aux[elem].tolist()
                if type(aux[elem]) == type(testma):
                    aux[elem] = aux[elem].tolist()
            json.dump(aux, results)

    def classifier(self, estimator, param_grid, estimator_file):
        estimators = [('reduce_dim', PCA()), ('classify', estimator())]
        pipe = Pipeline(estimators)
        clf = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        clf.fit(self.features, self.labels)
        self.save_estimator(estimator_file, clf)

    def trying_classifiers(self):
        param_grid = {
            'classify__max_features': ["auto", "sqrt", "log2", None]
        }
        self.classifier(DecisionTreeClassifier, param_grid, "tree_estimator")

        param_grid = {}
        self.classifier(GaussianNB, param_grid, "gaussian_estimator")

        param_grid = {}
        self.classifier(LinearRegression, param_grid, "linear_estimator")

        param_grid = {
            'classify__hidden_layer_sizes': [10, 50, 100, 150, 200],
            'classify__activation': ['logistic', 'relu'],
            'classify__solver': ['sgd', 'adam'],
            'classify__alpha': [0.0001, 0.001, 0.01],
            'classify__max_iter': [150, 200, 250]
        }
        self.classifier(MLPClassifier, param_grid, "neural_estimator")

        param_grid = {
            'classify__C': [0.01, 0.05, 0.1, 0.5, 1, 3],
            'classify__max_iter': [75, 100, 125, 150, 200, 250],
            'classify__solver': ["newton-cg", "lbfgs", "liblinear", "sag"]
        }
        self.classifier(LogisticRegression, param_grid, "logistic_estimator")

    def main(self):
        self.load_dataset()
        ### Task 2: Remove outliers
        self.outliers()

        data = featureFormat(self.data_dict, self.features_list, sort_keys=True)
        self.labels, self.features = targetFeatureSplit(data)

        ### Task 4: Try a varity of classifiers
        #self.trying_classifiers()
        self.tree()


    def tree(self):
        param_grid = {
            'max_features': ["auto", "sqrt", "log2", None],
            'criterion': ["gini", "entropy"],
            'class_weight': ["balanced", None],
            'min_impurity_split': [1e-8, 1e-7, 1e-6, 1e-5],
            'presort': [True, False]
        }
        best_params = {
            'max_features': "",
            'criterion': "",
            'class_weight': "",
            'min_impurity_split': "",
            'presort': "",
            'results': [0,0,0]
        }
        for elem in param_grid['max_features']:
            for criter in param_grid['criterion']:
                for weight in param_grid['class_weight']:
                    for impurity in param_grid['min_impurity_split']:
                        for sort in param_grid['presort']:

                            clf = DecisionTreeClassifier(max_features=elem, criterion=criter,
                                                         class_weight=weight, min_impurity_split=impurity,
                                                         presort=sort)
                            clf.fit(self.features, self.labels)
                            acu, prec, rec = test_classifier(clf, self.data_dict, self.features_list)
                            if acu >=best_params['results'][0]:
                                if prec >= best_params['results'][1]:
                                    if rec >= best_params['results'][2]:
                                        best_params['results'] = [acu, prec, rec]
                                        best_params['max_features'] = elem
                                        best_params['criterion'] = criter
                                        best_params['class_weight'] = weight
                                        best_params['min_impurity_split'] = impurity
                                        best_params['presort'] = sort
        print best_params




if __name__ == '__main__':
    final_project = Final_project()
    final_project.main()

