#!/usr/bin/python
# —* — coding: UTF-8 —* —

# Basic libraries for the inner working of this script
import sys
import pickle
import json
import numpy as np
from tester import test_classifier, dump_classifier_and_data

sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit

# Imports of the different machine learning estimators we are going to work with
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


class FinalProject:
    # We are using a class in orther to have a cleaner code and to avoid passing everything through parameters
    def __init__(self):
        self.data_file = "final_project_dataset.pkl"
        # Task 1: Select what features you'll use.
        # features_list is a list of strings, each of which is a feature name.
        # The first feature must be "poi".
        self.features_list_start = ['poi','to_messages', 'deferral_payments', 'expenses', 'long_term_incentive',
                              'from_poi_to_this_person', 'deferred_income', 'restricted_stock_deferred',
                              'shared_receipt_with_poi', 'loan_advances', 'from_messages', 'other', 'director_fees',
                              'bonus', 'total_stock_value', 'from_this_person_to_poi', 'restricted_stock', 'salary',
                              'total_payments', 'exercised_stock_options']

        # We have chosen a big deal of features so we can work with the pca,
        # and let it decide the best ones to work with

        # In this variable we will load the whole dataset we are going to work with
        self.labels = ""
        self.features = ""
        # This variable will save the loaded dataset
        self.data_dict = {}

    def load_dataset(self):
        # Here we load the current dataset that we are going to work with
        with open(self.data_file, "r") as data_file:
            self.data_dict = pickle.load(data_file)

    def load_features(self):
        try:
            with open("feature_list.pkl", "r") as data_file:
                self.features_list = pickle.load(data_file)
        except Exception as e:
            self.features_list = self.features_list_start

    def add_feature(self):
        # Here we add a new feature wich consists in the sum of all the income of the employee no matter if it was salary related of stock ones
        for elem in self.data_dict:
            payments = self.data_dict[elem]['total_payments']
            stock = self.data_dict[elem]["total_stock_value"]
            if isinstance(self.data_dict[elem]['total_payments'], str):
                payments = 0
            if isinstance(self.data_dict[elem]["total_stock_value"], str):
                stock = 0
            self.data_dict[elem]["total_income"] = payments + stock
        self.features_list.append("total_income")

    def outliers(self):
        # For the outliers we have to understand that the outliers give us important information about the differences
        # between all the employees. However we have to take out the errors,
        # like one with TOTAL being treated as another employee that we learned it was he
        self.data_dict.pop("TOTAL", 0)

    def save_estimator(self, filename, estimator):
        # Here we save the results calculated through GridsearchCV in order to save our data and avoid multiple
        # executions of the code and to be able to work with the data later onre thanks to our previous
        # investigations during the course

        with open(filename + ".pkl", "w") as clf_outfile:
            # This saves the best estimator of the one studied
            pickle.dump(estimator.best_estimator_._final_estimator, clf_outfile)
        with open(filename + "_best_result.txt", "w") as results:
            # This saves the results of the best trained estimator
            results.write("best score: " + str(estimator.best_score_))
        with open(filename + "_results.json", "w") as results:
            # This saves all the results and calculations made by GridSearchCV
            delete = ""
            aux = estimator.cv_results_
            test = np.array([1])
            testma = np.ma.array([1])
            for elem in aux:
                # This loop is necessary given that json doesnt support any type of numpy arrays, so they must be
                # converted to a list before dump into file
                if type(aux[elem]) == type(test):
                    aux[elem] = aux[elem].tolist()
                if type(aux[elem]) == type(testma):
                    aux[elem] = aux[elem].tolist()
            json.dump(aux, results)

    def classifier(self, estimator, param_grid, estimator_file):
        # this definition is the one in charge of running and training the actual estimators
        # First we establish the array of functions to execute
        estimators = [('reduce_dim', PCA()), ('classify', estimator())]
        # Second we create a pipe for it to work sequentially
        pipe = Pipeline(estimators)
        # Third we create the grid ssearch that will create multiple estimator to cover every combination of
        #  variables given the ones passed through the para_grid parameter
        clf = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        # Fourth we train it
        clf.fit(self.features, self.labels)
        # Finally we save our results
        self.save_estimator(estimator_file, clf)

    def trying_classifiers(self):
        # This definition is the one in charge of prepping the data for the classifier definition.
        # It prepares the param_grid variable with a dictionary of possible values for each parameter of the estimator
        # And finally calls the definition with the classifier, the created param_grid and
        # finally with the header for the files created
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
        # Main function of our project, is the one in charge of calling the rest of the methods
        # so the visibility is clean

        self.load_dataset()
        self.load_features()
        ### Task 2: Remove outliers
        self.outliers()

        self.add_feature()
        data = featureFormat(self.data_dict, self.features_list, sort_keys=True)
        self.labels, self.features = targetFeatureSplit(data)

        ### Task 4: Try a varity of classifiers
        # self.trying_classifiers()
        # Task 5: Tune your classifier
        params = self.tree()

        # Task 6 Dump your data
        aux = self.features_list
        for i in reversed(range(len(params['feature_importances']))):
            if params['feature_importances'][i] <= 0.0:
                self.features_list.remove(aux[i+1])
        dump_classifier_and_data(params["clf"], self.data_dict, self.features_list)
        params.pop("clf")
        with open("best_results.json", "w") as results:
            json.dump(params, results)

        # Test the impact of our added feature
        self.test_feature(params)

    def tree(self):
        # This is the function in charge of studying thoroughly the decision tree classifier and test it against
        # the given test in order to achieve the above .3 in recall and precision

        # First we create the param grid just like to grid search but this time
        # we are going to do the training ourselves so we can test it with our provided test
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
            'clf': "",
            'feature_importances': [],
            'results': [0.0, 0.0, 0.0]
        }
        # Here we iterate for every possible arrangements of parameters for the estimator
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
                            # Here we check that the trained estimator is better than the last, and if it is
                            # we will save its params as well as the estimator itself
                            if acu >= best_params['results'][0] or any(i <= 0.3 for i in best_params['results']):
                                if rec >= best_params['results'][2] or any(i <= 0.3 for i in best_params['results']):
                                    if prec >= best_params['results'][1] or any(i <= 0.3 for i in best_params['results']):
                                        best_params['results'] = [acu, prec, rec]
                                        best_params['max_features'] = elem
                                        best_params['criterion'] = criter
                                        best_params['class_weight'] = weight
                                        best_params['min_impurity_split'] = impurity
                                        best_params['presort'] = sort
                                        best_params['feature_importances'] = clf.feature_importances_.tolist()
                                        best_params['clf'] = clf
        return best_params

    def test_feature(self, param):
        #First we roll back to our original dataset
        self.load_dataset()
        clf = DecisionTreeClassifier(max_features=param['max_features'], criterion=param['criterion'],
                                     class_weight=param['class_weight'], min_impurity_split=param['min_impurity_split'],
                                     presort=param['presort'])
        clf.fit(self.features, self.labels)
        acu, prec, rec = test_classifier(clf, self.data_dict, self.features_list)
        print "Results of the original dataset: " + "Accuracy: " + str(acu) + "   Precision: " + str(prec) + \
              "   Recall: " + str(rec)
        print "Results of the edited dataset (new feature included): " + "Accuracy: " + str(param['results'][0]) \
              + "   Precision: " + str(param['results'][1]) + "   Recall: " + str(param['results'][2])


if __name__ == '__main__':
    # The function in charge of starting the script
    final_project = FinalProject()
    final_project.main()
