
# loading the dataset
import sys
import pickle
import pprint as pp
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from tester import test_classifier

scaler = MinMaxScaler()
kbest = SelectKBest()
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Explore the dataset
#print "TOTAL NUMBER OF PERSONS: ", len(my_dataset)
#print "TOTAL NUMBER OF FEATURES: ", len(my_dataset[my_dataset.keys()[1]])
#print "LIST OF FEATURES:"
#pp.pprint(my_dataset[my_dataset.keys()[1]].keys())
all_poi = 0
num_missing_values = {}
for name in my_dataset:
    if my_dataset[name]['poi'] == True:
        all_poi += 1;
    for feature in my_dataset[name]:
        if my_dataset[name][feature] == "NaN":
            if not feature in num_missing_values:
                num_missing_values[feature] = 1
            else:
                num_missing_values[feature] += 1

#print "TOTAL NUMBER OF PERSON OF INTEREST: ", all_poi
num_missing_values = {}
for name in my_dataset:
    for feature in my_dataset[name]:
        if my_dataset[name][feature] == "NaN":
            if not feature in num_missing_values:
                num_missing_values[feature] = 1
            else:
                num_missing_values[feature] += 1
#print "COUNTS OF MISSING ENTRIES IN DIFFERENT FEATURES: "
#pp.pprint(num_missing_values)


# ### OUTLIER DETECTION

# Delete the outliers
del my_dataset['LOCKHART EUGENE E']
del my_dataset['TOTAL']
del my_dataset['THE TRAVEL AGENCY IN THE PARK']


# ### Features list

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'long_term_incentive',
                 'loan_advances', 'bonus', 'restricted_stock', 'restricted_stock_deferred',
                 'total_stock_value', 'exercised_stock_options',
                 'deferred_income', 'expenses', 'director_fees']
#pp.pprint(features_list)


# creating new features
def computeFraction(poi_messages, all_messages):
    """ computes fraction of poi with give messages type and total of that message type """
    fraction = 0.
    if all_messages != "NaN":
        fraction = float(poi_messages)/float(all_messages)
    else:
        fraction = 0
    return fraction

for name in data_dict:
    from_poi_to_this_person = my_dataset[name]["from_poi_to_this_person"]
    to_messages = my_dataset[name]["to_messages"]
    my_dataset[name]["fraction_from_poi"] = computeFraction(from_poi_to_this_person, to_messages)

    from_this_person_to_poi = my_dataset[name]["from_this_person_to_poi"]
    from_messages = my_dataset[name]["from_messages"]
    my_dataset[name]["fraction_to_poi"] = computeFraction(from_this_person_to_poi, from_messages)
features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



# decision tree
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=42)
pipe = Pipeline([('kbest', kbest), ('clf_dt', clf_dt)])
parameters = {'kbest__k':range(5,15), 'clf_dt__max_depth':range(2,6), 'clf_dt__min_samples_leaf':range(1,5)}
folder = StratifiedShuffleSplit(labels, n_iter = 100, random_state = 40)
grid = GridSearchCV(pipe, param_grid=parameters, cv = folder, scoring = 'f1')
grid.fit(features, labels)
clf = grid.best_estimator_
test_classifier(clf, my_dataset, features_list)



# K-NN Learner
from sklearn.neighbors import KNeighborsClassifier
scaler = MinMaxScaler()
kbest = SelectKBest()
clf_knn = KNeighborsClassifier()
pipe = Pipeline([('scaler', scaler), ('kbest', kbest), ('clf_knn', clf_knn)])
parameters = {'kbest__k':range(5,15), 'clf_knn__n_neighbors':range(3,8)}
folder = StratifiedShuffleSplit(labels, n_iter = 100, random_state = 40)
grid = GridSearchCV(pipe, param_grid=parameters, cv = folder, scoring = 'f1')
grid.fit(features, labels)
clf = grid.best_estimator_
test_classifier(clf, my_dataset, features_list)


# Gaussian Naive Bayes 

kbest = SelectKBest()
clf_nb = GaussianNB()
pipe = Pipeline([('kbest', kbest), ('clf_nb', clf_nb)])
parameters = {'kbest__k':range(5,15)}
folder = StratifiedShuffleSplit(labels, n_iter = 100, random_state = 40)
grid = GridSearchCV(pipe, param_grid=parameters, cv = folder, scoring = 'f1')
grid.fit(features, labels)
clf = grid.best_estimator_
test_classifier(clf, my_dataset, features_list)


#print "scores with different values of K i.e numbers of features"
grid.grid_scores_


kbest_exp = SelectKBest(k=6)
kbest_exp.fit(features, labels)
features_selected = [features_list[i+1] for i in kbest_exp.get_support(indices=True)]
#print 'features selected by kbest'
#print features_selected


# To find effect of our new feature, we find best score without our new features


# remove 2 new features
features_list.remove("fraction_from_poi")
features_list.remove("fraction_to_poi")

from sklearn.naive_bayes import GaussianNB

kbest = SelectKBest()
clf_nb = GaussianNB()
pipe = Pipeline([('kbest', kbest), ('clf_nb', clf_nb)])
parameters = {'kbest__k':range(5,15)}
folder = StratifiedShuffleSplit(labels, n_iter = 100, random_state = 42)
grid = GridSearchCV(pipe, param_grid=parameters, cv = folder, scoring = 'f1')
grid.fit(features, labels)
clf1 = grid.best_estimator_
test_classifier(clf, my_dataset, features_list)


# We can see that Precision score is decreased to 0.40 from 0.42 and Recall is decreases to 0.32 from 0.35 without our new features confirming usefulness of our new feature.
dump_classifier_and_data(clf, my_dataset, features_list)