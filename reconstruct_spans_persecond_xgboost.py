#!/usr/bin/env python
# md5: 8e9d69bb084321702e622bbb1f5ad94b
# coding: utf-8

from tmilib import *

from reconstruct_focus_times_common import *
from sorted_collection import SortedCollection
from rescuetime_utils import *

import sklearn
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.tree
import xgboost

from math import log

import datetime
import time

import cPickle as pickle


import traceback
import sys


training_users = get_training_users()
test_users = get_test_users()


#print domain_to_id('www.facebook.com')
#print id_to_domain(29708)







def extract_secondlevel_dataset_from_users(users):
  all_labels = []
  all_sinceprev = []
  all_tonext = []
  all_fromdomain = []
  all_todomain = []
  for user in users:
    data = extract_secondlevel_dataset_from_user(user)
    all_labels.extend(data['label'])
    all_sinceprev.extend(data['sinceprev'])
    all_tonext.extend(data['tonext'])
    all_fromdomain.extend(data['fromdomain'])
    all_todomain.extend(data['todomain'])
  return {
    'label': all_labels,
    'sinceprev': all_sinceprev,
    'tonext': all_tonext,
    'fromdomain': all_fromdomain,
    'todomain': all_todomain,
  }

@jsonmemoized
def extract_secondlevel_dataset_for_training():
  return extract_secondlevel_dataset_from_users(training_users)

@jsonmemoized
def extract_secondlevel_dataset_for_test():
  return extract_secondlevel_dataset_from_users(test_users)


def extract_tensecondlevel_dataset_from_users(users):
  all_labels = []
  all_sinceprev = []
  all_tonext = []
  all_fromdomain = []
  all_todomain = []
  for user in users:
    data = extract_secondlevel_dataset_from_user(user, True)
    all_labels.extend(data['label'])
    all_sinceprev.extend(data['sinceprev'])
    all_tonext.extend(data['tonext'])
    all_fromdomain.extend(data['fromdomain'])
    all_todomain.extend(data['todomain'])
  return {
    'label': all_labels,
    'sinceprev': all_sinceprev,
    'tonext': all_tonext,
    'fromdomain': all_fromdomain,
    'todomain': all_todomain,
  }

@jsonmemoized
def extract_tensecondlevel_dataset_for_training():
  return extract_tensecondlevel_dataset_from_users(training_users)

@jsonmemoized
def extract_tensecondlevel_dataset_for_test():
  return extract_tensecondlevel_dataset_from_users(test_users)





#a=extract_secondlevel_training_data_from_user(training_users[0])
#print 'extracting dataset for training'
#extract_secondlevel_dataset_for_training()
#extract_tensecondlevel_dataset_for_training()
#print 'extracting dataset for test'
#extract_secondlevel_dataset_for_test()
#extract_tensecondlevel_dataset_for_test()
#print 'extraction done'


#print zipkeys({'a':[3,4,5], 'b':[6,7,8]}, 'a', 'b')


'''
@memoized
def total_usage_of_domains_in_training():
  return sum_values_in_list_of_dict([get_domain_to_time_spent_for_user(user)for user in training_users])

@memoized
def top_n_domains_by_usage(n=10):
  domain_to_usage = total_usage_of_domains_in_training()
  return [x[0] for x in sorted(domain_to_usage.items(), key=itemgetter(1), reverse=True)[:n]]
'''


#print top_n_domains_by_visits()


#print top_n_domains_by_usage(10)
#print top_n_domains_by_usage(5)


#print domain_to_id('newtab')


'''
def dataset_to_feature_vectors(dataset):
  topdomains = numpy.array([domain_to_id(x) for x in top_n_domains_by_visits(20)])
  num_features = 3 + 2*len(topdomains) + 2*len(get_rescuetime_productivity_levels())
  output = [[0]*num_features for x in xrange(len(dataset['sinceprev']))]
  #output = numpy.zeros((len(dataset['sinceprev']), num_features), dtype=object) # object instead of float, so we can have floats and ints
  for idx,sinceprev,tonext,fromdomain,todomain in zipkeys_idx(dataset, 'sinceprev', 'tonext', 'fromdomain', 'todomain'):
    cur = output[idx]
    cur[0] = sinceprev
    cur[1] = tonext
    cur[2] = int(fromdomain == todomain)
    feature_num = 3
    for domain_idx,domain in enumerate(topdomains):
      cur[feature_num+domain_idx] = int(fromdomain == domain)
    feature_num += len(topdomains)
    for domain_idx,domain in enumerate(topdomains):
      cur[feature_num+domain_idx] = int(todomain == domain)
    feature_num += len(topdomains)
    fromdomain_name = id_to_domain(fromdomain)
    todomain_name = id_to_domain(todomain)
    fromdomain_productivity = domain_to_productivity(fromdomain_name)
    todomain_productivity = domain_to_productivity(todomain_name)
    for productivity_idx,productivity in enumerate(get_rescuetime_productivity_levels()):
      cur[feature_num+productivity_idx] = int(fromdomain_productivity == productivity)
    feature_num += len(get_rescuetime_productivity_levels())
    for productivity_idx,productivity in enumerate(get_rescuetime_productivity_levels()):
      cur[feature_num+productivity_idx] = int(todomain_productivity == productivity)
    feature_num += len(get_rescuetime_productivity_levels())
  return output
'''

def remove_cached_features():
  os.remove('get_test_feature_vector.msgpack')
  os.remove('get_training_feature_vector.msgpack')
  #os.remove('get_test_labels.msgpack')
  #os.remove('get_training_labels.msgpack')


#remove_cached_features()


'''
@msgpackmemoized
def get_training_feature_vector():
  return dataset_to_feature_vectors(extract_tensecondlevel_dataset_for_training())

@msgpackmemoized
def get_training_labels():
  return extract_tensecondlevel_dataset_for_training()['label']

@msgpackmemoized
def get_test_feature_vector():
  return dataset_to_feature_vectors(extract_tensecondlevel_dataset_for_test())

@msgpackmemoized
def get_test_labels():
  return extract_tensecondlevel_dataset_for_test()['label']
'''


'''
def train_classifier_on_data(training_data):
  #classifier = sklearn.tree.DecisionTreeClassifier(max_depth=2) # .71 on test
  #classifier = sklearn.tree.DecisionTreeClassifier(max_depth=1)
  #classifier = sklearn.tree.DecisionTreeClassifier()
  #classifier = sklearn.naive_bayes.GaussianNB() # .73 on test
  #classifier = sklearn.svm.LinearSVC()
  #classifier = sklearn.linear_model.SGDClassifier(class_weight='balanced')
  classifier = sklearn.linear_model.SGDClassifier(loss='modified_huber') # .73 on test
  classifier.fit(dataset_to_feature_vectors(training_data), numpy.array(training_data['label']))
  return classifier

def get_classifier():
  return train_classifier_on_data(extract_tensecondlevel_dataset_for_training())
'''

global_feature_filter = '1'*53
#global_feature_filter = '11100000000000000000000000000000000000000000000000000'

def get_feature_filter():
  return global_feature_filter
  return ''.join(map(str, selected_features))

is_second = len(sys.argv) > 1 and sys.argv[1] == 'second'

@memoized
def get_filtered_features_train():
  selected_features = get_feature_filter()
  if is_second:
    return numpy.array(get_feature_vector_for_secondlevel_train(selected_features))
  return numpy.array(get_feature_vector_for_tensecondlevel_train(selected_features))

@memoized
def get_filtered_features_test():
  selected_features = get_feature_filter()
  if is_second:
    return numpy.array(get_feature_vector_for_secondlevel_test(selected_features))
  return numpy.array(get_feature_vector_for_tensecondlevel_test(selected_features))

@memoized
def get_test_labels():
  if is_second:
    return numpy.array(get_labels_for_secondlevel_test())
  return numpy.array(get_labels_for_tensecondlevel_test())

@memoized
def get_training_labels():
  if is_second:
    return numpy.array(get_labels_for_secondlevel_train())
  return numpy.array(get_labels_for_tensecondlevel_train())

#def get_labels_for_user(user):
#  return get_tensecondlevel_activespan_labels_for_user(user)

'''
# we normally want to use the tensecond level ones
def get_filtered_features_train():
  selected_features = get_feature_filter()
  return get_feature_vector_for_secondlevel_train(selected_features)

def get_filtered_features_test():
  selected_features = get_feature_filter()
  return get_feature_vector_for_secondlevel_test(selected_features)

def get_test_labels():
  return get_labels_for_secondlevel_test()

def get_training_labels():
  return get_labels_for_secondlevel_train()

def get_labels_for_user(user):
  return get_secondlevel_activespan_labels_for_user(user)
'''


'''
def filter_features(arr):
  # from get_selected_features()
  selected_features = get_feature_filter()
  #selected_feature_idx = [i for i,x in enumerate(selected_features) if x]
  #return arr[:,selected_feature_idx]
  output = []
  for line in arr:
    output.append([line[i] for i,x in enumerate(selected_features) if x])
    #output.append([x for i,x in enumerate(line) if selected_features[i]])
  return output
'''

classifier_algorithm = xgboost.XGBClassifier
xgboost_params = {}

def get_classifier():
  #classifier = sklearn.naive_bayes.GaussianNB()
  #classifier = sklearn.linear_model.SGDClassifier(loss='modified_huber') # .73 on test
  #classifier = sklearn.linear_model.SGDClassifier()
  #classifier = sklearn.ensemble.RandomForestClassifier()
  classifier = xgboost.XGBClassifier()
  classifier.set_params(**xgboost_params)
  #classifier.train()
  #classifier = sklearn.ensemble.AdaBoostClassifier()
  #classifier = sklearn.ensemble.GradientBoostingClassifier()
  #classifier = classifier_algorithm()
  classifier.fit(get_filtered_features_train(), get_training_labels())
  return classifier








#a=get_filtered_features_test()


#print len(a)


#b=get_test_labels()


#print len(b)


#print len(get_feature_filter())


#print len(get_training_feature_vector()[0])


'''
def make_predictions_with_classifier_on_dataset(classifier, dataset):
  return classifier.predict(dataset_to_feature_vectors(dataset))

def make_proba_predictions_with_classifier_on_dataset(classifier, dataset):
  return [x[1] for x in classifier.predict_proba(dataset_to_feature_vectors(dataset))]

def evaluate_classifier(classifier):
  test_predictions = make_predictions_with_classifier_on_dataset(classifier, test_data)
  print sklearn.metrics.classification_report(test_data['label'], test_predictions)
'''

def make_predictions_with_classifier_on_test(classifier):
  #return classifier.predict(filter_features(numpy.array(get_test_feature_vector())))
  return classifier.predict(get_filtered_features_test())

def make_predictions_with_classifier_on_train(classifier):
  #return classifier.predict(filter_features(numpy.array(get_test_feature_vector())))
  return classifier.predict(get_filtered_features_train())


def evaluate_classifier(classifier):
  test_predictions = make_predictions_with_classifier_on_test(classifier)
  print sklearn.metrics.classification_report(get_test_labels(), test_predictions)

def evaluate_classifier_train(classifier):
  train_predictions = make_predictions_with_classifier_on_train(classifier)
  print sklearn.metrics.classification_report(get_training_labels(), test_predictions)
  
def evaluate_classifier_for_user(classifier, user):
  dataset = extract_secondlevel_dataset_from_user(user, True)
  test_labels = get_labels_for_user(user)
  





#a= get_training_feature_vector()








'''
global_feature_filter = '1'*53
#classifier_algorithm = xgboost.XGBClassifier()
xgboost_params = {}

pickle_file = 'classifier_allfeatures_xgboost.pickle'
if path.exists(pickle_file):
  classifier = pickle.load(open(pickle_file))
else:
  classifier = get_classifier()
  pickle.dump(classifier, open(pickle_file, 'w'), pickle.HIGHEST_PROTOCOL)

evaluate_classifier(classifier)
'''


'''
xgboost_params = {'objective': "binary:logistic"}

pickle_file = 'classifier_allfeatures_xgboost_binary_logistic.pickle'
if path.exists(pickle_file):
  classifier = pickle.load(open(pickle_file))
else:
  classifier = get_classifier()
  pickle.dump(classifier, open(pickle_file, 'w'), pickle.HIGHEST_PROTOCOL)

evaluate_classifier(classifier)
'''


xgboost_params = {'objective': "binary:logistic"}
for colsample_bytree in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
  for subsample in shuffled([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]):
    xgboost_params['subsample'] = subsample
    xgboost_params['colsample_bytree'] = colsample_bytree
    features_string = '_'.join(map(str, ['subsample', subsample, 'colsample_bytree', colsample_bytree]))
    if is_second:
      classifier_filename = 'xgboost_second_allfeatures_' + features_string + '.pickle'
    else:
      classifier_filename = 'xgboost_tensecond_allfeatures_' + features_string + '.pickle'
    if path.exists(classifier_filename):
      continue
    print classifier_filename
    try:
      classifier = get_classifier()
      pickle.dump(classifier, open(classifier_filename, 'w'), pickle.HIGHEST_PROTOCOL)
      evaluate_classifier(classifier)
    except:
      traceback.print_exc()
      continue


'''
xgboost_params = {'max_depth': 4, 'num_parallel_tree': 1000, 'subsample': 0.5, 'colsample_bytree': 0.5, 'nround': 1, 'objective': "binary:logistic"}

pickle_file = 'classifier_allfeatures_xgboost_randomforest.pickle'
if path.exists(pickle_file):
  classifier = pickle.load(open(pickle_file))
else:
  classifier = get_classifier()
  pickle.dump(classifier, open(pickle_file, 'w'), pickle.HIGHEST_PROTOCOL)

evaluate_classifier(classifier)
'''


















































































