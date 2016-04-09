#!/usr/bin/env python
# md5: 917a616f23114232d7cd8c5480c365a4
# coding: utf-8

from tmilib import *
import cPickle as pickle


is_second = True
is_insession = True


def get_reference_labels_for_user(user):
  if is_second:
    if is_insession:
      return get_secondlevel_activespan_dataset_insession_labels_for_user(user)
    else:
      return get_secondlevel_activespan_dataset_labels_for_user(user)
  else:
    if is_insession:
      return get_tensecondlevel_activespan_dataset_insession_labels_for_user(user)
    else:
      return get_tensecondlevel_activespan_dataset_labels_for_user(user)

def get_predicted_labels_for_user(user):
  if is_second:
    if is_insession:
      feature_vector = get_feature_vector_for_secondlevel_insession_allfeatures_for_user(user)
    else:
      feature_vector = get_feature_vector_for_secondlevel_allfeatures_for_user(user)
  else:
    if is_insession:
      feature_vector = get_feature_vector_for_tensecondlevel_insession_allfeatures_for_user(user)
    else:
      feature_vector = get_feature_vector_for_tensecondlevel_allfeatures_for_user(user)
  return classifier.predict(feature_vector)

def get_stats_for_user(user):
  stats = Counter()
  for ref,rec in zip(get_reference_labels_for_user(user), get_predicted_labels_for_user(user)):
    if ref == True and rec == True:
      stats['tp'] += 1
      continue
    if ref == False and rec == False:
      stats['tn'] += 1
      continue
    if ref == True and rec == False:
      stats['fn'] += 1
      continue
    if ref == False and rec == True:
      stats['fp'] += 1
      continue
  return stats


def get_stats_for_all_users():
  stats = Counter()
  for user in get_test_users():
    for k,v in get_stats_for_user(user).viewitems():
      stats[k] += v
  return stats





def print_evaluation_stats(stats):
  tp = float(stats['tp'])
  tn = float(stats['tn'])
  fp = float(stats['fp'])
  fn = float(stats['fn'])
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1 = 2*(precision*recall)/(precision+recall)
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  print 'precision', precision
  print 'recall', recall
  print 'f1', f1
  print 'accuracy', accuracy
  print 'tp', tp
  print 'tn', tn
  print 'fp', fp
  print 'fn', fn

#print_evaluation_stats(overall_stats)


for pickle_file in sdir_glob('scikit*.pickle'):
  #pickle_file = 'classifier_threefeatures_randomforest_v2.pickle'
  print pickle_file
  classifier = pickle.load(open(pickle_file))
  #user = get_test_users()[0]
  #print user
  #print get_stats_for_user(user)
  overall_stats = get_stats_for_all_users()
  print_evaluation_stats(overall_stats)







