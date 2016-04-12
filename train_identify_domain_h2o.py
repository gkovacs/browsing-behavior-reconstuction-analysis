#!/usr/bin/env python
# md5: c8864930a096ab388f167e1cd2c0a0ef
# coding: utf-8

import csv
import sys
import traceback
import os


data_version = int(sys.argv[1])


from tmilib import *





import h2o
import h2o.grid
h2o.init(port=int(os.environ.get('h2o_port', 54321)))



train_dataset = sdir_path('domainclass_cpn_train_v' + str(data_version) +'.csv')


def train_classifier(model_name):
  model_file = sdir_path(model_name)
  if path.exists(model_file):
    print 'already exists', model_name
    return
  print model_name
  #global train_dataset
  #train_dataset = sdir_path('catdata_train_tensecond_v2.csv')
  classifier = get_classifier()
  print classifier
  h2o.save_model(classifier, model_file)
  


classifier_algorithm = lambda: h2o.estimators.H2ORandomForestEstimator(build_tree_one_node=True)

def get_classifier():
  classifier = classifier_algorithm() #h2o.estimators.H2ORandomForestEstimator(binomial_double_trees=True)
  training_data = h2o.import_file(train_dataset)
  test_data = h2o.import_file(train_dataset.replace('train', 'test'))
  classifier.train(x=training_data.columns[1:], y=training_data.columns[0], training_frame=training_data, validation_frame=test_data)
  return classifier


train_classifier('domainclass_cpn_v' + str(data_version) + '_randomforest_v1.h2o')

