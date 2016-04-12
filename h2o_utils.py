#!/usr/bin/env python
# md5: d2c9f914febab4be64592eb69f9b793e
# coding: utf-8

import h2o
from glob import glob
from os import path
from tmilib_base import *


def load_h2o_model(model_path):
  if not path.exists(model_path):
    model_path = sdir_path(model_path)
  return h2o.load_model(glob(model_path + '/*')[0])

def load_h2o_data(csv_file):
  if not path.exists(csv_file):
    csv_file = sdir_path(csv_file)
  return h2o.import_file(csv_file)


def make_predictions_and_save(classifier, test_data, output_file, columns_offset):
  if path.exists(output_file) or sdir_exists(output_file):
    print 'already exists', output_file
    return
  if type(classifier) == str:
    classifier = load_h2o_model(classifier)
  if type(test_data) == str:
    test_data = load_h2o_data(test_data)
  predictions = classifier.predict(test_data[:,columns_offset:])
  if '/' not in output_file:
    output_file = sdir_path(output_file)
  h2o.download_csv(predictions, output_file)


def clear_h2o_memory():
  for x in h2o.ls():
    try:
      if x[0] == 'key':
        continue
      h2o.remove(x[0])
    except:
      continue

