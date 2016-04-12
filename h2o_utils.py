#!/usr/bin/env python
# md5: 17df6e7b144d4dc2106c6c5610115b59
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

