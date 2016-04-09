#!/usr/bin/env python
# md5: 4f661977a19858a897fcf8b6f8b8697f
# coding: utf-8

import h2o
from glob import glob


def load_h2o_model(model_path):
  return h2o.load_model(glob(model_path + '/*')[0])

