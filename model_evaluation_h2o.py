#!/usr/bin/env python
# md5: dd33245d9893bd42b01276a1b0a5b1cf
# coding: utf-8

from tmilib import *
from h2o_utils import *

import h2o
h2o.init()


import traceback


#print len(sdir_glob('*mtries_*_sample_rate_*'))


#classifier = load_h2o_model(sdir_path('binclassifier_catfeatures_gradientboost_v3.h2o'))
#print classifier



for model_file in sdir_glob('*mtries_*_sample_rate_*'):
  print model_file
  try:
    classifier = load_h2o_model(model_file)
    print classifier
  except:
    traceback.print_exc()
    continue


model_file = sdir_path('binclassifier_catfeatures_randomforest_v6.h2o')
classifier = load_h2o_model(model_file)


print classifier


test_data = h2o.import_file(sdir_path('catdata_test_second_v2.csv'))


#test_data_2[0] = test_data_2[0].asfactor()
#print test_data_2.describe()


#test_data = h2o.import_file(sdir_path('catdata_test_second.csv'))


#print test_data.describe()
#test_data[0] = test_data[0].asfactor()
#test_data[0,:] = 1


#test_predictions = classifier.predict(test_data)


#print classifier


#print h2o.confusion_matrix(test_predictions, )
print classifier.model_performance(test_data)


#print classifier.confusion_matrix
#print test_data['label']
#print test_predictions
#print classifier.F1


#print test_data.describe()


testdata= h2o.import_file(sdir_path('catdata_test_insession_tensecond.csv'))


print testdata.describe()


#h2o.export_file(h2o.get_frame('h2odata_test_threefeatures_insession.hex'), sdir_path('h2odata_test_threefeatures_insession.hex'))


#print classifier.confusion_matrix(test_data)

