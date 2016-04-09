#!/usr/bin/env python
# md5: d4fb0249d559bd4333028fd5bbb1e14e
# coding: utf-8

from tmilib import *
from h2o_utils import *

import h2o
h2o.init()


model_file = sdir_path('classifier_threefeatures_randomforest_insession.h2o')
classifier = load_h2o_model(model_file)


test_data = h2o.import_file(sdir_path('h2odata_test_threefeatures_insession.csv'))


test_predictions = classifier.predict(test_data)


#print classifier


#print h2o.confusion_matrix(test_predictions, )
print classifier.model_performance(test_data)


#print classifier.confusion_matrix
#print test_data['label']
#print test_predictions
print classifier.F1


print test_data.describe()


testdata= h2o.import_file(sdir_path('catdata_test_insession_tensecond.csv'))


print testdata.describe()


#h2o.export_file(h2o.get_frame('h2odata_test_threefeatures_insession.hex'), sdir_path('h2odata_test_threefeatures_insession.hex'))


#print classifier.confusion_matrix(test_data)

