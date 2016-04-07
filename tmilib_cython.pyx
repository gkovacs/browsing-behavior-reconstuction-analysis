#!/usr/bin/env python
# md5: 4bad6dda6d99919c42d1df2ff956999f
# coding: utf-8

import numpy
cimport numpy
import tmilib


def dataset_to_feature_vectors(dataset, enabled_feat=None):
  cdef long[:] topdomains = numpy.array([tmilib.domain_to_id(x) for x in tmilib.top_n_domains_by_visits(20)], dtype=int)
  cdef long num_topdomains = len(topdomains)
  cdef long[:] domain_id_to_productivity = numpy.array(tmilib.get_domain_id_to_productivity(), dtype=numpy.int64)
  cdef long[:] rescuetime_productivity_levels = numpy.array(tmilib.get_rescuetime_productivity_levels(), dtype=int)
  cdef long num_rescuetime_productivity_levels = len(rescuetime_productivity_levels)
  cdef long num_features = 3 + 2*num_topdomains + 2*num_rescuetime_productivity_levels
  cdef long[:] enabled_features
  if type(enabled_feat) == str:
    enabled_features = numpy.array(map(int, enabled_feat), dtype=int)
  elif enabled_feat == None:
    enabled_features = numpy.array([1]*num_features, dtype=int)
  else:
    enabled_features = numpy.array(enabled_feat, dtype=int)
  #cdef list output = [[0]*num_features for x in range(len(dataset['sinceprev']))]
  cdef float[:,:] output = numpy.zeros((len(dataset['sinceprev']), num_features), dtype=float)
  #cdef list output = []
  #output = numpy.zeros((len(dataset['sinceprev']), num_features), dtype=object) # object instead of float, so we can have floats and ints
  #for idx,sinceprev,tonext,fromdomain,todomain in zipkeys_idx(dataset, 'sinceprev', 'tonext', 'fromdomain', 'todomain'):
  #cdef list cur
  cdef long feature_num, fromdomain_productivity, todomain_productivity
  cdef long label, fromdomain, todomain
  cdef float sinceprev, tonext
  cdef long productivity_idx, productivity, domain_idx, domain
  cdef long output_idx = 0
  cdef long cur_idx = 0
  for label,sinceprev,tonext,fromdomain,todomain in dataset:
    #cur = output[output_idx]
    output_idx += 1
    cur_idx = 0
    if enabled_features[0]:
      output[output_idx,cur_idx] = sinceprev
      cur_idx += 1
    if enabled_features[1]:
      output[output_idx,cur_idx] = tonext
      cur_idx += 1
    if enabled_features[2]:
      output[output_idx,cur_idx] = fromdomain == todomain
      cur_idx += 1
    feature_num = 3
    for domain_idx in range(num_topdomains):
      if enabled_features[feature_num+domain_idx]:
        output[output_idx,cur_idx] = fromdomain == topdomains[domain_idx]
        cur_idx += 1
    feature_num += num_topdomains
    for domain_idx in range(num_topdomains):
      if enabled_features[feature_num+domain_idx]:
        output[output_idx,cur_idx] = todomain == topdomains[domain_idx]
        cur_idx += 1
    feature_num += num_topdomains
    fromdomain_productivity = domain_id_to_productivity[fromdomain]
    todomain_productivity = domain_id_to_productivity[todomain]
    for productivity_idx in range(num_rescuetime_productivity_levels):
      if enabled_features[feature_num+productivity_idx]:
        output[output_idx,cur_idx] = fromdomain_productivity == rescuetime_productivity_levels[productivity_idx]
        cur_idx += 1
    feature_num += num_rescuetime_productivity_levels
    for productivity_idx in range(num_rescuetime_productivity_levels):
      if enabled_features[feature_num+productivity_idx]:
        output[output_idx,cur_idx] = todomain_productivity == rescuetime_productivity_levels[productivity_idx]
        cur_idx += 1
    #feature_num += len(get_rescuetime_productivity_levels())
  return output


