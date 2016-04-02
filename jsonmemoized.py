#!/usr/bin/env python
# md5: db8f0caefdb8999a9a044a59421b9b55
# coding: utf-8

try:
  import ujson as json
except:
  import json
from os import path

# based on http://www.artima.com/weblogs/viewpost.jsp?thread=240845

class jsonmemoized(object):
  def __init__(self, f):
    self.f = f
    self.filename = f.__name__ + '.json'
  def __call__(self):
    if path.exists(self.filename):
      return json.load(open(self.filename))
    result = self.f()
    json.dump(result, open(self.filename, 'w'))
    return result

class jsonmemoized_fileloc(object):
  def __init__(self, filename=None):
    if filename != None:
      if not filename.endswith('.json'):
        filename = filename + '.json'
    self.filename = filename
  def __call__(self, f):
    if self.filename == None:
      self.filename = f.__name__ + '.json'
    def wrapped_f():
      if path.exists(self.filename):
        return json.load(open(self.filename))
      result = f()
      json.dump(result, open(self.filename, 'w'))
      return result
    return wrapped_f


#@jsonmemoized
#def compute_array():
#  print 'compute_array called'
#  return [3,5,7,9]


#print compute_array()

