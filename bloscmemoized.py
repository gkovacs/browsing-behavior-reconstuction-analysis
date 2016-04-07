#!/usr/bin/env python
# md5: cc1fcc77e6e57678e4542788465ead40
# coding: utf-8

import bloscpack
from os import path

# based on http://www.artima.com/weblogs/viewpost.jsp?thread=240845

class bloscmemoized(object):
  def __init__(self, f):
    self.f = f
    self.filename = f.__name__ + '.blosc'
  def __call__(self):
    if path.exists(self.filename):
      return bloscpack.unpack_ndarray_file(self.filename)
    result = self.f()
    bloscpack.pack_ndarray_file(result, self.filename)
    return result

class bloscmemoized_fileloc(object):
  def __init__(self, filename=None):
    if filename != None:
      if not filename.endswith('.blosc'):
        filename = filename + '.blosc'
    self.filename = filename
  def __call__(self, f):
    if self.filename == None:
      self.filename = f.__name__ + '.blosc'
    def wrapped_f():
      if path.exists(self.filename):
        return bloscpack.unpack_ndarray_file(self.filename)
      result = f()
      bloscpack.pack_ndarray_file(result, self.filename)
      return result
    return wrapped_f


@bloscmemoized
def compute_array():
  print 'compute_array called'
  return [3,5,7,9]


print compute_array()

