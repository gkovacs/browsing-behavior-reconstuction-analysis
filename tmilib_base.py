#!/usr/bin/env python
# md5: 3e7f5be5401985527206006aca5815a4
# coding: utf-8

import urlparse
from glob import glob

import os
from os import path

import decompress_lzstring

from memoized import memoized

try:
  import ujson as json
except:
  import json

from collections import Counter
import numpy
import time
import datetime


@memoized
def get_basedir():
  return sorted([x for x in glob('/home/gkovacs/tmi-data/local_*') if path.isfile(x + '/active')], reverse=True)[0]
  #return '/home/gkovacs/tmi-data/latest'

@memoized
def list_logfiles():
  return glob(get_basedir() + '/logs_*.json')

@memoized
def list_mlogfiles():
  return glob(get_basedir() + '/mlogs_*.json')

@memoized
def list_histfiles():
  return glob(get_basedir() + '/hist_*.json')


@memoized
def get_sdir():
  return get_basedir().replace('local_', 'sdir_')

def ensure_sdir_exists():
  sdir = get_sdir()
  if path.exists(sdir):
    return
  os.makedirs(sdir)

def sdir_path(filename):
  return get_sdir() + '/' + filename

def sdir_exists(filename):
  return path.exists(sdir_path(filename))

def sdir_open(filename, mode='r'):
  return open(sdir_path(filename), mode)

def sdir_loadjson(filename):
  return json.load(sdir_open(filename))

def sdir_dumpjson(filename, data):
  ensure_sdir_exists()
  return json.dump(data, sdir_open(filename, 'w'))


def iterate_data(filename):
  for x in json.load(open(filename)):
    if 'windows' in x:
      x['windows'] = json.loads(decompress_lzstring.decompressFromBase64(x['windows']))
    if 'data' in x:
      x['data'] = json.loads(decompress_lzstring.decompressFromBase64(x['data']))
    yield x

def iterate_data_compressed(filename):
  for x in json.load(open(filename)):
    yield x

def iterate_data_reverse(filename):
  for x in reversed(json.load(open(filename))):
    if 'windows' in x:
      x['windows'] = json.loads(decompress_lzstring.decompressFromBase64(x['windows']))
    if 'data' in x:
      x['data'] = json.loads(decompress_lzstring.decompressFromBase64(x['data']))
    yield x

def iterate_data_compressed_reverse(filename):
  for x in reversed(json.load(open(filename))):
    yield x


def url_to_domain(url):
  return urlparse.urlparse(url).netloc

