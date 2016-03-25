#!/usr/bin/env python
# md5: 0d3fa0c6c9d2b1cde9fcaf4593fd3f17
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
import random
from operator import itemgetter


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
def list_users():
  return [filename_to_username(x) for x in list_logfiles()]

@memoized
def list_users_with_hist():
  return [filename_to_username(x) for x in list_histfiles()]

@memoized
def list_users_with_mlog():
  return [filename_to_username(x) for x in list_mlogfiles()]


@memoized
def get_sdir():
  return get_basedir().replace('local_', 'sdir_')

def ensure_sdir_exists():
  sdir = get_sdir()
  if path.exists(sdir):
    return
  os.makedirs(sdir)

def ensure_sdir_subdir_exists(subdir):
  ensure_sdir_exists()
  sdir = get_sdir()
  if path.exists(sdir + '/' + subdir):
    return
  os.makedirs(sdir + '/' + subdir)

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


def dumpdir_path(filename):
  return get_basedir() + '/' + filename

def get_logfile_for_user(user):
  return dumpdir_path('logs_' + user + '.json')

def get_mlogfile_for_user(user):
  return dumpdir_path('mlogs_' + user + '.json')

def get_histfile_for_user(user):
  return dumpdir_path('hist_' + user + '.json')


def filename_to_username(filename):
  if not filename.endswith('.json'):
    raise Exception('expected filename to end with .json ' + filename)
  filename = filename[:-5] # removes the .json
  return filename.split('_')[-1] # returns part after the last _ which is the username


def decompress_data_lzstring_base64(data):
  data_type = type(data)
  if data_type == unicode or data_type == str:
    return json.loads(decompress_lzstring.decompressFromBase64(data))
  return data

def iterate_data_jsondata(data):
  for x in data:
    if 'windows' in x:
      data_type = type(x['windows'])
      if data_type == unicode or data_type == str:
        x['windows'] = json.loads(decompress_lzstring.decompressFromBase64(x['windows']))
    if 'data' in x:
      data_type = type(x['data'])
      if data_type == unicode or data_type == str:
        x['data'] = json.loads(decompress_lzstring.decompressFromBase64(x['data']))
    yield x

def iterate_data(filename):
  for x in json.load(open(filename)):
    if 'windows' in x:
      data_type = type(x['windows'])
      if data_type == unicode or data_type == str:
        x['windows'] = json.loads(decompress_lzstring.decompressFromBase64(x['windows']))
    if 'data' in x:
      data_type = type(x['data'])
      if data_type == unicode or data_type == str:
        x['data'] = json.loads(decompress_lzstring.decompressFromBase64(x['data']))
    yield x

def iterate_data_timesorted(filename):
  return sorted(iterate_data(filename), key=itemgetter('time'))

def iterate_data_compressed(filename):
  for x in json.load(open(filename)):
    yield x

def iterate_data_jsondata_reverse(data):
  for x in reversed(data):
    if 'windows' in x:
      data_type = type(x['windows'])
      if data_type == unicode or data_type == str:
        x['windows'] = json.loads(decompress_lzstring.decompressFromBase64(x['windows']))
    if 'data' in x:
      data_type = type(x['data'])
      if data_type == unicode or data_type == str:
        x['data'] = json.loads(decompress_lzstring.decompressFromBase64(x['data']))
    yield x

def iterate_data_reverse(filename):
  for x in reversed(json.load(open(filename))):
    if 'windows' in x:
      data_type = type(x['windows'])
      if data_type == unicode or data_type == str:
        x['windows'] = json.loads(decompress_lzstring.decompressFromBase64(x['windows']))
    if 'data' in x:
      data_type = type(x['data'])
      if data_type == unicode or data_type == str:
        x['data'] = json.loads(decompress_lzstring.decompressFromBase64(x['data']))
    yield x

def iterate_data_compressed_reverse(filename):
  for x in reversed(json.load(open(filename))):
    yield x


def url_to_domain(url):
  return urlparse.urlparse(url).netloc


def shuffled(l):
  l = l[:]
  random.shuffle(l)
  return l

