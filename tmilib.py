#!/usr/bin/env python
# md5: e481bd9c9e275428f21c51fc70a662cd
# coding: utf-8

from tmilib_base import *


def create_if_doesnt_exist(filename, function):
  if sdir_exists(filename):
    return
  data = function()
  sdir_dumpjson(filename, data)

def create_and_get(filename, function):
  create_if_doesnt_exist(filename, function)
  return sdir_loadjson(filename)


def compute_username_to_mturk_id():
  username_to_mturk_id = {}
  #for filename in (list_logfiles() + list_mlogfiles()):
  for filename in list_logfiles():
    print filename
    data = json.load(open(filename))
    last_item = data[len(data) - 1]
    user = last_item['user']
    if 'mturkid' not in last_item:
      continue
    mturkid = last_item['mturkid']
    username_to_mturk_id[user] = mturkid
  return username_to_mturk_id

def get_username_to_mturk_id():
  return create_and_get('username_to_mturk_id.json', compute_username_to_mturk_id)


@memoized
def compute_mturkid_to_history_pages_and_visits():
  username_to_mturk_id = get_username_to_mturk_id()
  mturkid_to_history_pages = {}
  mturkid_to_history_visits = {}
  mturkid_to_hid = {}
  for filename in list_histfiles():
    print filename
    all_lines = json.load(open(filename))
    for line in reversed(all_lines):
      evt = line['evt']
      hid = line['hid']
      user = line.get('user', None)
      mturkid = line.get('mturkid', None)
      if mturkid == None:
        if user == None:
          continue
        mturkid = username_to_mturk_id.get(user, None)
        if mturkid == None:
          continue
      orig_hid = mturkid_to_hid.get(mturkid, 0)
      if orig_hid > hid:
        continue
      if hid > orig_hid:
        mturkid_to_hid[mturkid] = hid
        mturkid_to_history_pages[mturkid] = []
        mturkid_to_history_visits[mturkid] = {}
      data = json.loads(decompress_lzstring.decompressFromBase64(line['data']))
      if evt == 'history_pages':
        mturkid_to_history_pages[mturkid] = data
      if evt == 'history_visits':
        for k,v in data.items():
          mturkid_to_history_visits[mturkid][k] = v
  return mturkid_to_history_pages,mturkid_to_history_visits

def compute_mturkid_to_history_pages():
  return compute_mturkid_to_history_pages_and_visits()[0]

def compute_mturkid_to_history_visits():
  return compute_mturkid_to_history_pages_and_visits()[1]

def get_mturkid_to_history_pages():
  return create_and_get('mturkid_to_history_pages.json', compute_mturkid_to_history_pages)

def get_mturkid_to_history_visits():
  return create_and_get('mturkid_to_history_visits.json', compute_mturkid_to_history_visits)


def compute_mturkid_to_time_last_active():
  mturkid_to_time_last_active = {}
  for logfile in list_logfiles():
    print logfile
    for data in iterate_data_compressed_reverse(logfile):
      if 'mturkid' not in data:
        break
      mturkid = data['mturkid']
      time = data['time']
      mturkid_to_time_last_active[mturkid] = time
      break
  return mturkid_to_time_last_active

def get_mturkid_to_time_last_active():
  return create_and_get('mturkid_to_time_last_active.json', compute_mturkid_to_time_last_active)


def compute_domains_list():
  mturkid_to_history_pages = get_mturkid_to_history_pages()
  alldomains = set()
  for k,v in mturkid_to_history_pages.items():
    for pageinfo in v:
      url = pageinfo['url']
      domain = url_to_domain(url)
      alldomains.add(domain)
  return list(alldomains)

def get_domains_list():
  return create_and_get('domains_list.json', compute_domains_list)

