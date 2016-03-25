#!/usr/bin/env python
# md5: 5e66effacf4bcdedaddf76c7eacf5cb1
# coding: utf-8

from tmilib_base import *
from session_tracker import SessionTracker


def get_compute_function_from_name(name):
  if name.endswith('.json'):
    name = name[:-5] # removes .json
  mapping = {
    # single json files
    'username_to_mturk_id': compute_username_to_mturk_id,
    'mturkid_to_history_pages': compute_mturkid_to_history_pages,
    'mturkid_to_history_visits': compute_mturkid_to_history_visits,
    'mturkid_to_time_last_active': compute_mturkid_to_time_last_active,
    'domains_list': compute_domains_list,
    # multiuser directories
    'tab_focus_times_for_user': compute_tab_focus_times_for_user,
  }
  if name in mapping:
    return mapping[name]
  raise Exception('get_function_for_name failed for ' + name)


def create_if_doesnt_exist(filename, function=None):
  if function == None:
    function = get_compute_function_from_name(filename)
  if sdir_exists(filename):
    return
  data = function()
  sdir_dumpjson(filename, data)

def create_and_get(filename, function=None):
  create_if_doesnt_exist(filename, function)
  return sdir_loadjson(filename)


def compute_function_for_key(key, name, function=None):
  if function == None:
    function = get_compute_function_from_name(name)
  outfile = name + '/' + key + '.json'
  if sdir_exists(outfile):
    return
  result = function(key)
  ensure_sdir_subdir_exists(name)
  sdir_dumpjson(outfile, result)

def get_function_for_key(key, name, function=None):
  compute_function_for_key(key, name, function)
  outfile = name + '/' + key + '.json'
  return sdir_loadjson(outfile)





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
  #return create_and_get('username_to_mturk_id.json', compute_username_to_mturk_id)
  return create_and_get('username_to_mturk_id.json')


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
  #return create_and_get('mturkid_to_history_pages.json', compute_mturkid_to_history_pages)
  return create_and_get('mturkid_to_history_pages.json')

def get_mturkid_to_history_visits():
  #return create_and_get('mturkid_to_history_visits.json', compute_mturkid_to_history_visits)
  return create_and_get('mturkid_to_history_visits.json')


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
  #return create_and_get('mturkid_to_time_last_active.json', compute_mturkid_to_time_last_active)
  return create_and_get('mturkid_to_time_last_active.json')


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
  #return create_and_get('domains_list.json', compute_domains_list)
  return create_and_get('domains_list.json')





def compute_tab_focus_times_for_user(user):
  logfile = get_logfile_for_user(user)
  current_session_tracker = SessionTracker()
  for line in iterate_data(logfile):
    current_session_tracker.process_input(line)
  current_session_tracker.end_input()
  return current_session_tracker.get_output()

def get_tab_focus_times_for_user(user):
  return get_function_for_key(user, 'tab_focus_times_for_user')

def compute_tab_focus_times_for_all_users():
  for user in list_users():
    #filesize = path.getsize(filename)
    #filesize_megabytes = filesize / (1000.0*1000.0)
    #if filesize_megabytes > 0.1:
    #  continue
    print user
    compute_function_for_key(user, 'tab_focus_times_for_user')

def compute_tab_focus_times_for_all_users_randomized():
  for user in shuffled(list_users()):
    print user
    compute_function_for_key(user, 'tab_focus_times_for_user')

#compute_tab_focus_times_for_all_users()

