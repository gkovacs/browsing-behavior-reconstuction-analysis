#!/usr/bin/env python
# md5: de3c5b09077337e3de75c2b24f8bd391
# coding: utf-8

from tmilib_base import *
from session_tracker import SessionTracker





function_name_to_function_mapping = {
  # single json files
  
  #'username_to_mturk_id': compute_username_to_mturk_id,
  #'mturkid_to_history_pages': compute_mturkid_to_history_pages,
  #'mturkid_to_history_visits': compute_mturkid_to_history_visits,
  #'mturkid_to_time_last_active': compute_mturkid_to_time_last_active,
  #'domains_list': compute_domains_list,
  
  # multiuser directories
  
  #'tab_focus_times_for_user': compute_tab_focus_times_for_user,
  #'history_pages_for_user': compute_history_pages_for_user,
  #'history_visits_for_user': compute_history_visits_for_user,
  #'history_ordered_visits_for_user': compute_history_ordered_visits_for_user,
}

def get_compute_function_from_name(name):
  if name.endswith('.json'):
    name = name[:-5] # removes .json
  if name in function_name_to_function_mapping:
    return function_name_to_function_mapping[name]
  compute_function = globals().get('compute_' + name, None)
  if compute_function != None:
    function_name_to_function_mapping[name] = compute_function
    return compute_function
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





'''
def compute_tab_focus_times_for_user(user):
  logfile = get_logfile_for_user(user)
  current_session_tracker = SessionTracker()
  for line in iterate_data(logfile):
    current_session_tracker.process_input(line)
  current_session_tracker.end_input()
  return current_session_tracker.get_output()
'''

def compute_tab_focus_times_for_user(user):
  logfile = get_logfile_for_user(user)
  current_session_tracker = SessionTracker()
  for line in iterate_data_timesorted(logfile):
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


def compute_history_pages_for_user(user):
  filename = get_histfile_for_user(user)
  all_lines = json.load(open(filename))
  max_hid = 0
  for line in all_lines:
    hid = line['hid']
    max_hid = max(hid, max_hid)
  for line in all_lines:
    hid = line['hid']
    if hid < max_hid:
      continue
    evt = line['evt']
    if evt == 'history_pages':
      data = decompress_data_lzstring_base64(line['data'])
      return data

def compute_history_visits_for_user(user):
  filename = get_histfile_for_user(user)
  all_lines = json.load(open(filename))
  max_hid = 0
  for line in all_lines:
    hid = line['hid']
    max_hid = max(hid, max_hid)
  output = {}
  for line in all_lines:
    hid = line['hid']
    if hid < max_hid:
      continue
    evt = line['evt']
    if evt == 'history_visits':
      data = decompress_data_lzstring_base64(line['data'])
      for k,v in data.items():
        output[k] = v
  return output

def get_history_pages_for_user(user):
  return get_function_for_key(user, 'history_pages_for_user')

def get_history_visits_for_user(user):
  return get_function_for_key(user, 'history_visits_for_user')

def compute_history_pages_for_all_users():
  for user in list_users_with_hist():
    print user
    compute_function_for_key(user, 'history_pages_for_user')

def compute_history_pages_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    print user
    compute_function_for_key(user, 'history_pages_for_user')

def compute_history_visits_for_all_users():
  for user in list_users_with_hist():
    print user
    compute_function_for_key(user, 'history_visits_for_user')

def compute_history_visits_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    print user
    compute_function_for_key(user, 'history_visits_for_user')

#compute_tab_focus_times_for_all_users()


def compute_history_ordered_visits_for_user(user):
  url_to_visits = get_history_visits_for_user(user)
  ordered_visits = []
  for url,visits in url_to_visits.items():
    for visit in visits:
      visit['url'] = url
    ordered_visits.extend(visits)
  ordered_visits.sort(key=itemgetter('visitTime'))
  return ordered_visits

def get_history_ordered_visits_for_user(user):
  return get_function_for_key(user, 'history_ordered_visits_for_user')

def compute_history_ordered_visits_for_all_users():
  for user in list_users_with_hist():
    print user
    compute_function_for_key(user, 'history_ordered_visits_for_user')

def compute_history_ordered_visits_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    print user
    compute_function_for_key(user, 'history_ordered_visits_for_user')


def compute_mlog_active_times_for_user(user):
  mlogfile = get_mlogfile_for_user(user)
  output = []
  for line in iterate_data_timesorted(mlogfile):
    if 'data' in line:
      curitem = {'url': line['data']['location'], 'time': line['time']}
    else:
      curitem = {'url': line['location'], 'time': line['time']}
    output.append(curitem)
  return output

def get_mlog_active_times_for_user(user):
  return get_function_for_key(user, 'mlog_active_times_for_user')

def compute_mlog_active_times_for_all_users():
  for user in list_users_with_mlog():
    print user
    compute_function_for_key(user, 'mlog_active_times_for_user')

def compute_mlog_active_times_for_all_users_randomized():
  for user in shuffled(list_users_with_mlog()):
    print user
    compute_function_for_key(user, 'mlog_active_times_for_user')

#print compute_mlog_active_times_for_user('ZDMgTG3hUx')
#compute_function_for_key('ZDMgTG3hUx', 'mlog_active_times_for_user')

