#!/usr/bin/env python
# md5: 8908e6ab16a62338ba1f227494688285
# coding: utf-8

from tmilib_base import *
from session_tracker import SessionTracker, get_focused_tab
from reconstruct_focus_times import ReconstructFocusTimesBaseline
from reconstruct_focus_times_base import *
from jsonmemoized import *
from msgpackmemoized import *
#from bloscmemoized import *

from rescuetime_utils import *

from tmilib_cython import dataset_to_feature_vectors





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
  if sdir_exists(filename):
    return
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
  print outfile
  ensure_sdir_subdir_exists(name)
  result = function(key)
  if sdir_exists(outfile):
    return
  sdir_dumpjson(outfile, result)

def get_function_for_key(key, name, function=None):
  compute_function_for_key(key, name, function)
  outfile = name + '/' + key + '.json'
  return sdir_loadjson(outfile)



def compute_function_for_key_lines(key, name, function=None):
  if function == None:
    function = get_compute_function_from_name(name)
  outfile = name + '/' + key + '.jsonlines'
  if sdir_exists(outfile):
    return
  print outfile
  ensure_sdir_subdir_exists(name)
  result = function(key)
  if sdir_exists(outfile):
    return
  sdir_dumpjsonlines(outfile, result)  

def get_function_for_key_lines(key, name, function=None):
  compute_function_for_key_lines(key, name, function)
  outfile = name + '/' + key + '.jsonlines'
  return sdir_loadjsonlines(outfile)





'''
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
'''

def compute_username_to_mturk_id():
  output = {}
  for user in list_users_with_log():
    for line in iterate_logs_for_user_compressed(user):
      if 'mturkid' in line:
        mturkid = line['mturkid']
        output[user] = mturkid
        break
  return output

def get_username_to_mturk_id():
  #return create_and_get('username_to_mturk_id.json', compute_username_to_mturk_id)
  return create_and_get('username_to_mturk_id.json')

def precompute_username_to_mturk_id():
  create_if_doesnt_exist('username_to_mturk_id.json')


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
      data = json.loads(decompressFromBase64(line['data']))
      if evt == 'history_pages':
        mturkid_to_history_pages[mturkid] = data
      if evt == 'history_visits':
        for k,v in data.viewitems():
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


'''
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
'''

def compute_mturkid_to_time_last_active():
  mturkid_to_time_last_active = {}
  for user in list_users_with_log():
    for data in iterate_logs_for_user_compressed(user):
      if 'mturkid' not in data:
        continue
      mturkid = data['mturkid']
      time = data['time']
      if mturkid not in mturkid_to_time_last_active:
        mturkid_to_time_last_active[mturkid] = time
      else:
        mturkid_to_time_last_active[mturkid] = max(time, mturkid_to_time_last_active[mturkid])
  return mturkid_to_time_last_active

def get_mturkid_to_time_last_active():
  #return create_and_get('mturkid_to_time_last_active.json', compute_mturkid_to_time_last_active)
  return create_and_get('mturkid_to_time_last_active.json')

def precompute_mturkid_to_time_last_active():
  create_if_doesnt_exist('mturkid_to_time_last_active.json')


'''
def compute_domains_list():
  mturkid_to_history_pages = get_mturkid_to_history_pages()
  alldomains = set()
  for k,v in mturkid_to_history_pages.items():
    for pageinfo in v:
      url = pageinfo['url']
      domain = url_to_domain(url)
      alldomains.add(domain)
  return list(alldomains)
'''

def compute_domains_list():
  alldomains = set()
  for user in list_users_with_hist():
    for pageinfo in get_history_pages_for_user(user):
      url = pageinfo['url']
      domain = url_to_domain(url)
      alldomains.add(domain)
    # this part is technically redundant but seems things slip through?
    for url in get_history_visits_for_user(user).viewkeys(): 
      domain = url_to_domain(url)
      alldomains.add(domain)
  for user in list_users_with_log_and_mlog():
    for visit in get_tab_focus_times_for_user(user):
      domain = url_to_domain(visit['url'])
      alldomains.add(domain)
  return list(alldomains)

def get_domains_list():
  #return create_and_get('domains_list.json', compute_domains_list)
  return create_and_get('domains_list.json')

@memoized
def get_domains_list_memoized():
  return create_and_get('domains_list.json')

def precompute_domains_list():
  create_if_doesnt_exist('domains_list.json')





def compute_domains_to_id():
  output = {}
  for idx,domain in enumerate(get_domains_list()):
    output[domain] = idx
  return output

def get_domains_to_id():
  return create_and_get('domains_to_id.json')

@memoized
def get_domains_to_id_memoized():
  return create_and_get('domains_to_id.json')

def precompute_domains_to_id():
  create_if_doesnt_exist('domains_to_id.json')


def id_to_domain(domain_id):
  return get_domains_list_memoized()[domain_id]

def domain_to_id(domain):
  return get_domains_to_id_memoized()[domain]


'''
def compute_tab_focus_times_for_user(user):
  logfile = get_logfile_for_user(user)
  current_session_tracker = SessionTracker()
  for line in iterate_data(logfile):
    current_session_tracker.process_input(line)
  current_session_tracker.end_input()
  return current_session_tracker.get_output()
'''

'''
def compute_tab_focus_times_for_user(user):
  logfile = get_logfile_for_user(user)
  current_session_tracker = SessionTracker()
  for line in iterate_data_timesorted(logfile):
    current_session_tracker.process_input(line)
  current_session_tracker.end_input()
  return current_session_tracker.get_output()
'''

'''
def compute_tab_focus_times_for_user(user):
  current_session_tracker = SessionTracker()
  for line in iterate_logs_for_user(user):
    current_session_tracker.process_input(line)
  current_session_tracker.end_input()
  return current_session_tracker.get_output()
'''

def compute_tab_focus_times_for_user(user):
  current_session_tracker = SessionTracker()
  for line in get_log_with_mlog_active_times_for_user(user):
    current_session_tracker.process_input(uncompress_data_subfields(line))
  current_session_tracker.end_input()
  return current_session_tracker.get_output()

def get_tab_focus_times_for_user(user):
  return get_function_for_key(user, 'tab_focus_times_for_user')

def compute_tab_focus_times_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key(user, 'tab_focus_times_for_user')

def compute_tab_focus_times_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'tab_focus_times_for_user')


def compute_active_seconds_for_user(user):
  # this is in unix SECONDS timestamp, not in milliseconds!
  output = []
  last_output = None
  for item in get_tab_focus_times_for_user(user):
    start_seconds = int(round(item['start']/1000.0))
    end_seconds = int(round(item['end']/1000.0))
    for timestep in xrange(start_seconds, end_seconds+1):
      if timestep > last_output:
        last_output = timestep
        output.append(timestep)
  return output

def get_active_seconds_for_user(user):
  # will probably want to convert this into a set after returning
  # this is in unix SECONDS timestamp, not in milliseconds!
  return get_function_for_key(user, 'active_seconds_for_user')

def compute_active_seconds_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key(user, 'active_seconds_for_user')

def compute_active_seconds_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'active_seconds_for_user')


def compute_tab_focus_times_only_tab_updated_for_user(user):
  current_session_tracker = SessionTracker()
  for line in get_log_with_mlog_active_times_for_user(user):
    if line['evt'] != 'tab_updated':
      continue
    current_session_tracker.process_input(uncompress_data_subfields(line))
  current_session_tracker.end_input()
  return current_session_tracker.get_output()

def get_tab_focus_times_only_tab_updated_for_user(user):
  return get_function_for_key(user, 'tab_focus_times_only_tab_updated_for_user')

def compute_tab_focus_times_only_tab_updated_for_all_users():
  for user in list_users_with_log_and_mlog():
    #filesize = path.getsize(filename)
    #filesize_megabytes = filesize / (1000.0*1000.0)
    #if filesize_megabytes > 0.1:
    #  continue
    compute_function_for_key(user, 'tab_focus_times_only_tab_updated_for_user')

def compute_tab_focus_times_only_tab_updated_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'tab_focus_times_only_tab_updated_for_user')


def compute_tab_focus_times_only_tab_updated_urlchanged_for_user(user):
  current_session_tracker = SessionTracker()
  prev_url = None
  for line in get_log_with_mlog_active_times_for_user(user):
    if line['evt'] != 'tab_updated':
      continue
    url = line['tab']['url']
    if url == prev_url:
      continue
    prev_url = url
    current_session_tracker.process_input(uncompress_data_subfields(line))
  current_session_tracker.end_input()
  return current_session_tracker.get_output()

def get_tab_focus_times_only_tab_updated_urlchanged_for_user(user):
  return get_function_for_key(user, 'tab_focus_times_only_tab_updated_urlchanged_for_user')

def compute_tab_focus_times_only_tab_updated_urlchanged_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key(user, 'tab_focus_times_only_tab_updated_urlchanged_for_user')

def compute_tab_focus_times_only_tab_updated_urlchanged_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'tab_focus_times_only_tab_updated_urlchanged_for_user')


def compute_idealized_history_from_logs_for_user(user):
  output = []
  for line in get_log_with_mlog_active_times_for_user(user):
    if line['evt'] != 'tab_updated':
      continue
    url = line['tab']['url']
    time = line['time']
    output.append({'url': url, 'visitTime': time, 'transition': 'link'})
  return output

def get_idealized_history_from_logs_for_user(user):
  return get_function_for_key(user, 'idealized_history_from_logs_for_user')

def compute_idealized_history_from_logs_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key(user, 'idealized_history_from_logs_for_user')

def compute_idealized_history_from_logs_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'idealized_history_from_logs_for_user')


def compute_idealized_history_from_logs_urlchanged_for_user(user):
  output = []
  prev_url = None
  for line in get_log_with_mlog_active_times_for_user(user):
    if line['evt'] != 'tab_updated':
      continue
    url = line['tab']['url']
    if url != prev_url:
      prev_url = url
      time = line['time']
      output.append({'url': url, 'visitTime': time, 'transition': 'link'})
  return output

def get_idealized_history_from_logs_urlchanged_for_user(user):
  return get_function_for_key(user, 'idealized_history_from_logs_urlchanged_for_user')

def compute_idealized_history_from_logs_urlchanged_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key(user, 'idealized_history_from_logs_urlchanged_for_user')

def compute_idealized_history_from_logs_urlchanged_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'idealized_history_from_logs_urlchanged_for_user')


def compute_url_to_tab_focus_times_for_user(user):
  tab_focus_times = get_tab_focus_times_for_user(user)
  output = {}
  for visit in tab_focus_times:
    url = visit['url']
    if url not in output:
      output[url] = []
    output[url].append(visit)
  return output

def get_url_to_tab_focus_times_for_user(user):
  return get_function_for_key(user, 'url_to_tab_focus_times_for_user')

def compute_url_to_tab_focus_times_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key(user, 'url_to_tab_focus_times_for_user')

def compute_url_to_tab_focus_times_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'url_to_tab_focus_times_for_user')


def compute_domain_to_tab_focus_times_for_user(user):
  tab_focus_times = get_tab_focus_times_for_user(user)
  output = {}
  for visit in tab_focus_times:
    url = visit['url']
    domain = url_to_domain(url)
    if domain not in output:
      output[domain] = []
    output[domain].append(visit)
  return output

def get_domain_to_tab_focus_times_for_user(user):
  return get_function_for_key(user, 'domain_to_tab_focus_times_for_user')

def compute_domain_to_tab_focus_times_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key(user, 'domain_to_tab_focus_times_for_user')

def compute_domain_to_tab_focus_times_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'domain_to_tab_focus_times_for_user')


def compute_domain_to_time_spent_for_user(user):
  output = {}
  for domain,tab_focus_times in get_domain_to_tab_focus_times_for_user(user).viewitems():
    time_spent = 0
    for item in tab_focus_times:
      start = item['start']
      end = item['end']
      if start >= end:
        continue
      time_spent += end - start
    output[domain] = time_spent
  return output

def get_domain_to_time_spent_for_user(user):
  return get_function_for_key(user, 'domain_to_time_spent_for_user')

def compute_domain_to_time_spent_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key(user, 'domain_to_time_spent_for_user')

def compute_domain_to_time_spent_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key(user, 'domain_to_time_spent_for_user')


def compute_domain_to_num_history_visits_for_user(user):
  output = Counter()
  for url,visits in get_history_visits_for_user(user).viewitems():
    domain = url_to_domain(url)
    output[domain] += len(visits)
  return output

def get_domain_to_num_history_visits_for_user(user):
  return get_function_for_key(user, 'domain_to_num_history_visits_for_user')

def compute_domain_to_num_history_visits_for_all_users():
  for user in list_users_with_hist():
    compute_function_for_key(user, 'domain_to_num_history_visits_for_user')

def compute_domain_to_num_history_visits_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'domain_to_num_history_visits_for_user')





'''
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
'''

'''
def compute_history_pages_for_user(user):
  max_hid = 0
  for line in iterate_hist_for_user_compressed(user):
    hid = line['hid']
    max_hid = max(hid, max_hid)
  for line in iterate_hist_for_user_compressed(user):
    hid = line['hid']
    if hid < max_hid:
      continue
    evt = line['evt']
    if evt == 'history_pages':
      data = decompress_data_lzstring_base64(line['data'])
      return data
  return []

def compute_history_visits_for_user(user):
  max_hid = 0
  for line in iterate_hist_for_user_compressed(user):
    hid = line['hid']
    max_hid = max(hid, max_hid)
  output = {}
  for line in iterate_hist_for_user_compressed(user):
    hid = line['hid']
    if hid < max_hid:
      continue
    evt = line['evt']
    if evt == 'history_visits':
      data = decompress_data_lzstring_base64(line['data'])
      for k,v in data.items():
        output[k] = v
  return output
'''

def compute_history_valid_hids_for_user(user):
  hid_with_history_pages = set()
  hid_to_totalparts = {}
  hid_to_seenparts = {}
  hid_with_complete_history_visits = set()
  for line in iterate_hist_for_user_compressed(user):
    hid = line['hid']
    evt = line['evt']
    if evt == 'history_pages':
      hid_with_history_pages.add(hid)
      continue
    if evt == 'history_visits':
      totalparts = line['totalparts']
      idx = line['idx']
      if totalparts < 1:
        raise 'have totalparts value less than one of ' + str(totalparts) + ' for user ' + user
      if hid not in hid_to_totalparts:
        hid_to_totalparts[hid] = totalparts
      else:
        if hid_to_totalparts[hid] != totalparts:
          raise 'inconsistent totalparts for user ' + user + ' on hid ' + str(hid) + ' with values ' + str(totalparts) + ' and ' + str(hid_to_totalparts[hid])
      if hid not in hid_to_seenparts:
        hid_to_seenparts[hid] = set()
      hid_to_seenparts[hid].add(idx)
      num_parts_seen_so_far = len(hid_to_seenparts[hid])
      if num_parts_seen_so_far > totalparts:
        raise 'num parts seen so far ' + str(num_parts_seen_so_far) + ' is greater than totalparts ' + str(totalparts) + ' for user ' + user
      if num_parts_seen_so_far == totalparts:
        hid_with_complete_history_visits.add(hid)        
  output = [hid for hid in hid_with_complete_history_visits if hid in hid_with_history_pages]
  output.sort()
  return output

def get_history_valid_hids(user):
  return get_function_for_key(user, 'history_valid_hids_for_user')

def compute_history_valid_hids_for_all_users():
  for user in list_users_with_hist():
    compute_function_for_key(user, 'history_valid_hids_for_user')

def compute_history_valid_hids_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'history_valid_hids_for_user')

def compute_history_pages_for_user(user):
  valid_hids = get_history_valid_hids(user)
  if len(valid_hids) == 0:
    return []
  target_hid = max(valid_hids)
  for line in iterate_hist_for_user_compressed(user):
    hid = line['hid']
    if hid != target_hid:
      continue
    evt = line['evt']
    if evt == 'history_pages':
      data = decompress_data_lzstring_base64(line['data'])
      return data
  return []

def compute_history_visits_for_user(user):
  valid_hids = get_history_valid_hids(user)
  if len(valid_hids) == 0:
    return {}
  target_hid = max(valid_hids)
  output = {}
  for line in iterate_hist_for_user_compressed(user):
    hid = line['hid']
    if hid < target_hid:
      continue
    evt = line['evt']
    if evt == 'history_visits':
      data = decompress_data_lzstring_base64(line['data'])
      for k,v in data.viewitems():
        output[k] = v
  return output


def get_history_pages_for_user(user):
  return get_function_for_key(user, 'history_pages_for_user')

def get_history_visits_for_user(user):
  return get_function_for_key(user, 'history_visits_for_user')

def compute_history_pages_for_all_users():
  for user in list_users_with_hist():
    compute_function_for_key(user, 'history_pages_for_user')

def compute_history_pages_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'history_pages_for_user')

def compute_history_visits_for_all_users():
  for user in list_users_with_hist():
    compute_function_for_key(user, 'history_visits_for_user')

def compute_history_visits_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'history_visits_for_user')

#compute_tab_focus_times_for_all_users()


def compute_history_ordered_visits_for_user(user):
  url_to_visits = get_history_visits_for_user(user)
  ordered_visits = []
  for url,visits in url_to_visits.viewitems():
    for visit in visits:
      visit['url'] = url
    ordered_visits.extend(visits)
  ordered_visits.sort(key=itemgetter('visitTime'))
  return ordered_visits

def get_history_ordered_visits_for_user(user):
  return get_function_for_key(user, 'history_ordered_visits_for_user')

def compute_history_ordered_visits_for_all_users():
  for user in list_users_with_hist():
    compute_function_for_key(user, 'history_ordered_visits_for_user')

def compute_history_ordered_visits_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'history_ordered_visits_for_user')


'''
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
'''

def compute_mlog_active_times_for_user(user):
  output = []
  for line in iterate_mlogs_for_user(user):
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
    compute_function_for_key(user, 'mlog_active_times_for_user')

def compute_mlog_active_times_for_all_users_randomized():
  for user in shuffled(list_users_with_mlog()):
    compute_function_for_key(user, 'mlog_active_times_for_user')

#print compute_mlog_active_times_for_user('ZDMgTG3hUx')
#compute_function_for_key('ZDMgTG3hUx', 'mlog_active_times_for_user')


def compute_history_visit_times_for_user(user):
  output = set()
  for visit in get_history_ordered_visits_for_user(user):
    output.add(visit['visitTime'])
  return sorted(list(output))

def get_history_visit_times_for_user(user):
  return get_function_for_key(user, 'history_visit_times_for_user')

def compute_history_visit_times_for_all_users():
  for user in list_users_with_hist():
    compute_function_for_key(user, 'history_visit_times_for_user')

def compute_history_visit_times_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'history_visit_times_for_user')


def compute_windows_at_time_for_user(user):
  output = {}
  history_visit_times = get_history_visit_times_for_user(user)
  hidx = 0
  for line in iterate_logs_for_user(user):
    curtime = line['time']
    windows = line['windows']
    while hidx < len(history_visit_times): # still have visit times that need to be labeled
      next_history_visit_time = history_visit_times[hidx]
      if curtime < next_history_visit_time:
        break
      else: # next_history_visit_time <= curtime
        output[curtime] = windows
        hidx += 1
  return output

def get_windows_at_time_for_user(user):
  return get_function_for_key(user, 'windows_at_time_for_user')

def compute_windows_at_time_for_all_users():
  for user in list_users_with_log_and_hist():
    compute_function_for_key(user, 'windows_at_time_for_user')

def compute_windows_at_time_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'windows_at_time_for_user')



def compute_allurls_at_time_for_user(user):
  output = {}
  for time,windows in get_windows_at_time_for_user(user).viewitems():
    output[time] = []
    for window in windows:
      for tab in window['tabs']:
        url = tab['url']
        output[time].append(url)
  return output

def get_allurls_at_time_for_user(user):
  return get_function_for_key(user, 'allurls_at_time_for_user')

def compute_allurls_at_time_for_all_users():
  for user in list_users_with_log_and_hist():
    compute_function_for_key(user, 'allurls_at_time_for_user')

def compute_allurls_at_time_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'allurls_at_time_for_user')

  


def compute_alldomains_at_time_for_user(user):
  output = {}
  for time,urls in get_allurls_at_time_for_user(user).viewitems():
    output[time] = [url_to_domain(url) for url in urls]
  return output

def get_alldomains_at_time_for_user(user):
  return get_function_for_key(user, 'alldomains_at_time_for_user')

def compute_alldomains_at_time_for_all_users():
  for user in list_users_with_log_and_hist():
    compute_function_for_key(user, 'alldomains_at_time_for_user')

def compute_alldomains_at_time_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key(user, 'alldomains_at_time_for_user')

  


def compute_log_with_mlog_active_times_for_user(user):
  mlog_active_times_for_user_raw = get_mlog_active_times_for_user(user)
  def add_evt_mlog_to_generator(gen):
    for x in gen:
      x['evt'] = 'mlog'
      yield x
  mlog_active_times_for_user = add_evt_mlog_to_generator(mlog_active_times_for_user_raw)
  logs_for_user = get_log_timesorted_lines_for_user(user)
  return orderedMerge(logs_for_user, mlog_active_times_for_user, key=itemgetter('time'))

def get_log_with_mlog_active_times_for_user(user):
  return get_function_for_key_lines(user, 'log_with_mlog_active_times_for_user')

def compute_log_with_mlog_active_times_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key_lines(user, 'log_with_mlog_active_times_for_user')

def compute_log_with_mlog_active_times_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key_lines(user, 'log_with_mlog_active_times_for_user')


def compute_mlog_timesorted_lines_for_user(user):
  mlogfile = get_mlogfile_for_user(user)
  alldata = json.load(open(mlogfile))
  alldata.sort(key=itemgetter('time'))
  return alldata

def get_mlog_timesorted_lines_for_user(user):
  return get_function_for_key_lines(user, 'mlog_timesorted_lines_for_user')

def compute_mlog_timesorted_lines_for_all_users():
  for user in list_users_with_mlog():
    compute_function_for_key_lines(user, 'mlog_timesorted_lines_for_user')

def compute_mlog_timesorted_lines_for_all_users_randomized():
  for user in shuffled(list_users_with_mlog()):
    compute_function_for_key_lines(user, 'mlog_timesorted_lines_for_user')


def compute_log_timesorted_lines_for_user(user):
  logfile = get_logfile_for_user(user)
  alldata = json.load(open(logfile))
  alldata.sort(key=itemgetter('time'))
  return alldata

def get_log_timesorted_lines_for_user(user):
  return get_function_for_key_lines(user, 'log_timesorted_lines_for_user')

def compute_log_timesorted_lines_for_all_users():
  for user in list_users_with_log():
    compute_function_for_key_lines(user, 'log_timesorted_lines_for_user')

def compute_log_timesorted_lines_for_all_users_randomized():
  for user in shuffled(list_users_with_log()):
    compute_function_for_key_lines(user, 'log_timesorted_lines_for_user')


def compute_hist_timesorted_lines_for_user(user):
  histfile = get_histfile_for_user(user)
  alldata = json.load(open(histfile))
  alldata.sort(key=itemgetter('time'))
  return alldata

def get_hist_timesorted_lines_for_user(user):
  return get_function_for_key_lines(user, 'hist_timesorted_lines_for_user')

def compute_hist_timesorted_lines_for_all_users():
  for user in list_users_with_hist():
    compute_function_for_key_lines(user, 'hist_timesorted_lines_for_user')

def compute_hist_timesorted_lines_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key_lines(user, 'hist_timesorted_lines_for_user')


def compute_reconstruct_focus_times_baseline_for_user(user):
  # baseline algorithm = 60 seconds idle time assumed
  reconstructor = ReconstructFocusTimesBaseline()
  for visit in get_history_ordered_visits_for_user(user):
    reconstructor.process_history_line(visit)
  return reconstructor.get_output()

def get_reconstruct_focus_times_baseline_for_user(user):
  return get_function_for_key_lines(user, 'reconstruct_focus_times_baseline_for_user')

def compute_reconstruct_focus_times_baseline_for_all_users():
  for user in list_users_with_hist():
    compute_function_for_key_lines(user, 'reconstruct_focus_times_baseline_for_user')

def compute_reconstruct_focus_times_baseline_for_all_users_randomized():
  for user in shuffled(list_users_with_hist()):
    compute_function_for_key_lines(user, 'reconstruct_focus_times_baseline_for_user')


def compute_url_switch_sources_for_user(user):
  output = [] # {url, prev, evt}
  prevurl = None
  for data in get_log_with_mlog_active_times_for_user(user):
    data = uncompress_data_subfields(data)
    cururl = get_focused_tab(data)
    evt = data['evt']
    if cururl != prevurl:
      output.append({'evt': evt, 'url': cururl, 'prev': prevurl})
      prevurl = cururl
  return output

def get_url_switch_sources_for_user(user):
  return get_function_for_key_lines(user, 'url_switch_sources_for_user')

def compute_url_switch_sources_for_all_users():
  for user in list_users_with_log_and_mlog():
    compute_function_for_key_lines(user, 'url_switch_sources_for_user')

def compute_url_switch_sources_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog()):
    compute_function_for_key_lines(user, 'url_switch_sources_for_user')


def extract_secondlevel_activespan_dataset_for_user(user, only_tenseconds=False):
  ordered_visits = get_history_ordered_visits_for_user(user)
  ordered_visits = exclude_bad_visits(ordered_visits)
  ordered_visits_len = len(ordered_visits)
  tab_focus_times = get_tab_focus_times_for_user(user)
  if len(tab_focus_times) == 0:
    return
  if len(ordered_visits) == 0:
    return
  active_seconds_set = set(get_active_seconds_for_user(user))
  ref_start_time = max(get_earliest_start_time(tab_focus_times), get_earliest_start_time(ordered_visits))
  ref_end_time = min(get_last_end_time(tab_focus_times), get_last_end_time(ordered_visits))
  ref_start_time = max(ref_start_time, 1458371950000) # march 19th. may have had some data loss prior to that
  ref_end_time = max(ref_end_time, 1458371950000)
  ref_start_time_seconds = ref_start_time/1000.0
  ref_end_time_seconds = ref_end_time/1000.0
  tab_focus_times_sortedcollection = SortedCollection(tab_focus_times, key=itemgetter('start'))
  for idx,visit in enumerate(ordered_visits):
    if idx+1 == ordered_visits_len: # last visit, we probably should reconstruct this TODO
      continue
    next_visit = ordered_visits[idx + 1]
    visit_time_seconds = int(round(visit['visitTime']/1000.0))
    next_visit_time_seconds = int(round(next_visit['visitTime']/1000.0))
    if visit_time_seconds < ref_start_time_seconds:
      continue
    if next_visit_time_seconds > ref_end_time_seconds:
      continue
    if visit_time_seconds >= next_visit_time_seconds:
      continue
    from_domain_id = domain_to_id(url_to_domain(visit['url']))
    to_domain_id = domain_to_id(url_to_domain(next_visit['url']))
    # we actually want to do this per second, not per millisecond! so want to actually round to nearest 1000 milliseconds
    for timestep in xrange(visit_time_seconds, next_visit_time_seconds+1):
      if not visit_time_seconds < timestep < next_visit_time_seconds:
        continue
      if only_tenseconds:
        if timestep % 10 != 0:
          continue
      sinceprev = timestep - visit_time_seconds
      tonext = next_visit_time_seconds - timestep
      label = int((sinceprev <= 60) or (timestep in active_seconds_set))
      yield [label, log(sinceprev), log(tonext), from_domain_id, to_domain_id]

def compute_secondlevel_activespan_dataset_for_user(user):
  return extract_secondlevel_activespan_dataset_for_user(user, False)

def get_secondlevel_activespan_dataset_for_user(user):
  return get_function_for_key_lines(user, 'secondlevel_activespan_dataset_for_user')

def compute_secondlevel_activespan_dataset_for_all_users():
  for user in list_users_with_log_and_mlog_and_hist():
    compute_function_for_key_lines(user, 'secondlevel_activespan_dataset_for_user')

def compute_secondlevel_activespan_dataset_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog_and_hist()):
    compute_function_for_key_lines(user, 'secondlevel_activespan_dataset_for_user')


def compute_tensecondlevel_activespan_dataset_for_user(user):
  return extract_secondlevel_activespan_dataset_for_user(user, True) # not a typo. it handles both

def get_tensecondlevel_activespan_dataset_for_user(user):
  return get_function_for_key_lines(user, 'tensecondlevel_activespan_dataset_for_user')

def compute_tensecondlevel_activespan_dataset_for_all_users():
  for user in list_users_with_log_and_mlog_and_hist():
    compute_function_for_key_lines(user, 'tensecondlevel_activespan_dataset_for_user')

def compute_tensecondlevel_activespan_dataset_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog_and_hist()):
    compute_function_for_key_lines(user, 'tensecondlevel_activespan_dataset_for_user')


def compute_tensecondlevel_activespan_dataset_labels_for_user(user):
  for line in get_tensecondlevel_activespan_dataset_for_user(user):
    yield line[0]

def get_tensecondlevel_activespan_labels_for_user(user):
  return get_function_for_key(user, 'tensecondlevel_activespan_dataset_labels_for_user')

def compute_tensecondlevel_activespan_dataset_labels_for_all_users():
  for user in list_users_with_log_and_mlog_and_hist():
    compute_function_for_key(user, 'tensecondlevel_activespan_dataset_labels_for_user')

def compute_tensecondlevel_activespan_dataset_labels_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog_and_hist()):
    compute_function_for_key(user, 'tensecondlevel_activespan_dataset_labels_for_user')





def compute_secondlevel_activespan_dataset_labels_for_user(user):
  for line in get_secondlevel_activespan_dataset_for_user(user):
    yield line[0]

def get_secondlevel_activespan_labels_for_user(user):
  return get_function_for_key(user, 'secondlevel_activespan_dataset_labels_for_user')

def compute_secondlevel_activespan_dataset_labels_for_all_users():
  for user in list_users_with_log_and_mlog_and_hist():
    compute_function_for_key(user, 'secondlevel_activespan_dataset_labels_for_user')

def compute_secondlevel_activespan_dataset_labels_for_all_users_randomized():
  for user in shuffled(list_users_with_log_and_mlog_and_hist()):
    compute_function_for_key(user, 'secondlevel_activespan_dataset_labels_for_user')


@memoized
def total_visits_of_domains_in_training():
  return sum_values_in_list_of_dict([get_domain_to_num_history_visits_for_user(user)for user in get_training_users()])

@memoized
def top_n_domains_by_visits(n=10):
  domain_to_usage = total_visits_of_domains_in_training()
  return [x[0] for x in sorted(domain_to_usage.items(), key=itemgetter(1), reverse=True)[:n]]








def get_users_with_data():
  users_with_data = []
  for user in list_users():
    ordered_visits = get_history_ordered_visits_for_user(user)
    if len(ordered_visits) == 0:
      continue
    tab_focus_times = get_tab_focus_times_for_user(user)
    if len(tab_focus_times) == 0:
      continue
    first_visit = tab_focus_times[0]
    first_visit_time = first_visit['start']
    first_visit_time = max(first_visit_time, ordered_visits[0]['visitTime']) / 1000.0
    last_visit = ordered_visits[-1]
    last_visit_time = float(last_visit['visitTime'])
    last_visit_time = min(last_visit_time, tab_focus_times[-1]) / 1000.0
    time_spent = last_visit_time - first_visit_time # seconds
    #print user, time_spent/(3600.0*24)
    if time_spent/(3600.0*24) > 10: # have at least 10 days of data
      users_with_data.append(user)
    #print user, datetime.datetime.fromtimestamp(last_visit_time)
  return users_with_data


@memoized
def get_training_and_test_users():
  all_available_users = get_users_with_data()
  half_of_all = len(all_available_users) / 2
  training_users = random.sample(all_available_users, half_of_all)
  training_users_set = set(training_users)
  test_users = [x for x in all_available_users if x not in training_users_set]
  return training_users,test_users

@jsonmemoized
def get_training_users():
  return get_training_and_test_users()[0]

@jsonmemoized
def get_test_users():
  return get_training_and_test_users()[1]


def get_tensecondlevel_activespan_dataset_train():
  for user in get_training_users():
    for x in get_tensecondlevel_activespan_dataset_for_user(user):
      yield x

def get_tensecondlevel_activespan_dataset_test():
  for user in get_test_users():
    for x in get_tensecondlevel_activespan_dataset_for_user(user):
      yield x


def compute_labels_for_tensecondlevel_train():
  for user in get_training_users():
    for x in get_tensecondlevel_activespan_labels_for_user(user):
      yield x

def get_labels_for_tensecondlevel_train():
  return create_and_get('labels_for_tensecondlevel_train.json')

def precompute_labels_for_tensecondlevel_train():
  return create_if_doesnt_exist('labels_for_tensecondlevel_train.json')

def compute_labels_for_tensecondlevel_test():
  for user in get_test_users():
    for x in get_tensecondlevel_activespan_labels_for_user(user):
      yield x

def get_labels_for_tensecondlevel_test():
  return create_and_get('labels_for_tensecondlevel_test.json')

def precompute_labels_for_tensecondlevel_test():
  return create_if_doesnt_exist('labels_for_tensecondlevel_test.json')


def compute_feature_vector_for_tensecondlevel_train(enabled_features):
  return dataset_to_feature_vectors(get_tensecondlevel_activespan_dataset_train(), enabled_features)

def get_feature_vector_for_tensecondlevel_train(enabled_features):
  return get_function_for_key(enabled_features, 'feature_vector_for_tensecondlevel_train')


def compute_feature_vector_for_tensecondlevel_test(enabled_features):
  return dataset_to_feature_vectors(get_tensecondlevel_activespan_dataset_test(), enabled_features)

def get_feature_vector_for_tensecondlevel_test(enabled_features):
  return get_function_for_key(enabled_features, 'feature_vector_for_tensecondlevel_test')


def compute_domain_id_to_productivity():
  max_domain_id = max(get_domains_to_id().viewvalues())
  output = [0 for i in xrange(max_domain_id+1)]
  for domain,productivity in get_domain_to_productivity().viewitems():
    try:
      domain_id = domain_to_id(domain)
    except:
      continue
    output[domain_id] = productivity
  return output

def get_domain_id_to_productivity():
  return create_and_get('domain_id_to_productivity.json')

def precompute_domain_id_to_productivity():
  return create_if_doesnt_exist('domain_id_to_productivity.json')

def compute_domain_id_to_category():
  max_domain_id = max(get_domains_to_id().viewvalues())
  output = ['' for i in xrange(max_domain_id+1)]
  for domain,category in get_domain_to_category().viewitems():
    try:
      domain_id = domain_to_id(domain)
    except:
      continue
    output[domain_id] = category
  return output

def get_domain_id_to_category():
  return create_and_get('domain_id_to_category.json')

def precompute_domain_id_to_category():
  return create_if_doesnt_exist('domain_id_to_category.json')





def iterate_mlogs_for_user(user):
  for line in get_mlog_timesorted_lines_for_user(user):
    yield uncompress_data_subfields(line)

def iterate_mlogs_for_user_compressed(user):
  return get_mlog_timesorted_lines_for_user(user)

def iterate_logs_for_user(user):
  for line in get_log_timesorted_lines_for_user(user):
    yield uncompress_data_subfields(line)

def iterate_logs_for_user_compressed(user):
  return get_log_timesorted_lines_for_user(user)

def iterate_hist_for_user(user):
  for line in get_hist_timesorted_lines_for_user(user):
    yield uncompress_data_subfields(line)

def iterate_hist_for_user_compressed(user):
  return get_hist_timesorted_lines_for_user(user)




