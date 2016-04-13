#!/usr/bin/env python
# md5: 42ba1a93fe3c71d01bb611ccc54772ef
# coding: utf-8

import csv
import math


from tmilib import *


#user = get_training_users()[0]
#print user


#dataset = get_secondlevel_activespan_dataset_for_user(user)


#top_domains = top_n_domains_by_visits(20)


#print top_domains


twenty_letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]
#domain_to_letter = {x:twenty_letters[i] for i,x in enumerate(top_domains)}
domain_id_to_letter = {domain_to_id(x):twenty_letters[i] for i,x in enumerate(top_n_domains_by_visits(20))}
#print domain_id_to_letter
#print domain_to_letter


productivity_letters = {-2: 'v', -1: 'w', 0: 'x', 1: 'y', 2: 'z'}
domain_id_to_productivity_letter = [productivity_letters[x] for x in get_domain_id_to_productivity()]
#print domain_id_to_productivity[:10]
#print domain_id_to_productivity_letter[:10]


def user_data_to_csv(outpath, users, get_data_func):
  full_outpath = sdir_path(outpath)
  if sdir_exists(outpath):
    print 'already exists', full_outpath
    return
  print outpath
  outfile = csv.writer(open(full_outpath, 'w'))
  outfile.writerow(['user', 'time_sec', 'label', 'sinceprev', 'tonext', 'spanlen', 'samedomain', 'fromdomain', 'todomain', 'fromprod', 'toprod'])
  for user in users:
    for time_sec, is_active, sinceprev, tonext, spanlen, from_domain_id, to_domain_id in get_data_func(user):
      label = 'T' if is_active else 'F'
      samedomain = 'T' if (from_domain_id == to_domain_id) else 'F'
      fromdomain = domain_id_to_letter.get(from_domain_id, 'u')
      todomain = domain_id_to_letter.get(to_domain_id, 'u')
      fromdomain_prod = domain_id_to_productivity_letter[from_domain_id]
      todomain_prod = domain_id_to_productivity_letter[to_domain_id]
      outfile.writerow([user, time_sec, label, sinceprev, tonext, spanlen, samedomain, fromdomain, todomain, fromdomain_prod, todomain_prod])



def get_activespan_data(user, tensecond_only, insession_only):
  output = []
  ordered_visits = get_history_ordered_visits_corrected_for_user(user)
  num_ordered_visits = len(ordered_visits)
  active_seconds_set = set(get_active_insession_seconds_for_user(user))
  insession_seconds_list = get_insession_both_seconds_for_user(user)
  if len(insession_seconds_list) == 0:
    print 'warning have no data for', user
    return []
  insession_seconds_set = set(insession_seconds_list)
  first_session_start = insession_seconds_list[0]
  last_session_end = insession_seconds_list[-1]
  for idx,visit in enumerate(ordered_visits):
    if idx+1 >= num_ordered_visits:
      break
    next_visit = ordered_visits[idx+1]
    cur_time_sec = int(round(visit['visitTime']/1000.0))
    next_time_sec = int(round(next_visit['visitTime']/1000.0))
    from_domain_id = domain_to_id(url_to_domain(visit['url']))
    to_domain_id = domain_to_id(url_to_domain(next_visit['url']))
    if cur_time_sec >= next_time_sec:
      continue
    for time_sec in xrange(cur_time_sec+1, next_time_sec): # leave out seconds exactly on the marker - they will be errors
      if not first_session_start <= time_sec <= last_session_end:
        continue
      if insession_only and (time_sec not in insession_seconds_set):
        continue
      if tensecond_only and (time_sec % 10 != 0):
        continue
      is_active = time_sec in active_seconds_set
      sinceprev = time_sec - cur_time_sec
      tonext = next_time_sec - time_sec
      spanlen = sinceprev + tonext
      #if sinceprev == 0:
      #  is_active = True
      #  sinceprev = 0.0001
      #if tonext == 0:
      #  is_active = True
      #  tonext = 0.0001
      output.append([time_sec, is_active, log(sinceprev), log(tonext), log(spanlen), from_domain_id, to_domain_id])
  return output

def get_activespan_data_tensecond(user):
  return get_activespan_data(user, True, False)
      
def get_activespan_data_tensecond_insession(user):
  return get_activespan_data(user, True, True)

def get_activespan_data_second(user):
  return get_activespan_data(user, False, False)

def get_activespan_data_second_insession(user):
  return get_activespan_data(user, False, True)



training_users = get_training_users()
test_users = get_test_users()



user_data_to_csv('catdata_train_tensecond_evaluation_v4.csv', training_users, get_activespan_data_tensecond)
user_data_to_csv('catdata_test_tensecond_evaluation_v4.csv', test_users, get_activespan_data_tensecond)
user_data_to_csv('catdata_train_insession_tensecond_evaluation_v4.csv', training_users, get_activespan_data_tensecond_insession)
user_data_to_csv('catdata_test_insession_tensecond_evaluation_v4.csv', test_users, get_activespan_data_tensecond_insession)
user_data_to_csv('catdata_train_second_evaluation_v4.csv', training_users, get_activespan_data_second)
user_data_to_csv('catdata_test_second_evaluation_v4.csv', test_users, get_activespan_data_second)
user_data_to_csv('catdata_train_insession_second_evaluation_v4.csv', training_users, get_activespan_data_second_insession)
user_data_to_csv('catdata_test_insession_second_evaluation_v4.csv', test_users, get_activespan_data_second_insession)

'''
user_data_to_csv('catdata_train_tensecond.csv', training_users, get_tensecondlevel_activespan_dataset_for_user)
user_data_to_csv('catdata_test_tensecond.csv', test_users, get_tensecondlevel_activespan_dataset_for_user)
user_data_to_csv('catdata_train_insession_tensecond.csv', training_users, get_tensecondlevel_activespan_dataset_insession_for_user)
user_data_to_csv('catdata_test_insession_tensecond.csv', test_users, get_tensecondlevel_activespan_dataset_insession_for_user)
user_data_to_csv('catdata_train_second.csv', training_users, get_secondlevel_activespan_dataset_for_user)
user_data_to_csv('catdata_test_second.csv', test_users, get_secondlevel_activespan_dataset_for_user)
user_data_to_csv('catdata_train_insession_second.csv', training_users, get_secondlevel_activespan_dataset_insession_for_user)
user_data_to_csv('catdata_test_insession_second.csv', test_users, get_secondlevel_activespan_dataset_insession_for_user)
'''

