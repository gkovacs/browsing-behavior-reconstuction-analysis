#!/usr/bin/env python
# md5: e666fa10cde2987fe972628d79689bb3
# coding: utf-8

from tmilib import *


'''
print 'get_mturkid_to_time_last_active'
get_mturkid_to_time_last_active()
print 'get_username_to_mturk_id'
get_username_to_mturk_id()
print 'get_mturkid_to_history_pages'
get_mturkid_to_history_pages()
print 'get_mturkid_to_history_visits'
get_mturkid_to_history_visits()
print 'get_domains_list'
get_domains_list()
'''


if random.random() > 0.5:
  print 'history_pages_for_all_users_randomized'
  compute_history_pages_for_all_users_randomized()
else:
  print 'history_visits_for_all_users_randomized'
  compute_history_visits_for_all_users_randomized()
#compute_tab_focus_times_for_all_users_randomized()


#print 'compute_tab_focus_times_for_all_users'
#compute_tab_focus_times_for_all_users()

