#!/usr/bin/env python
# md5: 4ceaa82e414829a351b7155e21150cc4
# coding: utf-8

from tmilib import *

#compute_url_switch_sources_for_all_users_randomized()
overall_user_switch_sources = Counter()

for user in list_users_with_log_and_mlog():
  for data in get_url_switch_sources_for_user(user):
    evt = data['evt']
    overall_user_switch_sources[evt] += 1


print_counter(overall_user_switch_sources)


urls_transitioned_to_via_mlog = Counter()
urls_transitioned_from_via_mlog = Counter()
url_transitions_via_mlog = Counter()

for user in list_users_with_log_and_mlog():
  for data in get_url_switch_sources_for_user(user):
    evt = data['evt']
    url = data['url']
    prev = data['prev']
    transition = unicode(prev) + ' to ' + unicode(url)
    if evt == 'mlog':
      urls_transitioned_to_via_mlog[url] += 1
      urls_transitioned_from_via_mlog[prev] += 1
      url_transitions_via_mlog[transition] += 1


print_counter(urls_transitioned_to_via_mlog)


print_counter(urls_transitioned_from_via_mlog)


print_counter(url_transitions_via_mlog)


'''
def compute_url_switch_sources_for_user(user):
  output = [] # {url, prev, evt}
  prevurl = None
  for data in get_log_with_mlog_active_times_for_user(user):
    data = uncompress_data_subfields(data)
    cururl = get_focused_tab(data)
    if cururl != prevurl:
      output.append({'evt': evt, 'url': cururl, 'prev': prevurl})
      prevurl = cururl
  return output
'''


#print list_users()[0]
#for x in compute_url_switch_sources_for_user('Eq7EExfolE'):
#  print x

