#!/usr/bin/env python
# md5: bf7b2868bc7ae4da2c27644880820ff8
# coding: utf-8




from tmilib import *


#user = get_test_users()[0]
#print user











#tab_focus_times = get_tab_focus_times_for_user(user)
#print tab_focus_times[0].keys()


#print active_second_to_domain_id.keys()[:10]


def get_recently_seen_domain_stats_for_user(user):
  ordered_visits = get_history_ordered_visits_corrected_for_user(user)
  ordered_visits = exclude_bad_visits(ordered_visits)
  active_domain_at_time = get_active_domain_at_time_for_user(user)
  active_seconds_set = set(get_active_seconds_for_user(user))
  active_second_to_domain_id = get_active_second_to_domain_id_for_user(user)
  recently_seen_domain_ids = [-1]*100
  stats = Counter()
  for idx,visit in enumerate(ordered_visits):
    if idx+1 >= len(ordered_visits):
      break
    next_visit = ordered_visits[idx+1]
    cur_domain = url_to_domain(visit['url'])
    cur_domain_id = domain_to_id(cur_domain)
    if cur_domain_id != recently_seen_domain_ids[-1]:
      recently_seen_domain_ids.append(cur_domain_id)
    next_domain = url_to_domain(next_visit['url'])
    next_domain_id = domain_to_id(next_domain)
    cur_time_sec = int(round(visit['visitTime'] / 1000.0))
    next_time_sec = int(round(visit['visitTime'] / 1000.0))
    for time_sec in xrange(cur_time_sec, next_time_sec+1):
      if time_sec not in active_seconds_set:
        continue
      ref_domain_id = active_second_to_domain_id[str(time_sec)]
      stats['total'] += 1
      if cur_domain_id == ref_domain_id:
        if next_domain_id == cur_domain_id:
          stats['first and next equal and correct'] += 1
          continue
        else:
          stats['first correct only'] += 1
          continue
      else:
        if next_domain_id == cur_domain_id:
          stats['both incorrect'] += 1
          found_match = False
          for i in range(1,101):
            if recently_seen_domain_ids[-1-i] == ref_domain_id:
              stats['nth previous correct ' + str(abs(i))] += 1
              stats['some previous among past 100 correct'] += 1
              found_match = True
              break
          if not found_match:
            stats['no match found'] += 1
          continue
        if next_domain_id == ref_domain_id:
          stats['next correct only'] += 1
          continue
  return stats


#total_stats = Counter({'total': 544544, 'first and next equal and correct': 351081, 'first correct only': 88522, 'both incorrect': 51663, 'some previous among past 20 correct': 41231, 'nth previous correct 1': 31202, 'next correct only': 23569, 'no match found': 10432, 'nth previous correct 2': 3311, 'nth previous correct 3': 1635, 'nth previous correct 4': 905, 'nth previous correct 5': 862, 'nth previous correct 6': 545, 'nth previous correct 7': 412, 'nth previous correct 8': 357, 'nth previous correct 9': 269, 'nth previous correct 10': 259, 'nth previous correct 13': 234, 'nth previous correct 11': 229, 'nth previous correct 12': 190, 'nth previous correct 15': 183, 'nth previous correct 14': 140, 'nth previous correct 17': 139, 'nth previous correct 20': 95, 'nth previous correct 16': 90, 'nth previous correct 19': 88, 'nth previous correct 18': 86})

total_stats = Counter()

for user in get_test_users():
  for k,v in get_recently_seen_domain_stats_for_user(user).viewitems():
    total_stats[k] += v

#total_stats = Counter({'total': 544544, 'first and next equal and correct': 351081, 'first correct only': 88522, 'both incorrect': 51663, 'some previous among past 20 correct': 41136, 'nth previous correct 2': 31202, 'next correct only': 23569, 'no match found': 10527, 'nth previous correct 3': 3311, 'nth previous correct 4': 1635, 'nth previous correct 5': 905, 'nth previous correct 6': 862, 'nth previous correct 7': 545, 'nth previous correct 8': 412, 'nth previous correct 9': 357, 'nth previous correct 10': 269, 'nth previous correct 11': 259, 'nth previous correct 14': 234, 'nth previous correct 12': 229, 'nth previous correct 13': 190, 'nth previous correct 16': 183, 'nth previous correct 15': 140, 'nth previous correct 18': 139, 'nth previous correct 17': 90, 'nth previous correct 20': 88, 'nth previous correct 19': 86})


print total_stats


def sumkeys(d, *args):
  return sum(d.get(x, 0.0) for x in args)


norm = {k:float(v)/total_stats['total'] for k,v in total_stats.viewitems()}
print 'select prev gets answer correct', sumkeys(norm, 'first and next equal and correct', 'first correct only')
print 'prev or next gets answer correct', sumkeys(norm, 'first and next equal and correct', 'first correct only', 'next correct only')
for i in range(1, 101):
  sumprev = sum([norm.get('nth previous correct '+str(x),0.0) for x in range(i+1)])
  print 'prev or next or past ' + str(i), sumkeys(norm, 'first and next equal and correct', 'first correct only', 'next correct only')+sumprev

