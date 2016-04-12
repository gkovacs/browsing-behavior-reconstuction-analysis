#!/usr/bin/env python
# md5: db79ba39b675187b524720b8d21b3244
# coding: utf-8

from tmilib import *
import csv


import sys
num_prev_enabled = int(sys.argv[1])
num_labels_enabled = 1 + num_prev_enabled # since we disabled the n label
data_version = 4+8+8+8+8 + num_prev_enabled
print 'num_prev_enabled', num_prev_enabled
print 'data_version', data_version


twenty_letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]
#domain_to_letter = {x:twenty_letters[i] for i,x in enumerate(top_domains)}
domain_id_to_letter = {domain_to_id(x):twenty_letters[i] for i,x in enumerate(top_n_domains_by_visits(20))}
#print domain_id_to_letter
#print domain_to_letter


productivity_letters = {-2: 'v', -1: 'w', 0: 'x', 1: 'y', 2: 'z'}
domain_id_to_productivity_letter = [productivity_letters[x] for x in get_domain_id_to_productivity()]
#print domain_id_to_productivity[:10]
#print domain_id_to_productivity_letter[:10]


def get_row_names(include_domain=False):
  output_row_names = [
    'label',
    'spanlen',
    'since_cur',
    'cur_domain_letter',
    'cur_domain_productivity',
    'to_next',
    'next_domain_letter',
    #'next_domain_productivity',
    'n_eq_c',
    'nref_eq_c',
    'nref_eq_zero'
  ]
  for idx_p_zeroidx in range(num_prev_enabled):
    sp = str(idx_p_zeroidx + 1)
    new_feature_names_for_p = [
      'since_prev' + sp,
      'prev' + sp +'_domain_letter',
      #'prev' + sp + '_domain_productivity',
      'n_eq_p' + sp,
      'nref_eq_p' + sp,
    ]
    output_row_names.extend(new_feature_names_for_p)
  # only v8 and onwards
  output_row_names.extend([
    'switchto_in_session_cur',
    #'switchfrom_in_session_cur',
    'switchto_in_session_next',
    #'switchfrom_in_session_next',
  ])
  for idx_p_zeroidx in range(num_prev_enabled):
    sp = str(idx_p_zeroidx + 1)
    new_feature_names_for_p = [
      'switchto_in_session_prev' + sp,
      #'switchfrom_in_session_prev' + sp,
    ]
    output_row_names.extend(new_feature_names_for_p)
  if include_domain:
    output_row_names.insert(0, 'time_sec')
    output_row_names.insert(1, 'user')
    output_row_names.insert(2, 'ref_domain')
  return tuple(output_row_names)

row_names = []
#row_names = get_row_names()
#print row_names


def get_rows_for_user(user, include_domain=False):
  output = []
  #ordered_visits = get_history_ordered_visits_corrected_for_user(user)
  ordered_visits = get_history_ordered_visits_corrected_for_user(user)
  ordered_visits = exclude_bad_visits(ordered_visits)
  #active_domain_at_time = get_active_domain_at_time_for_user(user)
  active_seconds_set = set(get_active_insession_seconds_for_user(user))
  active_second_to_domain_id = {int(k):v for k,v in get_active_second_to_domain_id_for_user(user).viewitems()}
  prev_domain_ids = [-1]*8
  domain_id_to_most_recent_visit = {}
  domain_id_to_num_switchto = Counter()
  #domain_id_to_num_switchfrom = Counter()
  total_items = 0
  skipped_items = 0
  prev_visit_time = 0
  prev_visit_domain_id = -1
  visit_id_to_domain_id = {}
  for idx,visit in enumerate(ordered_visits):
    if idx+1 >= len(ordered_visits):
      break
    
    next_visit = ordered_visits[idx+1]
    referring_visit_id = next_visit['referringVisitId']
    nref_eq_zero = 'T' if referring_visit_id == 0 else 'F'

    new_session = False
    if visit['visitTime'] > prev_visit_time + 1000*60*20:
      new_session = True
    prev_visit_time = visit['visitTime']
    if new_session:
      #prev_visit_domain_id = -1
      #prev_domain_ids = [-1]*8
      #domain_id_to_most_recent_visit = {}
      domain_id_to_num_switchto = Counter()
      #domain_id_to_num_switchfrom = Counter()
    
    cur_domain = url_to_domain(visit['url'])
    cur_domain_id = domain_to_id(cur_domain)
    next_domain = url_to_domain(next_visit['url'])
    next_domain_id = domain_to_id(next_domain)

    visit_id_to_domain_id[visit['visitId']] = cur_domain_id
    nref_domain_id = visit_id_to_domain_id.get(referring_visit_id)
    
    if cur_domain_id != prev_visit_domain_id:
      domain_id_to_num_switchto[cur_domain_id] += 1
      prev_visit_domain_id = cur_domain_id

    cur_time_sec = int(round(visit['visitTime'] / 1000.0))
    next_time_sec = int(round(next_visit['visitTime'] / 1000.0))
    
    domain_id_to_most_recent_visit[cur_domain_id] = cur_time_sec
    if prev_domain_ids[0] != cur_domain_id:
      #prev_domain_ids = ([cur_domain_id] + [x for x in prev_domain_ids if x != cur_domain_id])[:4]
      if cur_domain_id in prev_domain_ids:
        prev_domain_ids.remove(cur_domain_id)
      prev_domain_ids.insert(0, cur_domain_id)
      while len(prev_domain_ids) > 8:
        prev_domain_ids.pop()
    # prev_domain_ids includes the current one

    if cur_time_sec > next_time_sec:
      continue

    prev1_domain_id = prev_domain_ids[1]
    prev2_domain_id = prev_domain_ids[2]
    prev3_domain_id = prev_domain_ids[3]
    prev4_domain_id = prev_domain_ids[4]
    prev5_domain_id = prev_domain_ids[5]
    prev6_domain_id = prev_domain_ids[6]
    prev7_domain_id = prev_domain_ids[7]
    n_eq_c = 'T' if (next_domain_id == cur_domain_id) else 'F'
    n_eq_p1 = 'T' if (next_domain_id == prev1_domain_id) else 'F'
    n_eq_p2 = 'T' if (next_domain_id == prev2_domain_id) else 'F'
    n_eq_p3 = 'T' if (next_domain_id == prev3_domain_id) else 'F'
    n_eq_p4 = 'T' if (next_domain_id == prev4_domain_id) else 'F'
    n_eq_p5 = 'T' if (next_domain_id == prev5_domain_id) else 'F'
    n_eq_p6 = 'T' if (next_domain_id == prev6_domain_id) else 'F'
    n_eq_p7 = 'T' if (next_domain_id == prev7_domain_id) else 'F'
    
    nref_eq_c = 'T' if nref_domain_id == cur_domain_id else 'F'
    nref_eq_p1 = 'T' if nref_domain_id == prev1_domain_id else 'F'
    nref_eq_p2 = 'T' if nref_domain_id == prev2_domain_id else 'F'
    nref_eq_p3 = 'T' if nref_domain_id == prev3_domain_id else 'F'
    nref_eq_p4 = 'T' if nref_domain_id == prev4_domain_id else 'F'
    nref_eq_p5 = 'T' if nref_domain_id == prev5_domain_id else 'F'
    nref_eq_p6 = 'T' if nref_domain_id == prev6_domain_id else 'F'
    nref_eq_p7 = 'T' if nref_domain_id == prev7_domain_id else 'F'
    
    
    for time_sec in xrange(cur_time_sec+1, next_time_sec):
      if time_sec not in active_seconds_set:
        continue
      ref_domain_id = active_second_to_domain_id[time_sec]
      ref_domain = id_to_domain(ref_domain_id)
      total_items += 1
      label = None
      available_labels = (
        (cur_domain_id, 'c'),
        # (next_domain_id, 'n'),
        (prev1_domain_id, 'p1'),
        (prev2_domain_id, 'p2'),
        (prev3_domain_id, 'p3'),
        (prev4_domain_id, 'p4'),
        (prev5_domain_id, 'p5'),
        (prev6_domain_id, 'p6'),
        (prev7_domain_id, 'p7'),
      )[:num_labels_enabled]
      # c p n p q r s t
      for label_value,label_name in available_labels:
        if ref_domain_id == label_value:
          label = label_name
          break
      if label == None:
        if include_domain:
          label = 'u'
        else:
          skipped_items += 1
          continue

      next_domain_letter = domain_id_to_letter.get(next_domain_id, 'u')
      cur_domain_letter = domain_id_to_letter.get(cur_domain_id, 'u')
      prev1_domain_letter = domain_id_to_letter.get(prev1_domain_id, 'u')
      prev2_domain_letter = domain_id_to_letter.get(prev2_domain_id, 'u')
      prev3_domain_letter = domain_id_to_letter.get(prev3_domain_id, 'u')
      prev4_domain_letter = domain_id_to_letter.get(prev4_domain_id, 'u')
      prev5_domain_letter = domain_id_to_letter.get(prev5_domain_id, 'u')
      prev6_domain_letter = domain_id_to_letter.get(prev6_domain_id, 'u')
      prev7_domain_letter = domain_id_to_letter.get(prev7_domain_id, 'u')
      
      next_domain_productivity = domain_id_to_productivity_letter[next_domain_id]
      cur_domain_productivity = domain_id_to_productivity_letter[cur_domain_id]
      prev1_domain_productivity = domain_id_to_productivity_letter[prev1_domain_id]
      prev2_domain_productivity = domain_id_to_productivity_letter[prev2_domain_id]
      prev3_domain_productivity = domain_id_to_productivity_letter[prev3_domain_id]
      prev4_domain_productivity = domain_id_to_productivity_letter[prev4_domain_id]
      prev5_domain_productivity = domain_id_to_productivity_letter[prev5_domain_id]
      prev6_domain_productivity = domain_id_to_productivity_letter[prev6_domain_id]
      prev7_domain_productivity = domain_id_to_productivity_letter[prev7_domain_id]
      
      since_cur = time_sec - cur_time_sec
      to_next = next_time_sec - time_sec
      spanlen = since_cur + to_next
      prev1_domain_last_visit = domain_id_to_most_recent_visit.get(prev1_domain_id, 0)
      prev2_domain_last_visit = domain_id_to_most_recent_visit.get(prev2_domain_id, 0)
      prev3_domain_last_visit = domain_id_to_most_recent_visit.get(prev3_domain_id, 0)
      prev3_domain_last_visit = domain_id_to_most_recent_visit.get(prev3_domain_id, 0)
      prev4_domain_last_visit = domain_id_to_most_recent_visit.get(prev4_domain_id, 0)
      prev5_domain_last_visit = domain_id_to_most_recent_visit.get(prev5_domain_id, 0)
      prev6_domain_last_visit = domain_id_to_most_recent_visit.get(prev6_domain_id, 0)
      prev7_domain_last_visit = domain_id_to_most_recent_visit.get(prev7_domain_id, 0)
      
      since_prev1 = time_sec - prev1_domain_last_visit
      since_prev2 = time_sec - prev2_domain_last_visit
      since_prev3 = time_sec - prev3_domain_last_visit
      since_prev4 = time_sec - prev4_domain_last_visit
      since_prev5 = time_sec - prev5_domain_last_visit
      since_prev6 = time_sec - prev6_domain_last_visit
      since_prev7 = time_sec - prev7_domain_last_visit
      
      since_cur = log(since_cur)
      to_next = log(to_next)
      spanlen = log(spanlen)
      since_prev1 = log(since_prev1)
      since_prev2 = log(since_prev2)
      since_prev3 = log(since_prev3)
      since_prev4 = log(since_prev4)
      since_prev5 = log(since_prev5)
      since_prev6 = log(since_prev6)
      since_prev7 = log(since_prev7)
      
      switchto_in_session_cur = domain_id_to_num_switchto[cur_domain_id]
      switchto_in_session_next = domain_id_to_num_switchto[next_domain_id]
      switchto_in_session_prev1 = domain_id_to_num_switchto[prev1_domain_id]
      switchto_in_session_prev2 = domain_id_to_num_switchto[prev2_domain_id]
      switchto_in_session_prev3 = domain_id_to_num_switchto[prev3_domain_id]
      switchto_in_session_prev4 = domain_id_to_num_switchto[prev4_domain_id]
      switchto_in_session_prev5 = domain_id_to_num_switchto[prev5_domain_id]
      switchto_in_session_prev6 = domain_id_to_num_switchto[prev6_domain_id]
      switchto_in_session_prev7 = domain_id_to_num_switchto[prev7_domain_id]
            
      cached_locals = locals()
      output.append([cached_locals[row_name] for row_name in row_names])
  #print 'user', user, 'guaranteed error', float(skipped_items)/total_items, 'skipped', skipped_items, 'total', total_items
  return {
    'rows': output,
    'skipped_items': skipped_items,
    'total_items': total_items,
  }





def create_domainclass_data_for_users(users, filename, include_domain=False):
  if sdir_exists(filename):
    print 'already exists', filename
    return
  outfile = csv.writer(open(sdir_path(filename), 'w'))
  global row_names
  row_names = get_row_names(include_domain)
  outfile.writerow(row_names)
  total_items = 0
  skipped_items = 0
  for user in users:
    data = get_rows_for_user(user, include_domain)
    total_items += data['total_items']
    if total_items == 0:
      print user, 'no items'
      continue
    skipped_items += data['skipped_items']
    print user, 'skipped', float(data['skipped_items'])/data['total_items'], 'skipped', data['skipped_items'], 'total', data['total_items']
    outfile.writerows(data['rows'])
  print 'guaranteed error', float(skipped_items) / total_items, 'skipped', skipped_items, 'total', total_items



create_domainclass_data_for_users(get_training_users(), 'domainclass_cpn_train_v' + str(data_version) +'.csv')
create_domainclass_data_for_users(get_test_users(), 'domainclass_cpn_test_v' + str(data_version) + '.csv')
create_domainclass_data_for_users(get_test_users(), 'domainclass_cpn_test_all_withdomain_v' + str(data_version) + '.csv', True)

