#!/usr/bin/env python
# md5: b8d418033e021ca6125dd881ea24ba5a
# coding: utf-8

import csv


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
  outfile.writerow(['label', 'sinceprev', 'tonext', 'samedomain', 'fromdomain', 'todomain', 'fromprod', 'toprod'])
  for user in users:
    for line in get_data_func(user):
      label = 'T' if line[0] else 'F'
      f1 = line[1]
      f2 = line[2]
      samedomain = 'T' if (line[3] == line[4]) else 'F'
      fromdomain = domain_id_to_letter.get(line[3], 'u')
      todomain = domain_id_to_letter.get(line[4], 'u')
      fromdomain_prod = domain_id_to_productivity_letter[line[3]]
      todomain_prod = domain_id_to_productivity_letter[line[4]]
      outfile.writerow([label, f1, f2, samedomain, fromdomain, todomain, fromdomain_prod, todomain_prod])



training_users = get_training_users()
test_users = get_test_users()

user_data_to_csv('catdata_train_tensecond.csv', training_users, get_tensecondlevel_activespan_dataset_for_user)
user_data_to_csv('catdata_test_tensecond.csv', test_users, get_tensecondlevel_activespan_dataset_for_user)
user_data_to_csv('catdata_train_insession_tensecond.csv', training_users, get_tensecondlevel_activespan_dataset_insession_for_user)
user_data_to_csv('catdata_test_insession_tensecond.csv', test_users, get_tensecondlevel_activespan_dataset_insession_for_user)
user_data_to_csv('catdata_train_second.csv', training_users, get_secondlevel_activespan_dataset_for_user)
user_data_to_csv('catdata_test_second.csv', test_users, get_secondlevel_activespan_dataset_for_user)
user_data_to_csv('catdata_train_insession_second.csv', training_users, get_secondlevel_activespan_dataset_insession_for_user)
user_data_to_csv('catdata_test_insession_second.csv', test_users, get_secondlevel_activespan_dataset_insession_for_user)

