#!/usr/bin/env python
# md5: 0cfd9e6278088360c93439887b4984d8
# coding: utf-8

from tmilib import *

#tmi_overrides['basedir'] = '/home/gkovacs/tmi-data/local_2016-03-30_16:39:38-07:00'

from reconstruct_focus_times_common import *
from sorted_collection import SortedCollection

import sklearn
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.tree

from math import log





training_users = get_training_users()
test_users = get_test_users()


#print len(training_users)
#print len(test_users)
#print sorted(training_users)
#print sorted(test_users)


#user = get_users_with_data()[0]
#print user
#user = '3a3FX1s9S6'


#for line in get_log_with_mlog_active_times_for_user(user):
#  if line['evt'] != 'tab_updated':
#    continue
#  print line
#  break


#tab_focus_times = get_tab_focus_times_for_user(user)
#reconstructed_tab_focus_times = get_tab_focus_times_only_tab_updated_for_user(user)
#print evalutate_tab_focus_reconstruction(tab_focus_times, reconstructed_tab_focus_times)


#tab_focus_times = get_tab_focus_times_for_user(user)
#reconstructed_tab_focus_times = get_tab_focus_times_only_tab_updated_urlchanged_for_user(user)
#print evalutate_tab_focus_reconstruction(tab_focus_times, reconstructed_tab_focus_times)


#tab_focus_times = get_tab_focus_times_for_user(user)
#reconstructed_tab_focus_times = list(get_reconstruct_focus_times_baseline_for_user(user))
#print evalutate_tab_focus_reconstruction(tab_focus_times, reconstructed_tab_focus_times)


# first: annotate the history visit items with whether the time between the two is majority activity or not
#for visit in get_history_ordered_visits_for_user(user):
#  print visit['transition']
#  break

def fraction_active_between_times(tab_focus_times_sortedcollection, target_start, target_end):
  # between start and end, does tab_focus_times indicate that majority activity is occurring?
  try:
    item_before_start = tab_focus_times_sortedcollection.find_le(target_start)
    item_before_start_idx = tab_focus_times_sortedcollection.index(item_before_start)
  except:
    item_before_start_idx = 0
  # if have valueerror, make it 0
  try:
    item_after_end = tab_focus_times_sortedcollection.find_ge(target_end)
    item_after_end_idx = tab_focus_times_sortedcollection.index(item_after_end)
  except:
    item_after_end_idx = len(tab_focus_times_sortedcollection) - 1
  # if have valueerror, make it the last item in the list
  time_span_duration = target_end - target_start
  if time_span_duration <= 60*1000: # less than 60 seconds
    return 1.0 # is this perhaps a cause of error?
  if time_span_duration == 0:
    return 1.0
    #raise Exception('target_start == target_end both have value ' + str(target_start))
    #return 0
  if time_span_duration < 0:
    raise Exception('target_end < target_start end=' + str(target_end) + ' start=' + str(target_start))
  total_coverage = 0.0
  for idx in range(item_before_start_idx, item_after_end_idx+1):
    item = tab_focus_times_sortedcollection[idx]
    if item['start'] == item['end']:
      continue
      #raise Exception('item has length 0')
    if item['start'] > item['end']:
      raise Exception('item start > end')
    if item['end'] < target_start:
      continue
    if item['start'] > target_end:
      continue
    start = max(target_start, item['start'])
    if start > target_end:
      continue
    end = min(target_end, item['end'])
    if start > end:
      raise Exception('start is greater than end start=' + str(start) + ' end=' + str(end))
    start_percentage = (start - target_start) / float(time_span_duration)
    if not 0.0 <= start_percentage <= 1.0:
      continue
    end_percentage = (end - target_start) / float(time_span_duration)
    if not 0.0 <= end_percentage <= 1.0:
      continue
    total_coverage += (end_percentage - start_percentage)
  return total_coverage

def have_majority_activity(tab_focus_times_sortedcollection, target_start, target_end):
  return fraction_active_between_times(tab_focus_times_sortedcollection, target_start, target_end) > 0.5


'''
def get_domain_to_visit_durations():
  output = {}
  for user in training_users:
    tab_focus_times = get_tab_focus_times_for_user(user)
    for idx,item in enumerate(tab_focus_times):
      url = item['url']
      domain = url_to_domain(url)
      duration = item['end'] - item['start']
      if duration <= 0:
        continue
      if domain not in output:
        output[domain] = []
      output[domain].append(duration)
  return output
get_domain_to_visit_durations()
'''

#def get_domain_to_canonical_visit_duration():
#  output = {}
#  for domain,visit_durations in get_domain_to_visit_durations().items():
#    output[domain] = numpy.median()


#domain_to_visit_durations = get_domain_to_visit_durations()


#print numpy.mean([x for x in domain_to_visit_durations['www.youtube.com'] if x > 30000])


#def get_domain_to_duration_active_after_domain():
#  output = {}
  


'''
tab_focus_times = get_tab_focus_times_for_user(user)
#tab_focus_times = get_tab_focus_times_only_tab_updated_urlchanged_for_user(user)
tab_focus_times_sortedcollection = SortedCollection(tab_focus_times, key=itemgetter('start'))

ordered_visits = get_history_ordered_visits_for_user(user)
ordered_visits_len = len(ordered_visits)

ref_start_time = max(get_earliest_start_time(tab_focus_times), get_earliest_start_time(ordered_visits))
ref_end_time = min(get_last_end_time(tab_focus_times), get_last_end_time(ordered_visits))
'''


#print (ref_end_time - ref_start_time)/(1000.0*3600*24)


#print ordered_visits[0]


'''
for idx,visit in enumerate(ordered_visits):
  if idx+1 == ordered_visits_len: # last visit
    continue
  next_visit = ordered_visits[idx + 1]
  visit_time = visit['visitTime']
  next_visit_time = next_visit['visitTime']
  if visit_time < ref_start_time:
    continue
  if next_visit_time > ref_end_time:
    continue
  if visit_time == next_visit_time:
    #print visit
    #print next_visit
    if visit['url'] != next_visit['url']:
      print visit
      print next_visit
      break
'''





def extract_tofill_dataset_from_user(user):
  training_samples = []
  training_labels = []
  training_weights = []
  from_domains = []
  to_domains = []
  #ordered_visits = get_history_ordered_visits_for_user(user)
  #ordered_visits = exclude_bad_visits(ordered_visits)
  #ordered_visits = get_idealized_history_from_logs_for_user(user)
  ordered_visits = get_idealized_history_from_logs_urlchanged_for_user(user)
  ordered_visits_len = len(ordered_visits)
  tab_focus_times = get_tab_focus_times_for_user(user)
  ref_start_time = max(get_earliest_start_time(tab_focus_times), get_earliest_start_time(ordered_visits))
  ref_end_time = min(get_last_end_time(tab_focus_times), get_last_end_time(ordered_visits))
  ref_start_time = max(ref_start_time, 1458371950000) # march 19th. may have had some data loss prior to that
  ref_end_time = max(ref_end_time, 1458371950000)
  tab_focus_times_sortedcollection = SortedCollection(tab_focus_times, key=itemgetter('start'))
  for idx,visit in enumerate(ordered_visits):
    if idx+1 == ordered_visits_len: # last visit, we probably should reconstruct this TODO
      continue
    next_visit = ordered_visits[idx + 1]
    visit_time = visit['visitTime']
    next_visit_time = next_visit['visitTime']
    if visit_time < ref_start_time:
      continue
    if next_visit_time > ref_end_time:
      continue
    if visit_time >= next_visit_time:
      continue
    fraction_active = fraction_active_between_times(tab_focus_times_sortedcollection, visit_time, next_visit_time)
    label = int(fraction_active > 0.5)
    visit_gap = log(next_visit_time - visit_time)
    weight = next_visit_time - visit_time
    training_samples.append([visit_gap])
    training_labels.append(label)
    training_weights.append(weight) # should we try weight or log(weight)
    from_domain = url_to_domain(visit['url'])
    to_domain = url_to_domain(next_visit['url'])
    from_domains.append(from_domain)
    to_domains.append(to_domain)
  return {
    'samples': training_samples,
    'labels': training_labels,
    'weights': training_weights,
    'fromdomains': from_domains,
    'todomains': to_domains,
  }

def extract_tofill_dataset_for_users(users):
  all_training_samples = []
  all_training_labels = []
  all_training_weights = []
  all_from_domains = []
  all_to_domains = []
  for user in users:
    data = extract_tofill_dataset_from_user(user)
    all_training_samples.extend(data['samples'])
    all_training_labels.extend(data['labels'])
    all_training_weights.extend(data['weights'])
    all_from_domains.extend(data['fromdomains'])
    all_to_domains.extend(data['todomains'])
  return {
    'samples': all_training_samples,
    'labels': all_training_labels,
    'weights': all_training_weights,
    'fromdomains': all_from_domains,
    'todomains': all_to_domains,
  }

@jsonmemoized
def extract_tofill_dataset_for_training():
  return extract_tofill_dataset_for_users(training_users)

@jsonmemoized
def extract_tofill_dataset_for_test():
  return extract_tofill_dataset_for_users(test_users)

def train_tofill_classifier():
  training_data = extract_tofill_dataset_for_training()
  return train_classifier_on_data(training_data)

def train_classifier_on_data(training_data):
  #classifier = sklearn.naive_bayes.GaussianNB()
  #classifier = sklearn.svm.LinearSVC()
  #classifier = sklearn.linear_model.LogisticRegression(class_weight='balanced')
  #classifier = sklearn.ensemble.RandomForestClassifier() #(class_weight='balanced')
  #classifier = sklearn.tree.DecisionTreeClassifier(max_depth=1, class_weight='balanced')
  classifier = sklearn.tree.DecisionTreeClassifier(max_depth=1)
  #classifier.fit(numpy.array(training_data['samples']), numpy.array(training_data['labels']))
  classifier.fit(numpy.array(training_data['samples']), numpy.array(training_data['labels']), numpy.array(training_data['weights']))
  return classifier

def train_tofill_classifier_v2():
  training_data = extract_tofill_dataset_for_training()
  def train_classifier_if_enough_data(key_to_data):
    key_to_classifier = {}
    for key,specialized_data in key_to_data.items():
      positive_samples = specialized_data['labels'].count(1)
      negative_samples = specialized_data['labels'].count(0)
      if min(positive_samples, negative_samples) > 100: # enough to bother with a specialized classifier
        key_to_classifier[key] = train_classifier_on_data(specialized_data)
    return key_to_classifier
  def extract_data_for_function(key_compute_function):
    key_to_data = {}
    for idx,sample in enumerate(training_data['samples']):
      fromdomain = training_data['fromdomains'][idx]
      todomain = training_data['todomains'][idx]
      label = training_data['labels'][idx]
      weight = training_data['weights'][idx]
      key = key_compute_function(fromdomain, todomain)
      if key not in key_to_data:
        key_to_data[key] = {
          'samples': [],
          'labels': [],
          'weights': [],
        }
      key_to_data[key]['samples'].append(sample)
      key_to_data[key]['labels'].append(label)
      key_to_data[key]['weights'].append(weight)
    return key_to_data
  def train_classifier_for_function(key_compute_function):
    return train_classifier_if_enough_data(extract_data_for_function(key_compute_function))
  general_classifier = train_classifier_on_data(training_data)
  fromdomain_to_classifier = train_classifier_for_function(lambda f,t: f)
  todomain_to_classifier = train_classifier_for_function(lambda f,t: t)
  fromdomain_and_todomain_to_classifier = train_classifier_for_function(lambda f,t: f + ' to ' + t)
  same_fromdomain_and_todomain_to_classifier = train_classifier_for_function(lambda f,t: f == t)
  def fromdomain_same_compute_key(f, t):
    if f == t:
      return f + ' to same'
    return f + ' to diff'
  def todomain_same_compute_key(f, t):
    if f == t:
      return t + ' from same'
    return t + ' from diff'
  fromdomain_same_to_classifier = train_classifier_for_function(fromdomain_same_compute_key)
  todomain_same_to_classifier = train_classifier_for_function(todomain_same_compute_key)
  print 'fromdomain_and_todomain_to_classifier', len(fromdomain_and_todomain_to_classifier)
  print 'todomain_to_classifier', len(todomain_to_classifier)
  print 'fromdomain_to_classifier', len(fromdomain_to_classifier)
  print 'same_fromdomain_and_todomain_to_classifier', len(same_fromdomain_and_todomain_to_classifier)
  print 'fromdomain_same_to_classifier', len(fromdomain_same_to_classifier)
  print 'todomain_same_to_classifier', len(todomain_same_to_classifier)
  def classify_logtime_fromdomain_todomain(logtime, fromdomain, todomain):
    fromdomain_and_todomain = fromdomain + ' to ' + todomain
    same_fromdomain_and_todomain = fromdomain == todomain
    fromdomain_same = fromdomain_same_compute_key(fromdomain, todomain)
    todomain_same = todomain_same_compute_key(fromdomain, todomain)
    if fromdomain_and_todomain in fromdomain_and_todomain_to_classifier:
      cur_classifier = fromdomain_and_todomain_to_classifier[fromdomain_and_todomain]
    elif fromdomain_same in fromdomain_to_classifier:
      cur_classifier = fromdomain_same_to_classifier[fromdomain_same]
    elif fromdomain in fromdomain_to_classifier:
      cur_classifier = fromdomain_to_classifier[fromdomain]
    elif todomain_same in todomain_same_to_classifier:
      cur_classifier = todomain_same_to_classifier[todomain_same]
    elif todomain in todomain_to_classifier:
      cur_classifier = todomain_to_classifier[todomain]
    elif same_fromdomain_and_todomain in same_fromdomain_and_todomain_to_classifier:
      cur_classifier = same_fromdomain_and_todomain_to_classifier[same_fromdomain_and_todomain]
    else:
      cur_classifier = general_classifier
    #return cur_classifier.predict([[logtime]])
    threshold = cur_classifier.tree_.threshold[0]
    return logtime < threshold
  return classify_logtime_fromdomain_todomain
    


'''
false_samples = []
training_data = extract_tofill_dataset_for_training()
for idx,label in enumerate(training_data['labels']):
  sample = training_data['samples'][idx]
  if label == False:
    false_samples.append(sample[0])

print numpy.histogram(false_samples)
'''


#print extract_tofill_dataset_from_user(user)['labels']
#print len(extract_tofill_dataset_for_training()['labels'])


def make_predictions_with_classifier_on_dataset(cur_classifier, dataset):
  output = []
  for idx,sample in enumerate(dataset['samples']):
    fromdomain = dataset['fromdomains'][idx]
    todomain = dataset['todomains'][idx]
    output.append(cur_classifier(sample[0], fromdomain, todomain))
  return output


classifier = train_tofill_classifier_v2()
#classifier = train_tofill_classifier()


test_data = extract_tofill_dataset_for_test()
test_predictions = make_predictions_with_classifier_on_dataset(classifier, test_data)
#test_predictions = classifier.predict(test_data['samples'])
print sklearn.metrics.classification_report(test_data['labels'], test_predictions)





def get_code(tree, feature_names=['a', 'b', 'c', 'd', 'e', 'f']):
  left      = tree.tree_.children_left
  right     = tree.tree_.children_right
  threshold = tree.tree_.threshold
  features  = [feature_names[i] for i in tree.tree_.feature]
  value = tree.tree_.value

  def recurse(left, right, threshold, features, node):
    if (threshold[node] != -2):
      print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
      if left[node] != -1:
        recurse (left, right, threshold, features,left[node])
      print "} else {"
      if right[node] != -1:
        recurse (left, right, threshold, features,right[node])
      print "}"
    else:
      print "return " + str(value[node])

  recurse(left, right, threshold, features, 0)

#get_code(classifier)


#sklearn.tree.export_graphviz(classifier, out_file='classifier.dot', feature_names=['a', 'b', 'c'])
#os.system('dot -Tpng classifier.dot -o classifier.png')
#from IPython.core.display import Image
#Image('classifier.png')


#print classifier.predict([[14]])
#print classifier.predict([[13]])
#classifier.predict([[log(6*60*1000.0)]])


def merge_contiguous_spans(visit_spans):
  output = []
  merged = {}
  for span in visit_spans:
    if 'url' not in merged:
      merged = {k:v for k,v in span.items()}
      continue
    if merged['url'] == span['url']: # merge this current span into the merged one
      if span['start'] <= merged['end']:
        merged['end'] = max(merged['end'], span['end'])
        merged['active'] = max(merged['active'], span['active'])
        continue
      else: # end of current merged segment, start of new one
        output.append(merged)
        merged = {k:v for k,v in span.items()}
    else: # end of current merged segment, start of new one
      output.append(merged)
      merged = {k:v for k,v in span.items()}
  if 'url' in merged:
    output.append(merged)
  return output

#print merge_contiguous_spans([{'url': 'a', 'start': 0, 'end': 2}, {'url': 'a', 'start': 5, 'end': 7}, {'url': 'b', 'start': 10, 'end': 13}])
#print merge_contiguous_spans([{'url': 'a', 'start': 0, 'end': 2}, {'url': 'a', 'start': 2, 'end': 7}, {'url': 'b', 'start': 10, 'end': 13}])


def reconstruct_for_user_v2(user):
  ordered_visits = get_history_ordered_visits_for_user(user)
  ordered_visits = exclude_bad_visits(ordered_visits)
  output = []
  ordered_visits_len = len(ordered_visits)
  for idx,visit in enumerate(ordered_visits):
    if idx+1 == ordered_visits_len: # last visit, TODO needs to be reconstructed
      continue
    next_visit = ordered_visits[idx+1]
    visit_time = visit['visitTime']
    next_visit_time = next_visit['visitTime']
    url = visit['url']
    next_url = next_visit['url']
    time_difference = next_visit_time - visit_time
    if time_difference <= 0:
      continue
    log_time_difference = log(time_difference)
    #extend_to_next_visit = classifier.predict([[log_time_difference]])[0]
    #extend_to_next_visit = log_time_difference < 13.5336971283
    #extend_to_next_visit = log_time_difference < classifier.tree_.threshold[0]
    fromdomain = url_to_domain(url)
    todomain = url_to_domain(next_url)
    extend_to_next_visit = classifier(log_time_difference, fromdomain, todomain)
    end_time = min(visit_time + 3.0*60*1000.0, next_visit_time)
    if extend_to_next_visit:
      end_time = next_visit_time
    output.append({'url': url, 'next_url': next_url, 'start': visit_time, 'active': visit_time, 'end': end_time})

  output = merge_contiguous_spans(output)
  return output



def reconstruct_for_user_perfect_predictions(user):
  ordered_visits = get_history_ordered_visits_for_user(user)
  ordered_visits = exclude_bad_visits(ordered_visits)
  tab_focus_times = get_tab_focus_times_for_user(user)
  tab_focus_times_sortedcollection = SortedCollection(tab_focus_times, key=itemgetter('start'))
  output = []
  ordered_visits_len = len(ordered_visits)
  for idx,visit in enumerate(ordered_visits):
    if idx+1 == ordered_visits_len: # last visit, TODO needs to be reconstructed
      continue
    next_visit = ordered_visits[idx+1]
    visit_time = visit['visitTime']
    next_visit_time = next_visit['visitTime']
    url = visit['url']
    next_url = next_visit['url']
    time_difference = next_visit_time - visit_time
    if time_difference <= 0:
      continue
    fraction_active = fraction_active_between_times(tab_focus_times_sortedcollection, visit_time, next_visit_time)
    end_time = min(visit_time + 60*1000.0, next_visit_time)
    if fraction_active > 0.5:
      end_time = next_visit_time
    output.append({'url': url, 'start': visit_time, 'active': visit_time, 'end': end_time})

  output = merge_contiguous_spans(output)
  return output



'''
def find_end_of_matching_span(tab_focus_times_sortedcollection, url, visit_time):
  try:
    item = tab_focus_times_sortedcollection.find_le(visit_time)
    if item['end'] >= visit_time:
      return item['end']
    item = tab_focus_times_sortedcollection.find_ge(visit_time)
    if item['end'] >= visit_time:
      return item['end']
  except:
    return None
'''

def find_best_matching_span(tab_focus_times_sortedcollection, url, visit_time):
  try:
    item = tab_focus_times_sortedcollection.find_le(visit_time)
    idx = tab_focus_times_sortedcollection.index(item)
  except:
    return None
  best_matching_span = None
  while idx < len(tab_focus_times_sortedcollection):
    span = tab_focus_times_sortedcollection[idx]
    if span['url'] == url and span['start'] <= visit_time+5000 and span['end'] > visit_time:
      return span
    idx += 1

def find_end_of_best_matching_span(tab_focus_times_sortedcollection, url, visit_time):
  span = find_best_matching_span(tab_focus_times_sortedcollection, url, visit_time)
  if span == None:
    return None
  return span['end']


# this one sadly does not appear to work correctly
def reconstruct_for_user_perfect_predictions_v2(user):
  ordered_visits = get_history_ordered_visits_for_user(user)
  ordered_visits = exclude_bad_visits(ordered_visits)
  tab_focus_times = get_tab_focus_times_for_user(user)
  tab_focus_times_sortedcollection = SortedCollection(tab_focus_times, key=itemgetter('start'))
  output = []
  ordered_visits_len = len(ordered_visits)
  for idx,visit in enumerate(ordered_visits):
    if idx+1 == ordered_visits_len: # last visit, TODO needs to be reconstructed
      continue
    next_visit = ordered_visits[idx+1]
    visit_time = visit['visitTime']
    next_visit_time = next_visit['visitTime']
    url = visit['url']
    next_url = next_visit['url']
    time_difference = next_visit_time - visit_time
    if time_difference <= 0:
      continue
    end_time_proposed = find_end_of_best_matching_span(tab_focus_times_sortedcollection, url, visit_time)
    if end_time_proposed != None:
      end_time = end_time_proposed
    else:
      fraction_active = fraction_active_between_times(tab_focus_times_sortedcollection, visit_time, next_visit_time)
      end_time = min(visit_time + 60*1000.0, next_visit_time)
      if fraction_active > 0.5:
        end_time = next_visit_time
    output.append({'url': url, 'start': visit_time, 'active': visit_time, 'end': end_time})

  output = merge_contiguous_spans(output)
  return output



def reconstruct_for_user_naive(minutes_threshold):
  def returned_function(user):
    ordered_visits = get_history_ordered_visits_for_user(user)
    ordered_visits = exclude_bad_visits(ordered_visits)
    output = []
    ordered_visits_len = len(ordered_visits)
    for idx,visit in enumerate(ordered_visits):
      if idx+1 == ordered_visits_len: # last visit, TODO needs to be reconstructed
        continue
      next_visit = ordered_visits[idx+1]
      visit_time = visit['visitTime']
      next_visit_time = next_visit['visitTime']
      url = visit['url']
      next_url = next_visit['url']
      time_difference = next_visit_time - visit_time
      if time_difference <= 0:
        continue
      #log_time_difference = log(time_difference)
      #extend_to_next_visit = classifier.predict([[log_time_difference]])[0]
      #extend_to_next_visit = log_time_difference < 13.5336971283
      #extend_to_next_visit = log_time_difference < classifier.tree_.threshold[0]
      #fromdomain = url_to_domain(url)
      #todomain = url_to_domain(next_url)
      #extend_to_next_visit = classifier(log_time_difference, fromdomain, todomain)
      end_time = min(visit_time + minutes_threshold*60*1000.0, next_visit_time)
      #if extend_to_next_visit:
      #  end_time = next_visit_time
      output.append({'url': url, 'start': visit_time, 'active': visit_time, 'end': end_time})

    output = merge_contiguous_spans(output)
    return output
  return returned_function



def reconstruct_for_user_v3(user):
  #ordered_visits = get_history_ordered_visits_for_user(user)
  #ordered_visits = exclude_bad_visits(ordered_visits)
  ordered_visits = get_idealized_history_from_logs_urlchanged_for_user(user)
  #ordered_visits = get_idealized_history_from_logs_for_user(user)
  output = []
  ordered_visits_len = len(ordered_visits)
  for idx,visit in enumerate(ordered_visits):
    if idx+1 == ordered_visits_len: # last visit, TODO needs to be reconstructed
      continue
    next_visit = ordered_visits[idx+1]
    visit_time = visit['visitTime']
    next_visit_time = next_visit['visitTime']
    url = visit['url']
    next_url = next_visit['url']
    time_difference = next_visit_time - visit_time
    if time_difference <= 0:
      continue
    log_time_difference = log(time_difference)
    #extend_to_next_visit = classifier.predict([[log_time_difference]])[0]
    extend_to_next_visit = log_time_difference < 13.5336971283
    #extend_to_next_visit = log_time_difference < classifier.tree_.threshold[0]
    end_time = min(visit_time + 60*1000.0, next_visit_time)
    if extend_to_next_visit:
      end_time = next_visit_time
    output.append({'url': url, 'start': visit_time, 'active': visit_time, 'end': end_time})

  output = merge_contiguous_spans(output)
  return output



def evaluate_reconstruction_algorithm_for_user(user, reconstruction_algorithm):
  #user = '3a3FX1s9S6'
  reconstructed_tab_focus_times = reconstruction_algorithm(user)
  tab_focus_times = get_tab_focus_times_for_user(user)
  #reconstructed_tab_focus_times = tab_focus_times = get_tab_focus_times_for_user(user)
  #reconstructed_tab_focus_times = list(get_reconstruct_focus_times_baseline_for_user(user))
  return evalutate_tab_focus_reconstruction_fast(tab_focus_times, reconstructed_tab_focus_times)

  #ref_start_time = max(get_earliest_start_time(tab_focus_times), get_earliest_start_time(reconstructed_tab_focus_times))
  #ref_end_time = min(get_last_end_time(tab_focus_times), get_last_end_time(reconstructed_tab_focus_times))
  #evaluated_reconstructed_tab_focus_times = ignore_all_before_start_or_after_end(reconstructed_tab_focus_times, ref_start_time, ref_end_time)
  #evaluated_tab_focus_times = ignore_all_before_start_or_after_end(tab_focus_times, ref_start_time, ref_end_time)

  #return evalutate_tab_focus_reconstruction(evaluated_tab_focus_times, evaluated_reconstructed_tab_focus_times)








def evaluate_reconstruction_algorithm(reconstruction_algorithm):
  overall_evaluation_results = Counter()
  #for user in test_users:
  for user in training_users[:10]:
    evaluation_results = evaluate_reconstruction_algorithm_for_user(user, reconstruction_algorithm)
    for k,v in evaluation_results.items():
      overall_evaluation_results[k] += v
  return overall_evaluation_results

def evaluate_reconstruction_algorithm_test(reconstruction_algorithm):
  overall_evaluation_results = Counter()
  for user in test_users:
    evaluation_results = evaluate_reconstruction_algorithm_for_user(user, reconstruction_algorithm)
    for k,v in evaluation_results.items():
      overall_evaluation_results[k] += v
  return overall_evaluation_results

def evaluate_reconstruction_algorithm_train(reconstruction_algorithm):
  overall_evaluation_results = Counter()
  for user in training_users:
    evaluation_results = evaluate_reconstruction_algorithm_for_user(user, reconstruction_algorithm)
    for k,v in evaluation_results.items():
      overall_evaluation_results[k] += v
  return overall_evaluation_results


def sumfields(d, *args):
  return sum(d.get(x, 0) for x in args)

def print_evaluation_results(results):
  ref_active_time = float(sumfields(results, 'correct_url', 'ref_active_but_rec_inactive', 'incorrect_domain', 'correct_domain', 'incorrect_domain_next_url_is_none', 'incorrect_domain_ref_equals_next_domain', 'incorrect_domain_other', 'nexturl_correct', 'nextdomain_correct'))
  correct_span = float(sumfields(results, 'correct_url', 'incorrect_domain', 'correct_domain', 'incorrect_domain_next_url_is_none', 'incorrect_domain_ref_equals_next_domain', 'incorrect_domain_other', 'nexturl_correct', 'nextdomain_correct'))
  print 'correct span', correct_span, 'which is', correct_span/ref_active_time, 'of ref_active_time'
  correct_url = results['correct_url']
  ref_inactive_but_rec_active = results['ref_inactive_but_rec_active']
  ref_active_but_rec_inactive = results['ref_active_but_rec_inactive']
  precision = correct_span/(correct_span + ref_inactive_but_rec_active)
  recall = correct_span/(correct_span + ref_active_but_rec_inactive)
  f1 = 2*(precision*recall)/(precision+recall)
  tp = correct_span
  tn = results['both_inactive']
  fp = results['ref_inactive_but_rec_active']
  fn = results['ref_active_but_rec_inactive']
  sum4 = float(sum([tp, tn, fp, fn]))
  print '=== span reconstruction evaluation ==='
  print 'precision', precision
  print 'recall', recall
  print 'f1', f1
  print 'tp', tp/sum4
  print 'tn', tn/sum4
  print 'fp', fp/sum4
  print 'fn', fn/sum4
  #print 'fp ref_inactive_but_rec_active', ref_inactive_but_rec_active/ref_active_time, 'of ref_active_time'
  print '=== url reconstruction evaluation ==='
  print 'correct url', correct_url/ref_active_time, 'of ref_active_time', correct_url/correct_span, 'of correct_span'
  nexturl_correct = results.get('nexturl_correct', 0)
  correct_domain = results['correct_domain']
  nextdomain_correct = results.get('nextdomain_correct', 0)
  incorrect_domain = float(sumfields(results, 'incorrect_domain', 'incorrect_domain_next_url_is_none', 'incorrect_domain_ref_equals_next_domain', 'incorrect_domain_other'))
  print 'nexturl_correct', nexturl_correct/ref_active_time, 'of ref_active_time', nexturl_correct/correct_span, 'of correct span'
  print 'correct_domain', correct_domain/ref_active_time, 'of ref_active_time', correct_domain/correct_span, 'of correct span'
  print 'nextdomain_correct', nextdomain_correct/ref_active_time, 'of ref_active_time', nextdomain_correct/correct_span, 'of correct span'
  print 'incorrect_domain', incorrect_domain/ref_active_time, 'of ref_active_time', incorrect_domain/correct_span, 'of correct span'
  print results






#print_evaluation_results(evaluate_reconstruction_algorithm_for_user('3a3FX1s9S6', reconstruct_for_user_naive))


#print_evaluation_results(evaluate_reconstruction_algorithm(reconstruct_for_user_naive))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(0.5)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(1.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(2.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(3.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(4.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(5.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(6.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(7.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(8.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(9.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_naive(10.0)))


#print_evaluation_results(evaluate_reconstruction_algorithm_for_user('3a3FX1s9S6', reconstruct_for_user_perfect_predictions_v2))








#print_evaluation_results(evaluate_reconstruction_algorithm_for_user('3a3FX1s9S6', reconstruct_for_user_perfect_predictions))


#print_evaluation_results(evaluate_reconstruction_algorithm(reconstruct_for_user_perfect_predictions))


print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_perfect_predictions))


print_evaluation_results(evaluate_reconstruction_algorithm_for_user('3a3FX1s9S6', reconstruct_for_user_v2))


print_evaluation_results(evaluate_reconstruction_algorithm(reconstruct_for_user_v2))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_v2))


print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_v2))


print_evaluation_results(evaluate_reconstruction_algorithm_train(reconstruct_for_user_v2))


#print_evaluation_results(evaluate_reconstruction_algorithm_for_user('3a3FX1s9S6', reconstruct_for_user_v3))


#print_evaluation_results(evaluate_reconstruction_algorithm(reconstruct_for_user_v3))


#print_evaluation_results(evaluate_reconstruction_algorithm_test(reconstruct_for_user_v3))


'''
max_weight = 0
samples_for_max_weight = []
num_printed = 0
for idx in range(len(test_predictions)):
  ref = test_data['labels'][idx]
  pred = test_predictions[idx]
  weight = test_data['weights'][idx]
  samples = test_data['samples'][idx]
  if ref == 0 and pred == 1:
    #max_weight = max(max_weight, weight)
    if weight > max_weight:
      max_weight = weight
      samples_for_max_weight = samples
    #print weight
    #print samples
    num_printed += 1
    #if num_printed >= 100:
    #  break
  
print max_weight
print samples_for_max_weight
'''


#print classifier.predict([[log(60*1000)]])


#print classifier.predict([[19.010298647]])
#print classifier.predict([[15.010298647]])





#a = [{'time': x} for x in [5,3,7,9,2]]
#b = SortedCollection(a, key=itemgetter('time'))
#print b.find_le(6)


#tab_focus_times = get_tab_focus_times_for_user(user)
#reconstructed_tab_focus_times = tab_focus_times = get_tab_focus_times_for_user(user)
#reconstructed_tab_focus_times = list(get_reconstruct_focus_times_baseline_for_user(user))
#print evalutate_tab_focus_reconstruction(tab_focus_times, reconstructed_tab_focus_times)

#for visit in get_history_ordered_visits_for_user(user):
#  print visit

