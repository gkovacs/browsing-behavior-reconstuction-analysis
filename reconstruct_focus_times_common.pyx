#!/usr/bin/env python
# md5: f585c2d66b2e51e433ab1ad9e4a8b2d9
# coding: utf-8

from tmilib import *


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


def get_earliest_start_time(visits):
  if len(visits) < 1:
    raise Exception('get_earliest_time called with empty list')
  first_visit = visits[0]
  if 'start' in first_visit:
    return first_visit['start']
  return first_visit['visitTime']

def get_last_end_time(visits):
  if len(visits) < 1:
    raise Exception('get_last_visit_time called with empty list')
  last_visit = visits[-1]
  if 'end' in last_visit:
    return last_visit['end']
  return last_visit['visitTime']


def evaluation_stats_for_reconstruction(result_reference, result_reconstructed):
  stats = Counter()
  for ref_answer,rec_answer in zip(result_reference.get_output(), result_reconstructed.get_output()):
    if ref_answer == 0 and rec_answer == 0:
      stats['both_inactive'] += 1
    elif ref_answer == 0 and rec_answer != 0:
      stats['ref_inactive_but_rec_active'] += 1
    elif ref_answer != 0 and rec_answer == 0:
      stats['ref_active_but_rec_inactive'] += 1
    elif ref_answer != 0 and rec_answer != 0:
      ref_url = result_reference.get_url_for_index(ref_answer)
      rec_url = result_reconstructed.get_url_for_index(rec_answer)
      ref_domain = url_to_domain(ref_url)
      rec_domain = url_to_domain(rec_url)
      if ref_domain == rec_domain:
        if ref_url == rec_url:
          stats['correct_url'] += 1
        else:
          stats['correct_domain'] += 1
      else:
        stats['incorrect_domain'] += 1
  return stats


class UrlAsIndex:
  def __init__(self, offset=0):
    self.index_to_url = []
    self.url_to_index = {}
    for idx in range(offset):
      self.index_to_url.append(None)
  def get_index_for_url(self, url):
    if url in self.url_to_index:
      return self.url_to_index[url]
    index = len(self.index_to_url)
    self.index_to_url.append(url)
    self.url_to_index[index] = url
    return index
  def get_url_for_index(self, index):
    return self.index_to_url[index]

class UrlAtTime:
  def __init__(self, visits, start_time=None, end_time=None):
    if start_time == None:
      start_time = get_earliest_start_time(visits)
    self.offset = start_time # milliseconds
    if end_time == None:
      end_time = get_last_end_time(visits)
    end_time_minus_offset = (end_time - start_time) / 1000.0
    #self.output = [None]*int(round(end_time_minus_offset) + 1)
    self.output = numpy.zeros(int(round(end_time_minus_offset) + 1), dtype=int)
    self.url_as_index = UrlAsIndex(1)
    for visit in visits:
      self.process_visit(visit)
  def process_visit(self, visit):
    url = visit['url']
    url_idx = self.url_as_index.get_index_for_url(url)
    start = visit['start']
    end = visit['end']
    start_idx = int(round((start - self.offset) / 1000.0))
    if start_idx < 0:
      start_idx = 0
      #return
    if start_idx >= len(self.output):
      return
    end_idx = int(round((end - self.offset) / 1000.0))
    if end_idx < 0:
      return
    if end_idx >= len(self.output):
      end_idx = len(self.output) - 1
    for idx in range(start_idx, end_idx+1):
      self.output[idx] = url_idx
  def get_url_for_index(self, index):
    return self.url_as_index.get_url_for_index(index)
  def get_output(self):
    return self.output

def second_to_active_url(visits, start_time, end_time):
  # offset: millseconds
  #start_time = get_earliest_start_time(visits)
  #end_time = get_last_end_time(visits)
  url_at_time = UrlAtTime(visits, start_time, end_time)
  return url_at_time


def add_empty_spans(spans):
  output = []
  spans_len = len(spans)
  if spans_len == 0:
    return output
  for idx,span in enumerate(spans):
    if idx+1 == spans_len:
      continue
    next_span = spans[idx+1]
    output.append(span)
    if next_span['start'] > span['end']: # need to insert an empty span
      output.append({
        'start': span['end'],
        'active': span['end'],
        'url': None,
        'end': next_span['start'],
      })
  output.append(spans[-1])
  return output

def extend_empty_spans_to_cover_time(spans, start, end):
  if len(spans) == 0:
    spans.append({
      'start': start,
      'active': start,
      'url': None,
      'end': end,
    })
  if spans[0]['start'] > start:
    spans.insert(0, {
      'start': start,
      'active': start,
      'url': None,
      'end': spans[0]['start'],
    })
  if spans[-1]['end'] < end:
    spans.append({
      'start': spans[-1]['end'],
      'active': spans[-1]['end'],
      'url': None,
      'end': end,
    })
  return spans

def restrict_spans_to_time(spans, start, end):
  output = []
  # discard items that occur before, and shorten the first item so that it starts at the start time
  # discard items that occur after, and shorten the final item so that it ends at the end time
  for span in spans:
    if span['end'] < start:
      continue
    if span['start'] > end:
      continue
    if span['start'] < start:
      span = span.copy()
      span['start'] = start
    if span['end'] > end:
      span = span.copy()
      span['end'] = end
    if span['start'] >= span['end']: # span has no duration
      continue
    # note that this does not adjust the value of active
    output.append(span)
  return output

def add_empty_spans_and_restrict_to_time(spans, start, end):
  spans = restrict_spans_to_time(spans, start, end)
  spans = extend_empty_spans_to_cover_time(spans, start, end)
  return add_empty_spans(spans)


#print add_empty_spans_and_restrict_to_time([{'start': 0, 'active': 0, 'end': 2}, {'start': 5, 'active': 0, 'end': 7}, {'start': 8, 'active': 8, 'end': 10}], 2, 8)
#add_empty_spans([{'start': 0, 'end': 2}, {'start': 5, 'end': 7}])


def add_block_segment_to_stats(stats, ref_block, rec_block, start, end):
  ref_url = ref_block['url']
  rec_url = rec_block['url']
  duration = end - start
  if ref_url == None and rec_url == None:
    stats['both_inactive'] += duration
    return
  if ref_url == rec_url:
    stats['correct_url'] += duration
    return
  if ref_url == None and rec_url != None:
    stats['ref_inactive_but_rec_active'] += duration
    return
  if ref_url != None and rec_url == None:
    stats['ref_active_but_rec_inactive'] += duration
    return
  ref_domain = url_to_domain(ref_url)
  rec_domain = url_to_domain(rec_url)
  if ref_domain == rec_domain:
    stats['correct_domain'] += duration
    return
  stats['incorrect_domain'] += duration

def evalutate_tab_focus_reconstruction_fast(evaluated_tab_focus_times, evaluated_reconstructed_tab_focus_times):
  if len(evaluated_reconstructed_tab_focus_times) == 0 or len(evaluated_tab_focus_times) == 0:
    return {}
  ref_start_time = max(get_earliest_start_time(evaluated_tab_focus_times), get_earliest_start_time(evaluated_reconstructed_tab_focus_times))
  ref_end_time = min(get_last_end_time(evaluated_tab_focus_times), get_last_end_time(evaluated_reconstructed_tab_focus_times))
  reference = add_empty_spans_and_restrict_to_time(evaluated_tab_focus_times, ref_start_time, ref_end_time)
  reconstructed = add_empty_spans_and_restrict_to_time(evaluated_reconstructed_tab_focus_times, ref_start_time, ref_end_time)
  ref_idx = 0
  rec_idx = 0
  stats = Counter()
  cur_time = ref_start_time
  while cur_time < ref_end_time:
    ref_block = reference[ref_idx]
    rec_block = reconstructed[rec_idx]
    if ref_block['end'] == rec_block['end']:
      ref_idx += 1
      rec_idx += 1
      cur_end = ref_block['end']
    elif ref_block['end'] < rec_block['end']:
      ref_idx += 1
      cur_end = ref_block['end']
    #elif ref_block['end'] > rec_block['end']:
    else:
      rec_idx += 1
      cur_end = rec_block['end']
    add_block_segment_to_stats(stats, ref_block, rec_block, cur_time, cur_end)
    cur_time = cur_end
  return stats
    


def evalutate_tab_focus_reconstruction(evaluated_tab_focus_times, evaluated_reconstructed_tab_focus_times):
  if len(evaluated_reconstructed_tab_focus_times) == 0 or len(evaluated_tab_focus_times) == 0:
    return {}
  ref_start_time = max(get_earliest_start_time(evaluated_tab_focus_times), get_earliest_start_time(evaluated_reconstructed_tab_focus_times))
  ref_end_time = min(get_last_end_time(evaluated_tab_focus_times), get_last_end_time(evaluated_reconstructed_tab_focus_times))
  result_reference = second_to_active_url(evaluated_tab_focus_times, ref_start_time, ref_end_time)
  result_reconstructed = second_to_active_url(evaluated_reconstructed_tab_focus_times, ref_start_time, ref_end_time)
  return evaluation_stats_for_reconstruction(result_reference, result_reconstructed)


def ignore_all_before_start_or_after_end(visit_lengths, start_time, end_time):
  output = []
  for x in visit_lengths:
    if x['end'] < start_time:
      continue
    if x['start'] > end_time:
      continue
    #if x['start'] < start_time: # - 1000: # 1 second before or after
    #  continue
    #if x['end'] > end_time: # + 1000:
    #  continue
    output.append(x)
  return output

