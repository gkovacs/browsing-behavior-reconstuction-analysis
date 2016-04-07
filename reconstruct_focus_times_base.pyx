#!/usr/bin/env python
# md5: dd00934d241cef385d9c0a04e0d0bd20
# coding: utf-8

from tmilib_base import *


def exclude_bad_visits(ordered_visits):
  output = []
  for visit in ordered_visits:
    transition = visit['transition']
    if transition in ['auto_subframe', 'manual_subframe']:
      continue
    output.append(visit)
  return output


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

