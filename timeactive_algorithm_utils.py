#!/usr/bin/env python
# md5: 534dac664bada9466dc70acfff63608e
# coding: utf-8

from tmilib import *
import csv
from itertools import izip


@jsonmemoized
def get_user_to_predicted_times_active_our_algorithm():
  predictions_csv = csv.reader(sdir_open('catdata_test_insession_second_evaluation_predictions_datav4_modelv6.csv'))
  predictions_header = next(predictions_csv)
  print predictions_header

  test_data_csv = csv.reader(sdir_open('catdata_test_insession_second_evaluation_v4.csv'))
  test_data_header = next(test_data_csv)
  print test_data_header
  
  assert test_data_header[0] == 'user'
  assert predictions_header[0] == 'predict'
  
  output = {}
  
  for predictions_line,test_line in izip(predictions_csv, test_data_csv):
    predict = predictions_line[0] == 'T'
    if predict:
      user = test_line[0]
      time_sec = int(test_line[1])
      if user not in output:
        output[user] = []
      output[user].append(time_sec)
  for k in output.keys():
    output[k].sort()
  return output


@jsonmemoized
def get_user_to_predicted_times_active_baseline_algorithm():
  predictions_csv = csv.reader(sdir_open('catdata_test_insession_second_evaluation_predictions_datav4_modelv6.csv'))
  predictions_header = next(predictions_csv)
  print predictions_header

  test_data_csv = csv.reader(sdir_open('catdata_test_insession_second_evaluation_v4.csv'))
  test_data_header = next(test_data_csv)
  print test_data_header
  
  assert test_data_header[0] == 'user'
  assert predictions_header[0] == 'predict'
  
  log_fivemin = log(5*60)
  
  output = {}
  
  for predictions_line,test_line in izip(predictions_csv, test_data_csv):
    sinceprev = float(test_line[3])
    predict = sinceprev < log_fivemin
    if predict:
      user = test_line[0]
      time_sec = int(test_line[1])
      if user not in output:
        output[user] = []
      output[user].append(time_sec)
  for k in output.keys():
    output[k].sort()
  return output


@jsonmemoized
def get_user_to_predicted_times_active_baseline3_algorithm():
  predictions_csv = csv.reader(sdir_open('catdata_test_insession_second_evaluation_predictions_datav4_modelv6.csv'))
  predictions_header = next(predictions_csv)
  print predictions_header

  test_data_csv = csv.reader(sdir_open('catdata_test_insession_second_evaluation_v4.csv'))
  test_data_header = next(test_data_csv)
  print test_data_header
  
  assert test_data_header[0] == 'user'
  assert predictions_header[0] == 'predict'
  
  log_onemin = log(1*60)
  
  output = {}
  
  for predictions_line,test_line in izip(predictions_csv, test_data_csv):
    sinceprev = float(test_line[3])
    predict = sinceprev < log_onemin
    if predict:
      user = test_line[0]
      time_sec = int(test_line[1])
      if user not in output:
        output[user] = []
      output[user].append(time_sec)
  for k in output.keys():
    output[k].sort()
  return output


@jsonmemoized
def get_user_to_predicted_times_active_baseline2_algorithm():
  predictions_csv = csv.reader(sdir_open('catdata_test_insession_second_evaluation_predictions_datav4_modelv6.csv'))
  predictions_header = next(predictions_csv)
  print predictions_header

  test_data_csv = csv.reader(sdir_open('catdata_test_insession_second_evaluation_v4.csv'))
  test_data_header = next(test_data_csv)
  print test_data_header
  
  assert test_data_header[0] == 'user'
  assert predictions_header[0] == 'predict'
  
  log_onemin = log(1*60)
  
  output = {}
  
  user_to_is_active_in_majority_of_sessions = get_username_to_is_active_in_majority_of_sessions()
  
  for predictions_line,test_line in izip(predictions_csv, test_data_csv):
    user = test_line[0]
    predict = user_to_is_active_in_majority_of_sessions[user]
    if predict:
      time_sec = int(test_line[1])
      if user not in output:
        output[user] = []
      output[user].append(time_sec)
  for k in output.keys():
    output[k].sort()
  return output


a=get_user_to_predicted_times_active_baseline_algorithm()
a=get_user_to_predicted_times_active_baseline3_algorithm()
a=get_user_to_predicted_times_active_baseline2_algorithm()

