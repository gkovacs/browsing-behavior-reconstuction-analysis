#!/usr/bin/env python
# md5: 386f7cb008e6b543b9652949497bbdbd
# coding: utf-8

import ujson as json
import subprocess
from glob import glob

from decompress_lzstring import decompressFromBase64

from tmilib import *

#print subprocess.check_output(['free', '-h'])

#basedir = sorted(glob('/home/gkovacs/tmi-data/local_*'), reverse=True)[0]
basedir = get_basedir()

#print basedir


history_files = glob(basedir + '/hist_*.json')
#print history_files


username_to_mturk_id = json.load(open('username_to_mturk_id.json'))

mturkid_to_history_pages = {}
mturkid_to_history_visits = {}
mturkid_to_hid = {}


for filename in history_files:
  print filename
  all_lines = json.load(open(filename))
  for line in reversed(all_lines):
    evt = line['evt']
    hid = line['hid']
    user = line.get('user', None)
    mturkid = line.get('mturkid', None)
    if mturkid == None:
      if user == None:
        continue
      mturkid = username_to_mturk_id.get(user, None)
      if mturkid == None:
        continue
    orig_hid = mturkid_to_hid.get(mturkid, 0)
    if orig_hid > hid:
      continue
    if hid > orig_hid:
      mturkid_to_hid[mturkid] = hid
      mturkid_to_history_pages[mturkid] = []
      mturkid_to_history_visits[mturkid] = {}
    data = json.loads(decompressFromBase64(line['data']))
    if evt == 'history_pages':
      mturkid_to_history_pages[mturkid] = data
    if evt == 'history_visits':
      for k,v in data.items():
        mturkid_to_history_visits[mturkid][k] = v

json.dump(mturkid_to_history_pages, open('mturkid_to_history_pages.json', 'w'))
json.dump(mturkid_to_history_visits, open('mturkid_to_history_visits.json', 'w'))


#print list(reversed([0,1,2]))




