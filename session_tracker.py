#!/usr/bin/env python
# md5: d807e1bea8d9d555368ee87e1984c639
# coding: utf-8

def get_focused_tab(data):
  if data['evt'] == 'mlog':
    return data['url']
  windows = data['windows']
  for window in windows:
    focused = window['focused']
    if not focused:
      continue
    tabs = window['tabs']
    for tab in tabs:
      if not tab['highlighted']:
        continue
      if not tab['selected']:
        continue
      return tab['url']

class SessionTracker:
  def __init__(self, **kwargs):
    self.output = []
    self.curitem = {}
    self.store_events = kwargs.get('store_events', False)
  def get_output(self):
    self.end_input()
    return self.output
  def end_input(self):
    if 'url' in self.curitem:
      last_active = self.curitem['active']
      self.curitem['end'] = last_active + 60*1000
      self.output.append(self.curitem)
      self.curitem = {}
  def end_session(self, curtime):
    if 'url' in self.curitem:
      last_active = self.curitem['active']
      # ensures that end < last_active+30 minutes=1800 secs (we would normally expect that an idle event would have caught this)
      self.curitem['end'] = min(curtime, last_active + 60*1000)
      self.output.append(self.curitem)
      self.curitem = {}
  def start_session(self, url, curtime):
    if url == None:
      raise Exception('start_session should not be called with url==None')
    self.end_session(curtime)
    # start: first event in the session
    # active: last event which was active in the session
    # end: when we believe the session ended
    self.curitem = {'url': url, 'start': curtime, 'active': curtime}
  def continue_session(self, url, curtime):
    if url == None:
      raise Exception('continue_session should not be called with url==None')
    if 'url' not in self.curitem:
      self.start_session(url, curtime)
      return
    prevurl = self.curitem['url']
    if url == prevurl: # still on same site
      # has it been less than 60 seconds since last activity?
      prev_active_time = self.curitem['active']
      if curtime < prev_active_time + 60*1000:
        self.curitem['active'] = curtime
        return
    # have gone to different site
    self.end_session(curtime)
    self.start_session(url, curtime)
  def record_events(self, data):
    if self.store_events:
      if 'events' not in self.curitem:
        self.curitem['events'] = []
      cloned_event = {k:v for k,v in data.items() if k != 'windows'}
      cururl = get_focused_tab(data)
      cloned_event['url'] = cururl
      self.curitem['events'].append(cloned_event)
  def record_events_again_if_new(self, data):
    if self.store_events:
      if 'events' not in self.curitem:
        self.curitem['events'] = []
        # only record again if is first one
        cloned_event = {k:v for k,v in data.items() if k != 'windows'}
        cururl = get_focused_tab(data)
        cloned_event['url'] = cururl
        cloned_event['dup'] = True
        self.curitem['events'].append(cloned_event)
  def process_input(self, data):
    evt = data['evt']
    curtime = data['time'] # this is timestamp in milliseconds
    cururl = get_focused_tab(data)
    self.record_events(data)
    if cururl == None: # browser is not focused
      self.end_session(curtime)
      self.record_events_again_if_new(data)
      return
    if evt == 'idle_changed':
      self.process_idle_changed(data)
      self.record_events_again_if_new(data)
      return
    if evt == 'still_browsing': # ignore still_browsing events
      return
    self.continue_session(cururl, curtime)
    self.record_events_again_if_new(data)
  def process_idle_changed(self, data):
    # idlestate can be either idle, locked, or active
    idlestate = data['idlestate']
    curtime = data['time']
    if idlestate == 'idle' or idlestate == 'locked':
      self.end_session(curtime)
      return
    if idlestate == 'active':
      cururl = get_focused_tab(data)
      self.start_session(cururl, curtime)



















