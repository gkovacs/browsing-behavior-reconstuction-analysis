#!/usr/bin/env python
# md5: 3228f7a5c1b954d0f52fee67b0623534
# coding: utf-8

from pymongo import MongoClient
import yaml
from memoized import memoized
from jsonmemoized import jsonmemoized


@memoized
def get_secrets():
  return yaml.load(open('.getsecret.yaml'))

@memoized
def get_mongoclient():
  return MongoClient(get_secrets()['mongourl'])

@memoized
def get_rescuetime_mongodb():
  return get_mongoclient()[get_secrets()['mongodb']]

@memoized
def get_category_mongodb_collection():
  return get_rescuetime_mongodb().category

@memoized
def get_productivity_mongodb_collection():
  return get_rescuetime_mongodb().productivity

@jsonmemoized
def get_domain_to_category_real():
  output = {}
  for x in get_category_mongodb_collection().find():
    domain = x['_id']
    category = x['val']
    output[domain] = category
  return output

@memoized
def get_domain_to_category():
  return get_domain_to_category_real()

def domain_to_category(domain):
  return get_domain_to_category().get(domain, None)

@jsonmemoized
def get_domain_to_productivity_real():
  output = {}
  for x in get_productivity_mongodb_collection().find():
    domain = x['_id']
    productivity = x['val']
    output[domain] = productivity
  return output

@memoized
def get_domain_to_productivity():
  return get_domain_to_productivity_real()

def domain_to_productivity(domain):
  return get_domain_to_productivity().get(domain, 0)

@jsonmemoized
def get_rescuetime_productivity_levels_real():
  output = set()
  for x in get_productivity_mongodb_collection().find():
    domain = x['_id']
    productivity = x['val']
    output.add(productivity)
  return sorted(list(output))

@memoized
def get_rescuetime_productivity_levels():
  return get_rescuetime_productivity_levels_real()

@jsonmemoized
def get_rescuetime_categories_real():
  output = set()
  for x in get_category_mongodb_collection().find():
    domain = x['_id']
    category = x['val']
    output.add(category)
  return sorted(list(output))

@memoized
def get_rescuetime_categories():
  return get_rescuetime_categories_real()


#print get_rescuetime_categories()
#print get_rescuetime_productivity_levels()


#print domain_to_productivity('mturk.com')
#print domain_to_category('mturk.com')

