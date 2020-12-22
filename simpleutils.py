# every utils that don't use torch
import hashlib
import time
import json

class Timing():
    def __init__(self, name='run time'):
        self.name = name
        self.t = time.time()
        self.entered = False
    def __enter__(self):
        self.t = time.time()
        self.entered = True
    def __exit__(self, *ignored):
        showRunTime(self.name)
    def showRunTime(self, name):
        print(self.name, ':', time.time() - self.t, 's')

def get_hash(s):
    m = hashlib.md5()
    m.update(s.encode('utf8'))
    return m.hexdigest()

def read_config(path):
    with open(path, 'r') as fin:
        return json.load(fin)
