# every utils that don't use torch
import csv
import hashlib
import json
import time

class Timing():
    def __init__(self, name='run time'):
        self.name = name
        self.t = time.time()
        self.entered = False
    def __enter__(self):
        self.t = time.time()
        self.entered = True
    def __exit__(self, *ignored):
        self.showRunTime(self.name)
    def showRunTime(self, name):
        print(self.name, ':', time.time() - self.t, 's')

def get_hash(s):
    m = hashlib.md5()
    m.update(s.encode('utf8'))
    return m.hexdigest()

def read_config(path):
    with open(path, 'r') as fin:
        return json.load(fin)

def read_file_list(list_file):
    files = []
    if list_file.endswith('.csv'):
        with open(list_file, 'r') as fin:
            reader = csv.reader(fin)
            firstrow = next(reader)
            files = [row[0] for row in reader]
    else:
        with open(list_file, 'r', encoding='utf8') as fin:
            for line in fin:
                if line.endswith('\n'):
                    line = line[:-1]
                files.append(line)
    return files
