# every utils that don't use torch
import csv
import hashlib
import json
import os
import tempfile
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

s3_resource = None
def get_s3_resource():
    import boto3
    global s3_resource
    if s3_resource is None:
        s3_resource = boto3.resource('s3', endpoint_url='https://cos.twcc.ai')
    return s3_resource

def download_tmp_from_s3(s3url):
    s3_res = get_s3_resource()
    d1 = s3url.find('/', 5)
    bucket_name = s3url[5:d1]
    object_name = s3url[d1+1:]
    ext = os.path.splitext(s3url)[1]
    obj = s3_res.Object(bucket_name, object_name)
    _, tmpname = tempfile.mkstemp(suffix=ext, prefix='pfann')
    try:
        obj.download_file(tmpname)
        return tmpname
    except Exception as x:
        os.unlink(tmpname)
        raise RuntimeError('Unable to download %s: %s' % (s3url, x))
