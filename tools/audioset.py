import argparse
import csv
import json

subway = '/m/0195fx'
singing = '/m/015lz1'
music = '/m/04rlf'
music_related = set()

def recursive_mark(ont, lbl):
    item = ont[lbl]
    if lbl not in music_related:
        music_related.add(lbl)
        for i in item['child_ids']:
            recursive_mark(ont, i)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('csv')
    args.add_argument('out')
    args.add_argument('--ontology')
    args = args.parse_args()

    if args.ontology:
        with open(args.ontology, 'r', encoding='utf8') as fin:
            ontology = json.load(fin)
            ontology = {o['id']: o for o in ontology}
        recursive_mark(ontology, singing)
        recursive_mark(ontology, music)

    with open(args.csv, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, skipinitialspace=True)
        segments = []
        for item in reader:
            if item[0].startswith('#'): continue
            lbls = set(item[3].split(','))
            if subway in lbls and len(music_related & lbls) == 0:
                segments.append(item)

    with open(args.out, 'w', encoding='utf8', newline='\n') as fout:
        writer = csv.writer(fout, lineterminator="\r\n")
        writer.writerow(['# YTID', 'start_seconds', 'end_seconds', 'positive_labels'])
        writer.writerows(segments)

    print(len(segments))
