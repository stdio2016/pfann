# pfann
reproduce paper: Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrasive Learning

2020/11/27: initial model

## Prepare dataset

### FMA dataset

Download fma_medium from https://github.com/mdeff/fma and unzip to
/path/to/fma_medium .

```
python tools/listaudio.py --folder /path/to/fma_medium --out configs/fma.csv
python tools/traintestsplit.py --csv configs/fma.csv --train configs/train.csv --train-size 10000 --test configs/testval.csv --test-size 1000
python tools/traintestsplit.py --csv configs/testval.csv --train configs/validate.csv --train-size 500 --test configs/test.csv --test-size 500
```

### AudioSet

Download 3 csv files and ontology json from https://research.google.com/audioset/download.html .
Then run these to list all the videos needed:

```
python tools/audioset.py /path/to/unbalanced_train_segments.csv configs/audioset1.csv --ontology /path/to/ontology.json
python tools/audioset.py /path/to/balanced_train_segments.csv configs/audioset2.csv --ontology /path/to/ontology.json
python tools/audioset.py /path/to/eval_segments.csv configs/audioset3.csv --ontology /path/to/ontology.json
```

Use these commands to crawl videos from youtube and convert to wav:

```
python tools/audioset2.py configs/audioset1.csv /path/to/audioset
python tools/audioset2.py configs/audioset2.csv /path/to/audioset
python tools/audioset2.py configs/audioset3.csv /path/to/audioset
```

After downloading, run this command to list all successfully downloaded files:

```
python tools/listaudio.py --folder /path/to/audioset --out configs/noise.csv
```

This command will show errors because some videos are unavailable.

Finally run the command:

```
python tools/traintestsplit.py --csv configs/noise.csv --train configs/noise_train.csv --train-size 8 --test configs/noise_val.csv --test-size 2 -p
```

### Microphone impulse response dataset

Go to http://micirp.blogspot.com/ , and download files to a folder named
"micirp". Then run the commands:

```
python tools/listaudio.py --folder /path/to/micirp --out configs/micirp.csv
python tools/traintestsplit.py --csv configs/micirp.csv --train configs/micirp_train.csv --train-size 8 --test configs/micirp_val.csv --test-size 2 -p
```

### Aachen Impulse Response Database

Download zip from https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/
and unzip to /path/to/AIR_1_4 .

```
python -m datautil.ir /path/to/AIR_1_4 configs/air.csv
python tools/traintestsplit.py --csv configs/air.csv --train configs/air_train.csv --train-size 8 --test configs/air_val.csv --test-size 2 -p
```

## Train

```
python train.py -d /path/to/fma_medium/ --noise /path/to/audioset/ --air /path/to/AIR_1_4/ --micirp /path/to/micirp/ --param configs/default.json --validate -w3
```

## Generate query
```
python genquery.py -d /path/to/fma_medium/ --noise /path/to/audioset/ --micirp /path/to/micirp/ --air /path/to/AIR_1_4/ --num 2000 --out /path/to/query6s/ --len 6
```

## Builder
```
python builder.py /path/to/music_list.txt /path/to/db configs/default.json
```

## Matcher
```
python matcher.py /path/to/query_list.txt /path/to/db /path/to/result.txt
```
