# pfann
reproduce paper: Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrasive Learning

2020/11/27: initial model

## Prepare dataset

### FMA dataset

Download fma_medium from https://github.com/mdeff/fma and unzip to
`../pfann_dataset/fma_medium` .

```
python tools/listaudio.py --folder ../pfann_dataset/fma_medium --out lists/fma_medium.csv
python tools/filterduration.py --csv lists/fma_medium.csv --min-len 29.9 --out lists/fma_medium_30s.csv
python tools/traintestsplit.py --csv lists/fma_medium_30s.csv --train lists/fma_medium_train.csv --train-size 10000 --test lists/fma_medium_valtest.csv --test-size 1000
python tools/traintestsplit.py --csv lists/fma_medium_valtest.csv --train lists/fma_medium_val.csv --train-size 500 --test lists/fma_medium_test.csv --test-size 500
python tools/traintestsplit.py --csv lists/fma_medium_train.csv --train-size 2000 --train lists/fma_inside_test.csv
rm test.csv
python tools/listaudio.py --folder ../pfann_dataset/fma_large --out lists/fma_large.csv
```

### AudioSet

Download 3 csv files `unbalanced_train_segments.csv`, `balanced_train_segments.csv`, `eval_segments.csv`, and `ontology.json` from https://research.google.com/audioset/download.html .
Then run these to list all the videos needed:

```
python tools/audioset.py /path/to/unbalanced_train_segments.csv lists/audioset1.csv --ontology /path/to/ontology.json
python tools/audioset.py /path/to/balanced_train_segments.csv lists/audioset2.csv --ontology /path/to/ontology.json
python tools/audioset.py /path/to/eval_segments.csv lists/audioset3.csv --ontology /path/to/ontology.json
```

Use these commands to crawl videos from youtube and convert to wav:

```
python tools/audioset2.py lists/audioset1.csv ../pfann_dataset/audioset
python tools/audioset2.py lists/audioset2.csv ../pfann_dataset/audioset
python tools/audioset2.py lists/audioset3.csv ../pfann_dataset/audioset
```

After downloading, run this command to list all successfully downloaded files:

```
python tools/listaudio.py --folder ../pfann_dataset/audioset --out lists/noise.csv
```

This command will show errors because some videos are unavailable.

Finally run the command:

```
python tools/filterduration.py --csv lists/noise.csv --min-len 9.9 --out lists/noise_10s.csv
python tools/traintestsplit.py --csv lists/noise_10s.csv --train lists/noise_train.csv --train-size 8 --test lists/noise_val.csv --test-size 2 -p
```

### Microphone impulse response dataset

Go to http://micirp.blogspot.com/ , and download files to `../pfann_dataset/micirp`. Then run the commands:

```
python tools/listaudio.py --folder ../pfann_dataset/micirp --out lists/micirp.csv
python tools/traintestsplit.py --csv lists/micirp.csv --train lists/micirp_train.csv --train-size 8 --test lists/micirp_val.csv --test-size 2 -p
```

### Aachen Impulse Response Database

Download zip from https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/
and unzip to `../pfann_dataset/AIR_1_4`.

```
python -m datautil.ir ../pfann_dataset/AIR_1_4 lists/air.csv
python tools/traintestsplit.py --csv lists/air.csv --train lists/air_train.csv --train-size 8 --test lists/air_val.csv --test-size 2 -p
```

## Train

```
python train.py --param configs/default.json -w4
```

## Generate query
Inside test:
```
python genquery.py --params configs/gentest.json --len 10 --num 2000 --mode train --out out/queries/inside
```

## Builder
Inside test:
```
python tools/csv2txt.py --dir ../pfann_dataset/fma_medium lists/fma_medium_train.csv --out lists/fma_medium_train.txt
python builder.py lists/fma_medium_train.txt /path/to/db configs/default.json
```

Usage of `builder.py`:
```
python builder.py <music list> <output db location> <model config>
```

## Matcher
```
python matcher.py /path/to/query6s/list.txt /path/to/db /path/to/result.txt
```
Will output `result.txt`, `result.bin`, and `result_detail.csv` files.

Usage of `matcher.py`:
```
python matcher.py <query list> <db location> <output result file>
```

## Evaluation
```
python tools/accuracy.py /path/to/query6s/expected.csv /path/to/result_detail.csv
```
