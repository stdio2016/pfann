# pfann
reproduce paper: Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrasive Learning

2020/11/27: initial model

## Prepare dataset

### FMA dataset

Download fma_medium from https://github.com/mdeff/fma and unzip to
`/path/to/fma_medium` .

```
python tools/listaudio.py --folder /path/to/fma_medium --out lists/fma_medium.csv
python tools/filterduration.py --csv lists/fma_medium.csv --min-len 29.9 --out lists/fma_medium_30s.csv
python tools/traintestsplit.py --csv lists/fma_medium_30s.csv --train lists/fma_medium_train.csv --train-size 10000 --test lists/fma_medium_valtest.csv --test-size 1000
python tools/traintestsplit.py --csv lists/fma_medium_valtest.csv --train lists/fma_medium_val.csv --train-size 500 --test lists/fma_medium_test.csv --test-size 500
```

### AudioSet

Download 3 csv files and ontology json from https://research.google.com/audioset/download.html .
Then run these to list all the videos needed:

```
python tools/audioset.py /path/to/unbalanced_train_segments.csv lists/audioset1.csv --ontology /path/to/ontology.json
python tools/audioset.py /path/to/balanced_train_segments.csv lists/audioset2.csv --ontology /path/to/ontology.json
python tools/audioset.py /path/to/eval_segments.csv lists/audioset3.csv --ontology /path/to/ontology.json
```

Use these commands to crawl videos from youtube and convert to wav:

```
python tools/audioset2.py lists/audioset1.csv /path/to/audioset
python tools/audioset2.py lists/audioset2.csv /path/to/audioset
python tools/audioset2.py lists/audioset3.csv /path/to/audioset
```

After downloading, run this command to list all successfully downloaded files:

```
python tools/listaudio.py --folder /path/to/audioset --out lists/noise.csv
```

This command will show errors because some videos are unavailable.

Finally run the command:

```
python tools/filterduration.py --csv lists/noise.csv --min-len 9.9 --out lists/noise_10s.csv
python tools/traintestsplit.py --csv lists/noise_10s.csv --train lists/noise_train.csv --train-size 8 --test lists/noise_val.csv --test-size 2 -p
```

### Microphone impulse response dataset

Go to http://micirp.blogspot.com/ , and download files to `/path/to/micirp`. Then run the commands:

```
python tools/listaudio.py --folder /path/to/micirp --out lists/micirp.csv
python tools/traintestsplit.py --csv lists/micirp.csv --train lists/micirp_train.csv --train-size 8 --test lists/micirp_val.csv --test-size 2 -p
```

### Aachen Impulse Response Database

Download zip from https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/
and unzip to `/path/to/AIR_1_4`.

```
python -m datautil.ir /path/to/AIR_1_4 lists/air.csv
python tools/traintestsplit.py --csv lists/air.csv --train lists/air_train.csv --train-size 8 --test lists/air_val.csv --test-size 2 -p
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
python tools/csv2txt.py --dir /path/to/fma_medium list/fma_medium_train.csv --out build_fma_medium_train.txt
python tools/csv2txt.py --dir /path/to/fma_medium list/fma_medium_test.csv --out build_fma_medium_test.txt
cat build_fma_medium_train.txt build_fma_medium_test.txt > build_fma_medium.txt
python builder.py build_fma_medium.txt /path/to/db configs/default.json
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
python builder.py <query list> <db location> <output result file>
```

## Evaluation
```
python tools/accuracy.py /path/to/query6s/expected.csv /path/to/result_detail.csv
```
