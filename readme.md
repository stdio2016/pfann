# pfann
This is an unofficial reproduction of paper ["Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrasive Learning."](https://arxiv.org/abs/2010.11910)

Now I have a thesis that is a "trivial" improvement to the above paper: "Improvement of Neural Network- and Landmark-based Audio Fingerprinting" (in Traditional Chinese). [Link here](thesis.pdf)

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
Inside test (not used in my thesis anymore):
```
python genquery.py --params configs/gentest.json --len 10 --num 2000 --mode train --out out/queries/inside
```

Assume that you have installed all the datasets, then just run this to generate all queries:
```sh
./genall.sh
```

Will output to folders `out/queries/out2_snr$snr`, where `$snr` is one of -6, -4, -2, 0, 2, 4, 6, 8.
The query list (used by `matcher.py`) is `out/queries/out2_snr$snr/list.txt`, and the ground truth is `out/queries/out2_snr$snr/expected.csv`.

## Build a fingerprint database
Inside test (not used in my thesis anymore):
```
python tools/csv2txt.py --dir ../pfann_dataset/fma_medium lists/fma_medium_train.csv --out lists/fma_medium_train.txt
python builder.py lists/fma_medium_train.txt /path/to/db configs/default.json
```

Usage of `builder.py`:
```
python builder.py <music list file> <output database location> <model config>
```
Music list file is a file containing list of music file paths.
File must be UTF-8 without BOM. For example:
```
/path/to/fma_medium/000/000002.mp3
/path/to/fma_medium/000/000005.mp3
/path/to/your/music/aaa.wav
/path/to/your/music/bbb.wav
```
Model config is a JSON file like in `configs/` folder.
It is used to load a trained model.
If omitted, the model config is `configs/default.json` by default.

This program supports both MP3 and WAV audio format.
Relative paths are supported but not recommended.

## Recognize music
Usage of `matcher.py`:
```
python matcher.py <query list> <database location> <output result file>
```

Query list is a file containing list of query file paths. For example:
```
/path/to/queries/out2_snr2/000002.wav
/path/to/queries/out2_snr2/000005.wav
/path/to/song_recorded_on_street1.wav
/path/to/song_recorded_on_street2.wav
```
Database location is the place where `builder.py` saves database.

The result file will be a TSV file with 2 fields: query file path, and matched music path, but without header.
It may look like this:
```
/path/to/queries/out2_snr2/000002.wav	/path/to/fma_medium/000/000002.mp3
/path/to/queries/out2_snr2/000005.wav	/path/to/fma_medium/000/000005.mp3
/path/to/song_recorded_on_street1.wav	/path/to/your/music/aaa.wav
/path/to/song_recorded_on_street2.wav	/path/to/your/music/aaa.wav
```

Matcher will also generate a `_detail.csv` file and a `.bin` file.
CSV file contains more information about the matches.
It has 5 columns: query, answer, score, time, and part_scores.
* query: Query file path
* answer: Matched music path
* score: Matching score, used in my thesis
* time: The time when the query clip starts in the matched music, in seconds
* part_scores: Mainly used for debugging, currently empty

BIN file contains matching scores of every database music for each query.
It is used in my ensemble experiments.
The file format is a flattened 2D array of following structure, without header:
```c++
struct match_t {
  float score; // Matching score
  float offset; // The time when the query clip starts in the matched music, in seconds
};
```
The matching score of j-th database music in i-th query is at index `i * database size + j`.

## Evaluation
```
python tools/accuracy.py /path/to/query6s/expected.csv /path/to/result_detail.csv
```

## Ensemble experiment
```bash
python ensemble/svmheatmap2.py out/lm_ out/shift_4_ out/svm lin_acc.csv
```
