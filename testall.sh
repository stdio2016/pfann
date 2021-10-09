# model is one of: baseline shuffle_1000 shuffle_100 shuffle_10 shuffle_1
# dataset is one of: inside out1 out2 mirex
builder() {
  # model dataset
  if [ $2 == inside ]; then
    list=lists/fma_medium_train.txt
  elif [ $2 == out1 ]; then
    list=lists/fma_out1.txt
  elif [ $2 == out2 ]; then
    list=lists/fma_out2.txt
  elif [ $2 == mirex ]; then
    list=lists/mirex-db.txt
  else
    echo $2 is not a supported dataset
    exit 2
  fi
  python builder.py $list out/dbs/$1_$2 out/models/$1
}
matcher() {
  # model dataset
  if [[ $1 =~ ^lm ]]; then
    prog=../pfa/matcher
  else
    prog="python matcher.py"
  fi
  if [ $2 == mirex ]; then
    $prog lists/mirex-query.txt out/dbs/$1_$2 out/results/$1_$2.txt
  else
    $prog out/queries/$2/list.txt out/dbs/$1_$2 out/results/$1_$2.txt
  fi
}
matcher_snr() {
  # model dataset snr
  if [[ $1 =~ ^lm ]]; then
    prog=../pfa/matcher
  else
    prog="python matcher.py"
  fi
  $prog out/queries/$2_snr$3/list.txt out/dbs/$1_$2 out/results/$1_$2_snr$3.txt
}
matcher_snr_full() {
  # model dataset snr
  if [[ $1 =~ ^lm ]]; then
    prog=../pfa/matcher
  else
    prog="python matcher.py"
  fi
  $prog out/queries/$2_snr$3/list.txt out/dbs/$1_full out/results/$1_$2_full_snr$3.txt
}
accuracy() {
  # model dataset
  if [ $2 == mirex ]; then
    python tools/mirexacc.py lists/mirex-answer.txt out/results/$1_$2.txt
  else
    python tools/accuracy.py out/queries/$2/expected.csv out/results/$1_$2_detail.csv
  fi
}
accuracy_snr() {
  echo snr=$3
  if [[ $1 =~ ^lm ]]; then
    python tools/accuracy.py out/queries/$2_snr$3/expected.csv out/results/$1_$2_snr$3.txt.csv
  else
    python tools/accuracy.py out/queries/$2_snr$3/expected.csv out/results/$1_$2_snr$3_detail.csv
  fi
}
accuracy_snr_full() {
  echo snr=$3
  if [[ $1 =~ ^lm ]]; then
    python tools/accuracy.py out/queries/$2_snr$3/expected.csv out/results/$1_$2_full_snr$3.txt.csv
  else
    python tools/accuracy.py out/queries/$2_snr$3/expected.csv out/results/$1_$2_full_snr$3_detail.csv
  fi
}
forall_snr() {
  # some_command model dataset
  for snr in -6 -4 -2 0 2 4 6 8
  do
    $1 $2 $3 $snr
  done
}
model="$1"
dataset="$2"
shift 2
while [[ $# -gt 0 ]]
do
  action="$1"
  shift
  case "$action" in
    "-build" )
      builder $model $dataset || exit 1;;
    "-match_snr" )
      forall_snr matcher_snr $model $dataset || exit 1;;
    "-accuracy_snr" )
      forall_snr accuracy_snr $model $dataset || exit 1;;
    "-match_snr_full" )
      forall_snr matcher_snr_full $model $dataset || exit 1;;
    "-accuracy_snr_full" )
      forall_snr accuracy_snr_full $model $dataset || exit 1;;
    "-match" )
      matcher $model $dataset || exit 1;;
    "-accuracy" )
      accuracy $model $dataset || exit 1;;
  esac
done
