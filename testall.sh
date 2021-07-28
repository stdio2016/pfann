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
  if [ $2 == mirex ]; then
    python matcher.py lists/mirex-query.txt out/dbs/$1_$2 out/results/$1_$2.txt
  else
    python matcher.py out/queries/$2_test/list.txt out/dbs/$1_$2 out/results/$1_$2.txt
  fi
}
matcher_snr() {
  # model dataset snr
  python matcher.py out/queries/$2_test_snr$3/list.txt out/dbs/$1_$2 out/results/$1_$2_snr$3.txt
}
accuracy() {
  # model dataset
  if [ $2 == mirex ]; then
    python tools/accuracy.py lists/mirex-answer.txt out/results/$1_$2.txt
  else
    python tools/accuracy.py out/queries/$2_test/expected.csv out/results/$1_$2_detail.csv
  fi
}
accuracy_snr() {
  echo snr=$3
  python tools/accuracy.py out/queries/$2_test_snr$3/expected.csv out/results/$1_$2_snr$3_detail.csv
}
forall_snr() {
  # some_command model dataset
  for snr in -6 -4 -2 0 2 4 6 8
  do
    $1 $2 $3 $snr
  done
}
builder baseline_model_twcc inside
forall_snr matcher_snr baseline_model_twcc inside
forall_snr accuracy_snr baseline_model_twcc inside
