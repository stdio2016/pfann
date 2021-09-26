for snr in -6 -4 -2 0 2 4 6 8
do
  python genquery.py --params configs/gentest_snr$snr.json --len 10 --num 2000 --mode test --out out/queries/out2_snr$snr
done
