for d in `ls`
do
  if [ -d $d ]
  then
    if [ $d == "MultiCross" -o $d == "log" ]
    then
      continue
    fi
    if [ ! -e $d/preprocess/csv_format ]
    then
      make transCSV MODE=DEBUG DATA=$d
    fi
    src=$(basename `find $d/preprocess/sentiModel -maxdepth 1 -mindepth 1 | head -n 1`)
    tgt=$(basename `find $d/preprocess/sentiModel -maxdepth 1 -mindepth 1 | tail -n 1`)
    bash baseline.sh $d $src $tgt
  fi
done
