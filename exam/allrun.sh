for d in `ls`; do if [ ! -d $d ]; then continue; fi; if [ $d == data -o $d == log ]; then continue; fi; ./check.sh $d; done
