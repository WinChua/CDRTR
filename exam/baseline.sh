checkarg() {
  if [ $# -lt 3 ]
  then
    echo Usage $0 data_dir src_domain tgt_domain
    exit
  fi
}
checkarg $@

main() {
  data_dir=$1
  src_domain=$2
  tgt_domain=$3
  tgt_train=`ls ${data_dir}/preprocess/csv_format/*${tgt_domain}*train*`
  tgt_test=`ls ${data_dir}/preprocess/csv_format/*${tgt_domain}*test*`
  src_train=`ls ${data_dir}/preprocess/csv_format/*${src_domain}*train*`
  src_test=`ls ${data_dir}/preprocess/csv_format/*${src_domain}*test*`

  for method in Random SlopeOne BiasedMatrixFactorization NaiveBayes LatentFeatureLogLinearModel FactorWiseMatrixFactorization GlobalAverage ItemAverage SVDPlusPlus
  do
    echo ${method}
    rating_prediction --training-file=${src_train} --test-file=${src_test} --recommender=${method} > log/baseline/${src_domain}.${method}.log
    rating_prediction --training-file=${tgt_train} --test-file=${tgt_test} --recommender=${method} > log/baseline/${tgt_domain}.${method}.log
  done
}

main $*


