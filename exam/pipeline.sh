checkarg() {
  if [ $# -lt 4 ]
  then
    echo Usage $0 data_dir src_domain tgt_domain epoch
    exit
  fi
}
checkarg $@

main() {
  data_dir=$1
  src_domain=$2
  tgt_domain=$3
  epoch=$4
  make generatevoca MODE=DEBUG DATA=$data_dir
  make sentitrain DATA=$data_dir DOMAIN=$src_domain
  make sentitrain DATA=$data_dir DOMAIN=$tgt_domain
  make mergeUI MODE=DEBUG DATA=$data_dir
  make DSNRec DATA=$data_dir SRCDO=$src_domain TGTDO=$tgt_domain EPOCH=$epoch MODE=DEBUG
  make DSNRec DATA=$data_dir SRCDO=$tgt_domain TGTDO=$src_domain EPOCH=$epoch MODE=DEBUG
}

main $*
