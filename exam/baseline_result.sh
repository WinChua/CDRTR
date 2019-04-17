# sed -r "s/(.*)\s+training_time(.*)RMSE ([^ ]*)(.*)/\1 RMSE \3/" log/baseline/Auto.Random.log | grep RMSE

for log in `ls log/baseline/*`
do
  grep -o "new users: RMSE \([^ ]*\)" $log | xargs echo $log
done
