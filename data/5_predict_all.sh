rm -r ./output/prediction
spark-submit ../src/pyspark-jobs/prediction.py ./output/joined/ ./parameters.json ./output/prediction/
