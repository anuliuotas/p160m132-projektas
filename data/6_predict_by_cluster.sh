rm -r ./output/prediction_by_clusters
spark-submit ../src/pyspark-jobs/prediction_clusters.py ./output/k_means/clustered_kmeans__k_5_parquet/ ./parameters.json ./output/prediction_by_clusters/
