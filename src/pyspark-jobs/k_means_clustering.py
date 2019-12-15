import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import click
import os
import operator

import jsonlines
import pandas as pd
import pyspark
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.clustering import KMeans, GaussianMixture
import json
import pyspark

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d - %(levelname)s - "
    "%(filename)s - %(lineno)s - %(message)s")
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d,%H:%M:%S")


@click.command()
@click.argument('params_json', type=click.Path(exists=True))
def main(params_json):
    logger = logging.getLogger(__name__)

    with open(params_json) as json_file:
        settings_json = json.load(json_file)

    spark = (
        pyspark.sql.SparkSession
            .builder
            .appName("Python Spark K-means")
            .enableHiveSupport()
            .getOrCreate()
    )

    path_aggregated_df = settings_json['path_aggregated_df']
    clustering_df = spark.read.parquet(path_aggregated_df)
    columns_clustering_features = columns_clustering_features = [
        'user_lifetime',
        'user_no_outgoing_activity_in_days',
        'user_account_balance_last',
        'user_spendings',
        'reloads_inactive_days',
        'reloads_count',
        'calls_outgoing_count',
        'calls_outgoing_spendings_max',
        'calls_outgoing_inactive_days',
        'calls_outgoing_to_onnet_count',
        'calls_outgoing_to_onnet_spendings',
        'calls_outgoing_to_abroad_count',
        'calls_outgoing_to_abroad_duration',
        'sms_outgoing_count',
        'sms_outgoing_spendings_max',
        'sms_outgoing_inactive_days',
        'sms_outgoing_to_onnet_count',
        'sms_outgoing_to_abroad_count',
        'gprs_session_count',
        'gprs_spendings',
        'gprs_inactive_days',
    ]
    vector_assembler = VectorAssembler(
        inputCols=columns_clustering_features,
        outputCol="initial_features")
    standard_scaler = StandardScaler(
        inputCol="initial_features",
        outputCol="features",
        withStd=True,
        withMean=True)
    vectorized_df = vector_assembler.transform(clustering_df)
    model_scaler = standard_scaler.fit(vectorized_df)
    featurized_clustering_df = model_scaler.transform(vectorized_df)
    featurization_pipeline = Pipeline(stages=[vector_assembler, standard_scaler])
    featurization_pipeline_model = featurization_pipeline.fit(clustering_df)
    model_scaler = featurization_pipeline_model.stages[-1]
    featurized_clustering_df = featurization_pipeline_model.transform(clustering_df)

    for k in settings_json['k_values']:
        kmeans = KMeans(featuresCol="features", k=k)
        model_kmeans = kmeans.fit(featurized_clustering_df)
        path_metrics_kmeans_sse = settings_json['path_metrics_kmeans_sse']
        sse = model_kmeans.computeCost(featurized_clustering_df)
        metrics_row = {"k": k, "sse": sse}

        with jsonlines.open(path_metrics_kmeans_sse, "a") as f:
            f.write(metrics_row)
        normalized_cluster_centers = model_kmeans.clusterCenters()
        scaler_mean = model_scaler.mean
        scaler_std = model_scaler.std
        cluster_sizes = model_kmeans.summary.clusterSizes
        n_obs = clustering_df.count()
        denormalized_cluster_centers = [
            (cluster_id,) + (size, 100 * size / n_obs) + tuple(center * scaler_std + scaler_mean)
            for cluster_id, (size, center) in
            enumerate(zip(cluster_sizes, normalized_cluster_centers))
        ]
        cluster_centers_pddf = pd.DataFrame.from_records(denormalized_cluster_centers)
        cluster_centers_pddf.columns = (
                ["cluster_id", "cluster_size", "cluster_size_pct"] +
                columns_clustering_features
        )
        pd.set_option("max_columns", 999)
        path_cluster_centers = settings_json['path_cluster_centers'] + "cluster_centers_kmeans__k_{}.csv".format(k)
        cluster_centers_pddf.to_csv(path_cluster_centers, index=False)
        clustered_kmeans_df = model_kmeans.transform(featurized_clustering_df)
        path_clustered_df = settings_json['path_cluster_centers'] + "clustered_kmeans__k_{}_parquet".format(k)
        clustered_kmeans_df.write.parquet(path_clustered_df)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

