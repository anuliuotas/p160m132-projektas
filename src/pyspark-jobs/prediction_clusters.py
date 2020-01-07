# -*- coding: utf-8 -*-
import json
import logging
import pathlib
import uuid
from pathlib import Path

import click
import pyspark
from dotenv import find_dotenv, load_dotenv
from pyspark.ml import classification
from pyspark.ml import evaluation
from pyspark.ml import feature
from pyspark.ml import pipeline


@click.command()
@click.argument('path_data', type=click.Path(exists=True))
@click.argument('path_parameters', type=click.Path(exists=True))
@click.argument('dir_models', type=click.Path())
def main(path_data, path_parameters, dir_models):
    logger = logging.getLogger(__name__)
    spark = (
        pyspark.sql.SparkSession
            .builder
            .appName("Python Spark Random Forest model training")
            .enableHiveSupport()
            .getOrCreate()
    )

    logger.info("Reading parquet data and splitting into test and train datasets")
    data_df = spark.read.parquet(path_data)
    data_df = data_df.withColumnRenamed("prediction", "cluster_index").withColumnRenamed("features", "cluster_features")
    clusters = data_df.select('cluster_index').distinct().toPandas()
    for index, row in clusters.iterrows():
        cluster_index = row['cluster_index']
        cluster_df = data_df.filter(data_df.cluster_index == cluster_index.item())

        splits = cluster_df.randomSplit([0.7, 0.3])
        training_df = splits[0]
        validation_df = splits[1]

        logger.info("Constructing pipeline for prediction model")
        with open(path_parameters) as json_file:
            parameters = json.load(json_file)
        feature_columns = parameters['feature_columns']
        rf_params = parameters['rf_params']
        assembler = feature.VectorAssembler(
            inputCols=feature_columns,
            outputCol="features")

        rf = classification.RandomForestClassifier(
            labelCol="churn", **rf_params)

        rf_pipeline = pipeline.Pipeline(stages=[assembler, rf])
        logger.info("Training prediction model")
        pipeline_model = rf_pipeline.fit(training_df)

        logger.info("Calculating model metrics")
        train_predictions_df = pipeline_model.transform(training_df)
        validation_predictions_df = pipeline_model.transform(validation_df)

        accuracy_evaluator = evaluation.MulticlassClassificationEvaluator(
            metricName="accuracy", labelCol="churn", predictionCol="prediction")

        precision_evaluator = evaluation.MulticlassClassificationEvaluator(
            metricName="weightedPrecision", labelCol="churn", predictionCol="prediction")

        recall_evaluator = evaluation.MulticlassClassificationEvaluator(
            metricName="weightedRecall", labelCol="churn", predictionCol="prediction")

        f1_evaluator = evaluation.MulticlassClassificationEvaluator(
            metricName="f1", labelCol="churn", predictionCol="prediction")

        auroc_evaluator = evaluation.BinaryClassificationEvaluator(metricName='areaUnderROC', labelCol="churn")

        logger.info("Saving model and metrics data")
        train_metrics = {
            "accuracy": accuracy_evaluator.evaluate(train_predictions_df),
            "precision": precision_evaluator.evaluate(train_predictions_df),
            "recall": recall_evaluator.evaluate(train_predictions_df),
            "f1": f1_evaluator.evaluate(train_predictions_df),
            "auroc": auroc_evaluator.evaluate(train_predictions_df)
        }
        validation_metrics = {
            "accuracy": accuracy_evaluator.evaluate(validation_predictions_df),
            "precision": precision_evaluator.evaluate(validation_predictions_df),
            "recall": recall_evaluator.evaluate(validation_predictions_df),
            "f1": f1_evaluator.evaluate(validation_predictions_df),
            "auroc": auroc_evaluator.evaluate(validation_predictions_df)
        }

        rf_model = pipeline_model.stages[-1]
        model_params = rf_model.extractParamMap()
        model_description = {
            "name": "Random Forest",
            "params": {param.name: value for param, value in model_params.items()},
        }

        dir_model = pathlib.Path(dir_models).joinpath(str(cluster_index.item()))
        dir_model.mkdir(parents=True, exist_ok=True)

        path_pipeline_model = pathlib.Path(dir_model).joinpath("pipeline_model")
        path_train_metrics = pathlib.Path(dir_model).joinpath("metrics_train.json")
        path_validation_metrics = pathlib.Path(dir_model).joinpath("metrics_validation.json")
        path_model_description = pathlib.Path(dir_model).joinpath("model_description.json")

        pipeline_model.save(str(path_pipeline_model))
        with open(path_train_metrics, "w") as f:
            json.dump(train_metrics, f)
        with open(path_validation_metrics, "w") as f:
            json.dump(validation_metrics, f)
        with open(path_model_description, "w") as f:
            json.dump(model_description, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
