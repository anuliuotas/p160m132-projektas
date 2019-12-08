import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import click

import pyspark

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d - %(levelname)s - "
    "%(filename)s - %(lineno)s - %(message)s")
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d,%H:%M:%S")


@click.command()
@click.argument('path_usage_csv', type=click.Path(exists=True))
@click.argument('path_churn_csv', type=click.Path(exists=True))
@click.argument('path_output', type=click.Path())
def main(path_usage_csv, path_churn_csv, path_output):
    logger = logging.getLogger(__name__)

    spark = (
        pyspark.sql.SparkSession
            .builder
            .appName("Python Spark SQL aggregation with join")
            .enableHiveSupport()
            .getOrCreate()
    )

    usage_df = spark.read.csv(
        path_usage_csv,
        header=True,
        inferSchema=True)
    churn_df = spark.read.csv(
        path_churn_csv,
        header=True,
        inferSchema=True)
    agg_usage_churn_df = (
        usage_df
            .join(churn_df, "user_account_id")
    )
    agg_usage_churn_df.write.parquet(path_output)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

