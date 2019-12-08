import logging
import os
import glob
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import shutil

import click

import pyspark

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d - %(levelname)s - "
    "%(filename)s - %(lineno)s - %(message)s")
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d,%H:%M:%S")

TMP_DIR = 'aggregated_customer_usage_TMP'

@click.command()
@click.argument('path_usage', type=click.Path(exists=True))
@click.argument('path_output', type=click.Path())
def main(path_usage, path_output):
    logger = logging.getLogger(__name__)

    tmp_dir_path = os.path.join(path_output, TMP_DIR)
    spark = (
        pyspark.sql.SparkSession
            .builder
            .appName("Python Spark SQL aggregation example")
            .enableHiveSupport()
            .getOrCreate()
    )
    os.makedirs(path_output, exist_ok=True)

    usage_df = spark.read.csv(path_usage, header=True, inferSchema=True, sep=',')
    usage_df.createOrReplaceTempView("customer_usage")

    date_columns = ["year", "month"]
    id_columns = ["user_account_id"]
    binary_columns = [
        "user_intake",
        "user_has_outgoing_calls", "user_has_outgoing_sms",
        "user_use_gprs", "user_does_reload"
    ]

    categorical_columns = date_columns + binary_columns + id_columns

    continuous_columns = [c for c in usage_df.columns if c not in categorical_columns]

    sql_expressions_avg = ["AVG({0}) AS {0}".format(c) for c in continuous_columns]
    sql_expressions_max = ["MAX({0}) AS {0}".format(c) for c in binary_columns]
    sql_expressions_count = ["COUNT(*) AS n_months"]
    sql_expressions_aggregation = sql_expressions_avg + sql_expressions_max + sql_expressions_count

    sql_query_aggregate_by_user_id = """
    SELECT user_account_id, {}
    FROM customer_usage
    GROUP BY user_account_id
    """.format("\n , ".join(sql_expressions_aggregation))

    aggregate_usage_df = spark.sql(sql_query_aggregate_by_user_id)
    aggregate_usage_df.write.csv(tmp_dir_path)

    with open(os.path.join(path_output, TMP_DIR, "header__aggregated_customer_usage.txt"), "w") as f:
        f.write(",".join(aggregate_usage_df.columns) + "\n")

    with open(os.path.join(path_output, TMP_DIR, "header__aggregated_customer_usage.txt")) as f:
        columns = f.read().rstrip().split(",")
    combined_df = join_csv_files(tmp_dir_path, columns)

    combined_df.to_csv(os.path.join(tmp_dir_path, "..", "combined_csv.csv"), index=False,
                        encoding='utf-8-sig')
    try:
        shutil.rmtree(tmp_dir_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def join_csv_files(file_dir, coll_names):
    extension = 'csv'
    all_filenames = [i for i in glob.glob(file_dir + '/*.{}'.format(extension))]
    combined_csv = pd.concat([pd.read_csv(f, sep=',', names=coll_names) for f in all_filenames])
    return combined_csv


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

