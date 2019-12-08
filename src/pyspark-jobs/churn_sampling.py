import logging
import os
import subprocess
import uuid
import shlex

import click

import pyspark

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d - %(levelname)s - "
    "%(filename)s - %(lineno)s - %(message)s")
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d,%H:%M:%S")


def validate_probability(ctx, param, value):
    if not 0.0 < value < 1.0:
        raise click.BadParameter("needs to be between 0.0 and 1.0")
    return value


@click.command(
    help=("generate random sample of customer churn dataset " 
          "\n\n"
          "sampling based on user_account_id, "
          "all usage rows are preserved for each sampled user, "
          "sample size is determined by provided sampling probability, "
          "sample files customer_usage.csv and customer_churn.csv "
          "are saved to provided output directory"))
@click.argument(
    "path_usage",
    type=click.Path(file_okay=True, dir_okay=False, exists=True,
                    resolve_path=True))
@click.argument(
    "path_churn",
    type=click.Path(file_okay=True, dir_okay=False, exists=True,
                    resolve_path=True))
@click.argument(
    "dir_output",
    type=click.Path(file_okay=False, dir_okay=True, exists=True,
                    resolve_path=True))
@click.argument(
    "probability",
    type=float, callback=validate_probability)
@click.option("-s", "--seed", type=int)
def main(path_usage, path_churn, dir_output, probability, seed):
    #path_usage = shlex.quote(path_usage)
    #path_churn = shlex.quote(path_churn)
    #dir_output = shlex.quote(dir_output)
    
    logger = logging.getLogger(__name__)

    logger.info("sampling with probability {:.2f}".format(
        probability))

    spark = (
        pyspark.sql.SparkSession
        .builder
        .appName("churn_sampling")
        .enableHiveSupport()
        .getOrCreate())
    
    logger.info("Path churn: {}".format(path_churn))
    logger.info("Path usage: {}".format(path_usage))
    churn_df = spark.read.csv(path_churn, header=True)
    usage_df = spark.read.csv(path_usage, header=True)

    n_rows_churn = churn_df.count()
    logger.info("initial churn rows count {}".format(n_rows_churn))

    n_rows_usage = usage_df.count()
    logger.info("initial usage rows count {}".format(n_rows_usage))

    seed = seed or 111111
    churn_sample_df = churn_df.sample(False, probability, seed=seed)

    n_sampled_users = churn_sample_df.count()
    logger.info("sample has {} ({:.2f}%) churn rows".format(
        n_sampled_users,
        100 * n_sampled_users / n_rows_churn))

    churn_sample_df.createOrReplaceTempView("churn_sample")
    usage_df.createOrReplaceTempView("usage_full")

    query = """
        SELECT usage_full.*
        FROM churn_sample
        JOIN usage_full
            ON usage_full.user_account_id = churn_sample.user_account_id
    """

    usage_sample_df = spark.sql(query)

    n_sampled_usage = usage_sample_df.count()
    logger.info("sample has {} ({:.2f}%) usage rows".format(
        n_sampled_usage,
        100 * n_sampled_usage / n_rows_usage))

    sample_uuid = uuid.uuid4()

    path_tmp_output_usage = os.path.join(
        dir_output,
        "tmp_usage_sample_{}".format(sample_uuid))

    path_tmp_output_churn = os.path.join(
        dir_output,
        "tmp_churn_sample_{}".format(sample_uuid))

    logger.info("saving temporary usage sample output to directory {}".format(
        path_tmp_output_usage))
    usage_sample_df.write.csv(path_tmp_output_usage)

    logger.info("saving temporary churn sample output to directory {}".format(
        path_tmp_output_churn))
    churn_sample_df.write.csv(path_tmp_output_churn)

    spark.stop()
    logger.info("spark session stopped")

    path_output_usage = os.path.join(dir_output, "customer_usage.csv")
    path_output_churn = os.path.join(dir_output, "customer_churn.csv")

    cmd_make_usage = mk_cmd_cat_header_to_output(
        path_usage, path_tmp_output_usage, path_output_usage)

    cmd_make_churn = mk_cmd_cat_header_to_output(
        path_churn, path_tmp_output_churn, path_output_churn)

    logger.info("creating usage sample csv file {}".format(path_output_usage))
    subprocess.call(cmd_make_churn, shell=True)

    logger.info("creating churn sample csv file {}".format(path_output_churn))
    subprocess.call(cmd_make_usage, shell=True)

    cmd_rm_tmp_usage = mk_cmd_rm_dir(path_tmp_output_usage)
    cmd_rm_tmp_churn = mk_cmd_rm_dir(path_tmp_output_churn)

    logger.info("removing temporary usage sample output directory {}".format(
        cmd_rm_tmp_usage))
    subprocess.call(cmd_rm_tmp_usage, shell=True)

    logger.info("removing temporary churn sample output directory {}".format(
        cmd_rm_tmp_churn))
    subprocess.call(cmd_rm_tmp_churn, shell=True)

    logging.info("sampling done!")

    spark.stop()


def mk_cmd_cat_header_to_output(
        path_initial_file,
        path_sample_output,
        path_output
):
    """
    Make Bash command to generate sample csv file
    
    Command concatenates header from initial input csv file,
    files from pySpark output directory and saves the sample
    csv file to provided output file
    
    Parameters
    ----------
    path_initial_file : str
        path to initial csv file
    path_sample_output : str
        path to pySpark output directory
    path_output : str
        path to output sample csv file

    Returns
    -------
    str

    """
    return "(head -n 1 {}; cat {}/*) > {}".format(
        path_initial_file, path_sample_output, path_output)


def mk_cmd_rm_dir(path_dir):
    """
    Make Bash command to remove director and its contents
    
    Parameters
    ----------
    path_dir : str
        path to directory to remove

    Returns
    -------
    str

    """
    return "rm -rf {}".format(path_dir)


if __name__ == "__main__":
    main()

