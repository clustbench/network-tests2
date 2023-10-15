#!/usr/bin/env python3

import pandas as pd
import numpy as np
from path import Path  # pip install --user path.py
import click  # pip install --user click
import logging
from predict import Predictor, load_test_results, load_topology_data, TestResults, TopologyData, \
                    check_all_classes_covered
from typing import Iterable, Dict  # mypy
import itertools


def join_dict_to_table(dict_: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Can be used on test_results.{medians,means,std_dev}.
    
    Takes a dict (msg_len -> df(node1, node2, ping))
    
    Returns df(msg_len, node1, node2, ping) with dummy index"""
    return pd.concat(dict_, names=["msg_len", "dumb_index"], verify_integrity=True) \
        .reset_index(level=1, drop=True) \
        .reset_index()


def join_ping_data(multiple_tests_results: Iterable[TestResults]) -> pd.DataFrame:
    """Takes medians from all of them and concatenates them all.
    You get a DataFrame with columns (msg_len, node1, node2, ping)"""
    return pd.concat(
        (join_dict_to_table(test_results.medians) for test_results in multiple_tests_results),
        ignore_index=True
    )


def count_validation_only_hostnames(
    training_dataset: TestResults, validation_datasets: Iterable[TestResults]
) -> int:
    training_hostnames = frozenset(training_dataset.hostnames)
    validation_hostnames = frozenset(
        itertools.chain.from_iterable(data.hostnames for data in validation_datasets)
    )
    hostnames_unique_to_validation = validation_hostnames.difference(training_hostnames)
    return len(hostnames_unique_to_validation)


def count_validation_only_hostnames_on_avg(
    training_datasets: Iterable[TestResults], all_datasets: Iterable[TestResults]
) -> None:
    counts = [
        count_validation_only_hostnames(
            training_dataset,
            [data for data in all_datasets if data != training_dataset]
        )
        for training_dataset in training_datasets
    ]
    print(
        "When training on one dataset and validating using all others,"
        "on average we have {0} nodes unique to validation".format(np.mean(counts))
    )


def calc_relative_error(
    topology_data: TopologyData, training_dataset: TestResults,
    validation_datasets: Iterable[TestResults]
) -> pd.DataFrame:
    """Returns DataFrame with columns: training_dataset, msg_len, ping, predicted_ping, rel_error.
    """
    validation_data = join_ping_data(dataset for dataset in validation_datasets)
    predictor = Predictor(topology_data, training_dataset)
    # build df with cols: msg_len, ping, predicted_ping
    predictions = predictor.predict_many(validation_data).drop(["node1", "node2"], axis=1)
    abs_error = (predictions["value"] - predictions["predicted_ping"]).abs()
    predictions["rel_error"] = (abs_error / predictions["value"]) \
        .fillna(0)  # 0/0 returns NaN. I replace it with 0.
    predictions["training_dataset"] = training_dataset.name
    return predictions


def calc_relative_error_for_all(
    topology_data: TopologyData, training_datasets: Iterable[TestResults],
    all_datasets: Iterable[TestResults]
) -> None:
    """
    For each training dataset, predict pings for all validation datasets.
    Calculate relative error for all those pings.
    (rel_error = abs(predicted_ping - actual_ping)/actual_ping or 0 if actual_ping == 0)
    Calculate mean of all relative error values.
    Calculate max of all relative error values.
    For percentage in (50%, 80%, 90%, 95%, 99%):
        calculate x such that this percentage of relative error values <= x
    Nicely print all these values."""
    training_datasets = list(training_datasets)
    all_datasets = list(all_datasets)
    # build iterable of pairs (training_dataset, validation_datasets)
    args = (
        (training_dataset, (data for data in all_datasets if data is not training_dataset))
        for training_dataset in training_datasets
    )
    # calculate all predictions and relative errors
    # it's a big df with cols: training_dataset, msg_len, ping, predicted_ping, rel_error
    results = pd.concat(
        [calc_relative_error(topology_data, *pair) for pair in args],
        ignore_index=True
    )
    mean_rel_error = results["rel_error"].mean(skipna=False)
    max_rel_error = results["rel_error"].max(skipna=False)
    quantiles = results["rel_error"].quantile([0.01, 0.05, 0.15, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99], interpolation="midpoint")
    print("""Mean of abs(predicted_ping - ping)/ping = {0}
          Max of abs(predicted_ping - ping)/ping = {1}""".format(mean_rel_error, max_rel_error))
    for (percentage, rel_error) in quantiles.iteritems():
        print(
            "{0:.0%} of abs(predicted_ping - ping)/ping values are lesser or equal than {1}"
                .format(percentage, rel_error)
        )


def validate(topology_data: TopologyData, datasets: Iterable[TestResults]) -> None:
    datasets = list(datasets)
    training_datasets = [
        data for data in datasets
        if check_all_classes_covered(topology_data, data.hostnames)
    ]
    non_training_datasets = [
        data for data in datasets
        if data not in training_datasets
    ]
    print("""The following datasets have at least one class of node pairs not represented
          in each of them, thus we will not train on them:""")
    for test_results in non_training_datasets:
        print("\t'{0}'".format(test_results.name))
    count_validation_only_hostnames_on_avg(training_datasets, datasets)
    calc_relative_error_for_all(topology_data, training_datasets, datasets)


@click.command()
@click.option(
    "--tests-results", "-t", required=True, envvar="NT2_TESTS_RESULTS",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
    help="""Path to directory that contains network-tests2 results in subdirectories.
         Can be passed as NT2_TESTS_RESULTS environment variable"""
)
@click.option("--verbose", "-v", is_flag=True)
def main(tests_results, verbose):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    topology_data = load_topology_data()
    datasets = (load_test_results(dir_) for dir_ in Path(tests_results).dirs())
    validate(topology_data, datasets)
        

if __name__ == "__main__":
    main()
