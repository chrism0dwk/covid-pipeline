"""Summary functions"""

import numpy as np
import xarray
import pandas as pd

from gemlib.util import compute_state
from covid19uk.model_spec import STOICHIOMETRY


SUMMARY_DAYS = np.array([1, 7, 14, 21, 28, 35, 42, 49, 56], np.int32)


def mean_and_ci(arr, q=(0.025, 0.975), axis=0, name=None):

    if name is None:
        name = ""
    else:
        name = name + "_"

    q = np.array(q)
    mean = np.mean(arr, axis=axis)
    ci = np.quantile(arr, q=q, axis=axis)

    results = dict()
    results[name + "mean"] = mean
    for i, qq in enumerate(q):
        results[name + str(qq)] = ci[i]
    return results


def rt(input_file, output_file):
    """Reads an array of next generation matrices and
       outputs mean (ci) local Rt values.

    :param input_file: a pickled xarray of NGMs
    :param output_file: a .csv of mean (ci) values
    """

    r_it = xarray.open_dataset(input_file, group="posterior_predictive")["R_it"]

    rt = r_it.isel(time=-1).drop("time")
    rt_summary = mean_and_ci(rt, name="Rt")
    exceed = np.mean(rt > 1.0, axis=0)

    rt_summary = pd.DataFrame(
        rt_summary, index=pd.Index(r_it.coords["location"], name="location")
    )
    rt_summary["Rt_exceed"] = exceed
    rt_summary.to_csv(output_file)


def infec_incidence(input_file, output_file):
    """Summarises cumulative infection incidence
      as a nowcast, 7, 14, 28, and 56 days.

    :param input_file: a pkl of the medium term prediction
    :param output_file: csv with prediction summaries
    """

    prediction = xarray.open_dataset(input_file, group="predictions")["events"]

    offset = 4
    timepoints = SUMMARY_DAYS + offset

    # Absolute incidence
    def pred_events(events, name=None):
        num_events = np.sum(events, axis=-1)
        return mean_and_ci(num_events, name=name)

    idx = pd.Index(prediction.coords["location"], name="location")

    abs_incidence = pd.DataFrame(
        pred_events(prediction[..., offset : (offset + 1), 2], name="cases"),
        index=idx,
    )
    for t in timepoints[1:]:
        tmp = pd.DataFrame(
            pred_events(prediction[..., offset:t, 2], name=f"cases{t-offset}"),
            index=idx,
        )
        abs_incidence = pd.concat([abs_incidence, tmp], axis="columns")

    abs_incidence.to_csv(output_file, index_label="location")


def prevalence(input_files, output_file):
    """Reconstruct predicted prevalence from
       original data and projection.

    :param input_files: a list of [data pickle, prediction netCDF]
    :param output_file: a csv containing prevalence summary
    """
    offset = 4  # Account for recording lag
    timepoints = SUMMARY_DAYS + offset

    data = xarray.open_dataset(input_files[0], group="constant_data")
    prediction = xarray.open_dataset(input_files[1], group="predictions")

    predicted_state = compute_state(
        prediction["initial_state"], prediction["events"], STOICHIOMETRY
    )

    def calc_prev(state, name=None):
        prev = np.sum(state[..., 1:3], axis=-1) / np.array(data["N"])
        return mean_and_ci(prev, name=name)

    idx = pd.Index(prediction.coords["location"], name="location")
    prev = pd.DataFrame(
        calc_prev(predicted_state[..., timepoints[0], :], name="prev"),
        index=idx,
    )
    for t in timepoints[1:]:
        tmp = pd.DataFrame(
            calc_prev(predicted_state[..., t, :], name=f"prev{t-offset}"),
            index=idx,
        )
        prev = pd.concat([prev, tmp], axis="columns")

    prev.to_csv(output_file, index_label="location")
