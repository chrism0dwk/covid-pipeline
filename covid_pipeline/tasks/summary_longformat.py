"""Produces a long-format summary of fitted model results"""

from datetime import date
import numpy as np
import pandas as pd
import xarray

from gemlib.util import compute_state
from covid19uk.model_spec import STOICHIOMETRY
from covid19uk.version import VERSION as model_version

from covid_pipeline.formats import make_dstl_template


def xarray2summarydf(arr):
    mean = arr.mean(dim="iteration").to_dataset(name="value")
    q = np.arange(start=0.05, stop=1.0, step=0.05)
    quantiles = arr.quantile(q=q, dim="iteration").to_dataset(dim="quantile")
    ds = mean.merge(quantiles).rename_vars({qi: f"{qi:.2f}" for qi in q})
    return ds.to_dataframe().reset_index()


def prevalence(prediction, popsize):
    prev = compute_state(
        prediction["initial_state"], prediction["events"], STOICHIOMETRY
    )
    prev = xarray.DataArray(
        prev.numpy(),
        coords=[
            np.arange(prev.shape[0]),
            prediction.coords["location"],
            prediction.coords["time"],
            np.arange(prev.shape[-1]),
        ],
        dims=["iteration", "location", "time", "state"],
    )
    prev_per_1e5 = (
        prev[..., 1:3].sum(dim="state").reset_coords(drop=True)
        / np.array(popsize)[np.newaxis, :, np.newaxis]
        * 100000
    )
    return xarray2summarydf(prev_per_1e5)


def weekly_pred_cases_per_100k(prediction, popsize):
    """Returns weekly number of cases per 100k of population"""

    prediction = prediction[..., 2]  # Case removals
    prediction = prediction.reset_coords(drop=True)

    # TODO: Find better way to sum up into weeks other than
    # a list comprehension.
    dates = pd.DatetimeIndex(prediction.coords["time"].data)
    first_sunday_index = np.where(dates.weekday == 6)[0][0]
    weeks = range(first_sunday_index, prediction.coords["time"].shape[0], 7)[:-1]
    week_incidence = [
        prediction[..., week : (week + 7)].sum(dim="time") for week in weeks
    ]
    week_incidence = xarray.concat(week_incidence, dim=prediction.coords["time"][weeks])
    week_incidence = week_incidence.transpose(*prediction.dims, transpose_coords=True)
    # Divide by population sizes
    week_incidence = (
        week_incidence / np.array(popsize)[np.newaxis, :, np.newaxis] * 100000
    )
    return xarray2summarydf(week_incidence)


def summary_longformat(input_files, output_file):
    """Draws together pipeline results into a long format
       csv file.

    :param input_files: a list of filenames [data_pkl,
                                             insample7_nc
                                             insample14_nc,
                                             medium_term_pred_nc,
                                             ngm_nc]
    :param output_file: the output CSV with columns `[date,
                        location,value_name,value,q0.025,q0.975]`
    """

    data = xarray.open_dataset(input_files[0], group="constant_data")
    cases = xarray.open_dataset(input_files[0], group="observations")["cases"]

    df = cases.to_dataframe(name="value").reset_index()
    df["value_name"] = "newCasesBySpecimenDate"
    df["0.05"] = np.nan
    df["0.5"] = np.nan
    df["0.95"] = np.nan

    # Insample predictive incidence
    insample = xarray.open_dataset(input_files[1], group="predictions")
    insample_df = xarray2summarydf(insample["events"][..., 2].reset_coords(drop=True))
    insample_df["value_name"] = "insample7_Cases"
    df = pd.concat([df, insample_df], axis="index")

    insample = xarray.open_dataset(input_files[2], group="predictions")
    insample_df = xarray2summarydf(insample["events"][..., 2].reset_coords(drop=True))
    insample_df["value_name"] = "insample14_Cases"
    df = pd.concat([df, insample_df], axis="index")

    # Medium term absolute incidence
    medium_term = xarray.open_dataset(input_files[3], group="predictions")
    medium_df = xarray2summarydf(medium_term["events"][..., 2].reset_coords(drop=True))
    medium_df["value_name"] = "absolute_incidence"
    df = pd.concat([df, medium_df], axis="index")

    # Cumulative cases
    medium_df = xarray2summarydf(
        medium_term["events"][..., 2].cumsum(dim="time").reset_coords(drop=True)
    )
    medium_df["value_name"] = "cumulative_absolute_incidence"
    df = pd.concat([df, medium_df], axis="index")

    # Medium term incidence per 100k
    medium_df = xarray2summarydf(
        (
            medium_term["events"][..., 2].reset_coords(drop=True)
            / np.array(data["N"])[np.newaxis, :, np.newaxis]
        )
        * 100000
    )
    medium_df["value_name"] = "incidence_per_100k"
    df = pd.concat([df, medium_df], axis="index")

    # Weekly incidence per 100k
    weekly_incidence = weekly_pred_cases_per_100k(medium_term["events"], data["N"])
    weekly_incidence["value_name"] = "weekly_cases_per_100k"
    df = pd.concat([df, weekly_incidence], axis="index")

    # Medium term prevalence
    prev_df = prevalence(medium_term, data["N"])
    prev_df["value_name"] = "prevalence"
    df = pd.concat([df, prev_df], axis="index")

    # Rt
    rt = xarray.load_dataset(input_files[4], group="posterior_predictive")["R_it"]
    rt_summary = xarray2summarydf(rt.isel(time=-1))
    rt_summary["value_name"] = "R"
    rt_summary["time"] = rt.coords["time"].data[-1] + np.timedelta64(1, "D")
    df = pd.concat([df, rt_summary], axis="index")

    quantiles = df.columns[df.columns.str.startswith("0.")]

    return make_dstl_template(
        group="Lancaster",
        model="SpatialStochasticSEIR",
        scenario="Nowcast",
        model_type="Cases",
        creation_date=date.today(),
        version=model_version,
        age_band="All",
        geography=df["location"],
        value_date=df["time"],
        value_type=df["value_name"],
        value=df["value"],
        quantiles={q: df[q] for q in quantiles},
    ).to_excel(output_file, index=False)


if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file")
    parser.add_argument(
        "resultsdir",
        type=str,
        help="Results directory",
    )
    args = parser.parse_args()

    input_files = [
        os.path.join(args.resultsdir, d)
        for d in [
            "inferencedata.nc",
            "insample7.nc",
            "insample14.nc",
            "medium_term.nc",
            "reproduction_number.nc",
        ]
    ]

    summary_longformat(input_files, args.output)
