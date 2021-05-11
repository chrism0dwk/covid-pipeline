"""Creates CrystalCast formatted report for incidence and prevalence"""

from pathlib import Path
from datetime import date
import pickle as pkl
import numpy as np
import xarray
import pandas as pd

from gemlib.util import compute_state

import covid19uk
from covid19uk import model_spec

from covid_pipeline.formats import make_dstl_template

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _events2xarray(samples, constant_data):
    event_samples = xarray.DataArray(
        samples["seir"],
        coords=[
            np.arange(samples["seir"].shape[0]),
            constant_data.coords["location"],
            constant_data.coords["time"],
            np.arange(samples["seir"].shape[-1]),
        ],
        dims=["iteration", "location", "time", "event"],
    )
    initial_state = xarray.DataArray(
        samples["initial_state"],
        coords=[
            constant_data.coords["location"],
            np.arange(samples["initial_state"].shape[1]),
        ],
        dims=["location", "state"],
    )
    return xarray.Dataset({"seir": event_samples, "initial_state": initial_state})


def _xarray2dstl(xarr, value_type, geography):

    quantiles = xarr.quantile(q=QUANTILES, dim="iteration")
    quantiles = {qi: v for qi, v in zip(QUANTILES, quantiles)}
    mean = xarr.mean(dim="iteration")

    return make_dstl_template(
        group="Lancaster",
        model="StochSpatMetaPopSEIR",
        model_type="Pillar Two Testing",
        scenario="Nowcast",
        version=covid19uk.__version__,
        creation_date=date.today(),
        value_date=xarr.coords["time"].data,
        age_band="All",
        geography=geography,
        value_type=value_type,
        value=mean,
        quantiles=quantiles,
    )


def incidence(event_samples, popsize):
    """Select infection events, aggregate over location, divide by total popsize"""
    infection_events = (
        event_samples["seir"].sel(event=0).sum(dim="location").reset_coords(drop=True)
    )
    return infection_events


def prevalence(event_samples, popsize):
    """Prevalence in percentage units"""
    state = compute_state(
        event_samples["initial_state"], event_samples["seir"], model_spec.STOICHIOMETRY
    ).numpy()
    state = state[..., 1:3].sum(axis=-1).sum(axis=1)  # Sum E+I and location
    state = xarray.DataArray(
        state,
        coords=[
            event_samples.coords["iteration"],
            event_samples.coords["time"],
        ],
        dims=["iteration", "time"],
    )
    return state / popsize.sum() * 100


def summarize_supergeography(event_samples, rt, population, geography_name):
    """Computes incidence, prevalence, and Rt for given super-geography

    :param event_samples: an xr.DataArray with dims ['iteration','location','time','event']
    :param population: an xr.DataArray with dims ['location']
    :param initial_state: initial state of epidemic model
    :param geography: name of the geography
    :returns pandas data frame
    """

    # Incidence
    incidence_xarr = incidence(event_samples, population)

    # Prevalence
    prev_xarr = prevalence(event_samples, population)

    # Rt
    rt_summary = (rt["R_it"] * population / population.sum()).sum(dim="location")

    df = pd.concat(
        [
            _xarray2dstl(incidence_xarr, "incidence", geography_name),
            _xarray2dstl(prev_xarr, "prevalence", geography_name),
            _xarray2dstl(rt_summary, "R", geography_name),
        ],
        axis=0,
    )
    return df


def crystalcast_output(input_files, output):
    """Computes incidence and prevalence and returns CrystalCast schema in XLSX format

    :param input_files: a list of [inferencedata, thin_samples, reproduction_number]
    :param output: name of output XLSX file
    """

    constant_data = xarray.open_dataset(files[0], group="constant_data")
    with open(files[1], "rb") as f:
        samples = pkl.load(f)
    rt = xarray.open_dataset(files[2], group="posterior_predictive")

    # xarray-ify events tensor for ease of reduction
    event_samples = _events2xarray(samples, constant_data)

    # Clip off first week for initial conditions burnin
    event_samples = event_samples.isel(time=slice(7, None, None))

    # population
    population = constant_data["N"]

    # UK
    df = [summarize_supergeography(event_samples, rt, population, "United Kingdom")]

    # DAs
    for country in ["England", "Scotland", "Wales", "Northern Ireland"]:
        regions = event_samples.coords["location"].str.startswith(country[0])
        country_samples = event_samples.sel(location=regions)
        country_population = population.sel(location=regions)
        country_rt = rt.sel(location=regions)
        df.append(
            summarize_supergeography(
                country_samples, country_rt, country_population, country
            )
        )

    pd.concat(df, axis="rows").to_excel(output, index=False)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results",
        required=True,
        type=str,
        help="Path to results directory",
    )
    parser.add_argument("output", type=str, help="Output xlsx")
    args = parser.parse_args()

    basedir = Path(args.results)
    files = [
        basedir / "inferencedata.nc",
        basedir / "thin_samples.pkl",
        basedir / "reproduction_number.nc",
    ]
    crystalcast_output(files, args.output)
