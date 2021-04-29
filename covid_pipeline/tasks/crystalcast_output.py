"""Creates CrystalCast formatted report for incidence and prevalence"""

from pathlib import Path
from datetime import date
import pickle as pkl
import numpy as np
import xarray
import pandas as pd

from gemlib.util import compute_state
from covid19uk import model_spec

from covid_pipeline.formats import make_dstl_template

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _events2xarray(samples, constant_data):
    return xarray.DataArray(
        samples,
        coords=[
            np.arange(samples.shape[0]),
            constant_data.coords["location"],
            constant_data.coords["time"],
            np.arange(samples.shape[-1]),
        ],
        dims=["iteration", "location", "time", "event"],
    )


def _xarray2dstl(xarr, value_type, geography):

    quantiles = xarr.quantile(q=QUANTILES, dim="iteration")
    quantiles = {qi: v for qi, v in zip(QUANTILES, quantiles)}
    mean = xarr.mean(dim="iteration")

    return make_dstl_template(
        group="Lancaster",
        model="StochSpatMetaPopSEIR",
        model_type="Pillar Two Testing",
        scenario="Nowcast",
        version=0.5,
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
        event_samples.sel(event=0).sum(dim="location").reset_coords(drop=True)
    )
    return infection_events


def prevalence(event_samples, initial_state, popsize):

    state = compute_state(
        initial_state, event_samples, model_spec.STOICHIOMETRY
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
    return state / popsize.sum()


def crystalcast_output(input_path, output, geography="United Kingdom"):
    """Computes incidence and prevalence and returns CrystalCast schema in XLSX format

    :param inputs: a list of [inferencedata, thin_samples]
    :param output: name of output XLSX file
    :param geography: the name of the Geography
    """
    basedir = Path(input_path)

    constant_data = xarray.open_dataset(
        basedir / "inferencedata.nc", group="constant_data"
    )
    with open(basedir / "thin_samples.pkl", "rb") as f:
        samples = pkl.load(f)

    # xarray-ify events tensor for ease of reduction
    event_samples = _events2xarray(samples["seir"], constant_data)

    # Clip off first week for initial conditions burnin
    event_samples = event_samples.isel(time=slice(7, None, None))

    # Incidence
    incidence_xarr = incidence(event_samples, constant_data["N"])

    # Prevalence
    prev_xarr = prevalence(event_samples, samples["initial_state"], constant_data["N"])

    # Rt
    rt = xarray.open_dataset(
        basedir / "reproduction_number.nc", group="posterior_predictive"
    )

    df = pd.concat(
        [
            _xarray2dstl(incidence_xarr, "incidence", geography),
            _xarray2dstl(prev_xarr, "prevalence", geography),
            _xarray2dstl(rt["R_t"], "R", geography),
        ],
        axis=0,
    )
    df.to_excel(output, index=False)


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
    parser.add_argument(
        "-g",
        "--geography",
        type=str,
        default="United Kingdom",
        help="DSTL geography",
    )
    args = parser.parse_args()

    crystalcast_output(args.results, args.output, args.geography)
