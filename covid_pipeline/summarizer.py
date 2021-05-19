"""Summarizer provides methods to summarise `covid19uk` output"""

import pickle as pkl
from pathlib import Path
import xarray
import numpy as np

from gemlib.util import compute_state as gl_compute_state
from covid19uk import model_spec

# Filenames within MCMC results directory
CONSTANT_DATA = "inferencedata.nc"
REPRODUCTION_NUMBER = "reproduction_number.nc"
SAMPLES = "thin_samples.pkl"
WITHIN_BETWEEN = "within_between_summary.csv"

# Location within results directory
GROUPS = {
    "CONSTANT_DATA": "constant_data",
    "REPRODUCTION_NUMBER": "posterior_predictive",
    "SAMPLES": None,
    "WITHIN_BETWEEN": None,
}


def _events2xarray(samples, constant_data):
    """Extracts events samples and initial state from dictionary of samples
    and packs into an xarray dataset.
    """
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
    return xarray.Dataset(
        {"seir": event_samples, "initial_state": initial_state}
    )


class PosteriorFunctions:
    """Summarizer provides methods to summarize `covid19uk` output"""

    def __init__(
        self,
        results_directory,
        aggregate=False,
    ):
        """Summarizer attaches to a `covid19uk` results directory and summarizes
           the output.
        :param results_directory: string giving the path of the results directory
        :param mean: should the mean be included in the summary?
        :param quantiles: tuple of quantiles to output in summary
        :param aggregate: should the metrics be aggregated (summed) across locations?
        """
        self.__path = Path(results_directory)
        self.__aggregate = True if aggregate is True else False

    @property
    def _samples(self):
        """Returns a dictionary of MCMC samples"""
        with open(self.__path / SAMPLES, "rb") as f:
            samples = pkl.load(f)
        return samples

    @property
    def _constant_data(self):
        """Returns an `xarray.Dataset` of covariate data"""
        return xarray.open_dataset(
            str(self.__path / CONSTANT_DATA), group=GROUPS["CONSTANT_DATA"]
        )

    def _compute_state(self):
        """Reconstructs the state of the population a each timepoint"""
        event_samples = _events2xarray(self._samples, self._constant_data)
        state = gl_compute_state(
            event_samples["initial_state"],
            event_samples["seir"],
            model_spec.STOICHIOMETRY,
        ).numpy()
        return xarray.DataArray(
            state,
            coords=[
                self._constant_data.coords["iteration"],
                self._constant_data.coords["location"],
                self._constant_data.coords["time"],
                np.arange(state.shape[-1]),
            ],
            dims=["iteration", "location", "time", "state"],
        )

    def _maybe_aggregate(self, x):
        if self.__aggregate:
            return x.sum(dim="location")
        return x

    def rt(self):
        """Return Rt summary as a `xarray.Dataset`"""
        rt = xarray.open_dataset(
            self.__path / REPRODUCTION_NUMBER,
            group=GROUPS["REPRODUCTION_NUMBER"],
        )
        if self.__aggregate:
            return rt["R_t"]
        return rt["R_it"]

    def absolute_incidence(self):
        """Return `xarray.Dataset` with daily absolute
        infection incidence samples"""
        event_samples = _events2xarray(self._samples, self._constant_data)
        infection_events = (
            event_samples["seir"].sel(event=0).reset_coords(drop=True)
        )
        return self._maybe_aggregate(infection_events)

    def incidence_proportion(self):
        """Samples of daily prevalence as a proportion
        of the population."""
        incidence = self.absolute_incidence() / self._maybe_aggregate(
            self._constant_data["N"]
        )
        return incidence

    def prevalence(self):
        """Prevalence in each location.
        :returns: a `xarray.Dataset` of prevelance samples
        """
        state = self._compute_state()
        state = state.isel(state=slice(1, 3)).sum(dim="state")
        return self._maybe_aggregate(state) / self._maybe_aggregate(
            self._constant_data["N"]
        )

    def incidence_proportion_per_week(self):
        """Summarize number of new cases per week staring on a Sunday
           as a proportion of the population.  Incomplete weeks at either
           end of the downsampled timeseries are truncated.

        :returns: a `xarray.Dataset` of summary measures.
        """
        abs_incidence = self.absolute_incidence().resample(time="W").sum()
        population_size = self._maybe_aggregate(self._constant_data["N"])
        return abs_incidence / population_size


class Summarizer:
    def __init__(
        self,
        posterior_functions,
        mean=True,
        quantiles=(0.025, 0.975),
    ):
        self.__pf = posterior_functions
        self.__mean = True if mean is True else False
        self.__quantiles = quantiles

        pf_methods = [m for m in dir(self.__pf) if m[0] != "_"]
        for method in pf_methods:
            self.__dict__[method] = self._summarize(getattr(self.__pf, method))

    def _summarize(self, func):
        def summary(dataset, dim="iteration"):
            data_arrays = {}
            if self.__mean:
                data_arrays["mean"] = dataset.mean(dim=dim)
                for q in self.__quantiles:
                    data_arrays[f"q{q}"] = dataset.quantile(
                        q=q, dim=dim
                    ).reset_coords(drop=True)
            return xarray.Dataset(data_arrays)

        def fn(*args, **kwargs):
            return summary(func(*args, **kwargs))

        fn.__doc__ = func.__doc__

        return fn
