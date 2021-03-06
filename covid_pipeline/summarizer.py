"""Summarizer provides methods to summarise `covid19uk` output"""

import pickle as pkl
from pathlib import Path
from warnings import warn
import xarray
import numpy as np

from gemlib.util import compute_state as gl_compute_state
from covid19uk import model_spec

# Filenames within MCMC results directory
CONSTANT_DATA = "inferencedata.nc"
OBSERVATION_DATA = "inferencedata.nc"
REPRODUCTION_NUMBER = "reproduction_number.nc"
SAMPLES = "thin_samples.pkl"
PREDICTIVE_CASES = "medium_term.nc"
WITHIN_BETWEEN = "within_between_summary.csv"

# Location within results directory
GROUPS = {
    "CONSTANT_DATA": "constant_data",
    "OBSERVATION_DATA": "observations",
    "REPRODUCTION_NUMBER": "posterior_predictive",
    "PREDICTIVE_CASES": "predictions",
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
    spatial_samples = xarray.DataArray(
        samples["spatial_effect"],
        coords=[
            np.arange(samples["spatial_effect"].shape[0]),
            constant_data.coords["location"],
        ],
        dims=["iteration", "location"],
    )

    return xarray.Dataset(
        {
            "seir": event_samples,
            "initial_state": initial_state,
            "spatial_effect": spatial_samples,
        }
    )


class PosteriorMetrics:
    def __init__(
        self, results_directory, aggregate=False, filter_location=None
    ):
        """Summarizer attaches to a `covid19uk` results directory and summarizes
           the output.
        :param results_directory: string giving the path of the results directory
        :param mean: should the mean be included in the summary?
        :param quantiles: tuple of quantiles to output in summary
        :param aggregate: should the metrics be aggregated (summed) across locations?
        """
        self._path = Path(results_directory)
        self._aggregate = True if aggregate is True else False
        self._filter_location_value = filter_location

    def _filter_location(self, data_array):
        if self._filter_location is None:
            return data_array
        is_location = data_array.coords["location"].str.startswith(
            self._filter_location_value
        )
        return data_array.sel(location=is_location)

    @property
    def _constant_data_unfiltered(self):
        """Returns an `xarray.Dataset` of covariate data"""
        return xarray.open_dataset(
            str(self._path / CONSTANT_DATA), group=GROUPS["CONSTANT_DATA"]
        )

    @property
    def _constant_data(self):
        return self._filter_location(self._constant_data_unfiltered)

    @property
    def _observations(self):
        """Returns an `xarray.Dataset` of observation data"""
        return xarray.open_dataset(
            str(self._path / OBSERVATION_DATA), group=GROUPS["OBSERVATION_DATA"]
        )

    def _maybe_aggregate(self, x):
        if self._aggregate:
            return x.sum(dim="location")
        return x


class PosteriorFunctions(PosteriorMetrics):
    """Summarizer provides methods to summarize `covid19uk` output"""

    @property
    def _samples(self):
        """Returns a dictionary of MCMC samples"""
        with open(self._path / SAMPLES, "rb") as f:
            samples = pkl.load(f)

        samples = _events2xarray(samples, self._constant_data_unfiltered)
        samples = self._filter_location(samples)
        return samples

    def _compute_state(self):
        """Reconstructs the state of the population a each timepoint"""
        event_samples = self._samples
        state = gl_compute_state(
            event_samples["initial_state"],
            event_samples["seir"],
            model_spec.STOICHIOMETRY,
        ).numpy()
        return xarray.DataArray(
            state,
            coords=[
                np.arange(state.shape[0]),
                self._constant_data.coords["location"],
                self._constant_data.coords["time"],
                np.arange(state.shape[-1]),
            ],
            dims=["iteration", "location", "time", "state"],
        )

    def rt(self):
        """Return Rt summary as a `xarray.Dataset`"""
        rt = xarray.open_dataset(
            self._path / REPRODUCTION_NUMBER,
            group=GROUPS["REPRODUCTION_NUMBER"],
        )
        if self._aggregate:
            return rt["R_t"]
        return rt["R_it"]

    def rt_exceed(self, r=1.0):
        """Returns the probability that Rt exceeds r, Pr(Rt > r)."""
        rt = self.rt()
        exceedance = (rt > r).mean(dim="iteration")
        return exceedance.rename(f"Pr(Rt>{r})")

    def absolute_incidence(self):
        """Return `xarray.Dataset` with daily absolute
        infection incidence samples"""
        event_samples = self._samples
        infection_events = (
            event_samples["seir"].sel(event=0).reset_coords(drop=True)
        )
        return self._maybe_aggregate(infection_events)

    def relative_incidence(self):
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

    def case_exceedance(self, lag=7):
        observed = self._maybe_aggregate(
            self._observations["cases"]
            .isel(time=slice(-lag, None))
            .sum(dim="time")
        )
        predicted = (
            self.absolute_incidence()
            .isel(time=slice(-lag, None))
            .sum(dim="time")
        )
        return (
            (observed > predicted)
            .mean(dim="iteration")
            .rename(f"Pr(obs[-{lag}:]>pred)")
        )

    def spatial_exceed(self, x=0.0):
        """Returns the probability of spatial random effect
           exceeding x on the log scale.
        :param x: an exceedance value on the log scale.
        :returns: a `xarray.DataArray` containing Pr(spatial_effect > x).
        """
        samples = self._samples["spatial_effect"]
        spatial_samples = xarray.DataArray(
            samples,
            coords=[
                np.arange(samples.shape[0]),
                self._constant_data.coords["location"],
            ],
            dims=["iteration", "location"],
        )
        return (
            (spatial_samples > x)
            .mean(dim="iteration")
            .rename(f"Pr(spatial_effect>{x}")
        )


class PosteriorPredictiveFunctions(PosteriorFunctions):
    """Provides epidemiological metrics on posterior predictive
    distribution."""

    @property
    def _predicted(self):
        arr = xarray.open_dataset(
            str(self._path / PREDICTIVE_CASES), group=GROUPS["PREDICTIVE_CASES"]
        )
        return self._filter_location(arr)

    def _compute_state(self):
        state = gl_compute_state(
            self._predicted["initial_state"],
            self._predicted["events"],
            model_spec.STOICHIOMETRY,
        ).numpy()
        return xarray.DataArray(
            state,
            coords=[
                np.arange(state.shape[0]),
                self._constant_data.coords["location"],
                self._constant_data.coords["time"],
                np.arange(state.shape[-1]),
            ],
            dims=["iteration", "location", "time", "state"],
        )

    def absolute_incidence(self):
        """Returns absolute predicted absolute incidence"""
        return self._maybe_aggregate(self._predicted["events"].sel(event=0))

    def cumulative_absolute_incidence(self):
        """Returns predicted cumulative absolute incidence"""
        return self._maybe_aggregate(self.absolute_incidence.cumsum(dim="time"))

    def relative_incidence(self):
        """Returns predicted relative incidence"""
        population_size = self._maybe_aggregate(self._constant_data["N"])
        return self.absolute_incidence() / population_size

    def prevalence(self):
        state = self._compute_state()
        state = state.isel(state=slice(1, 3)).sum(dim="state")
        return self._maybe_aggregate(state) / self._maybe_aggregate(
            self._constant_data["N"]
        )


def make_summary(
    mean=True,
    quantiles=(0.025, 0.975),
    dim="iteration",
):
    """Make a summarisation function

    :param mean: should the mean be included?
    :param quantiles: requests quantiles to be summarised
    :param dim: dimension to summarize across.
    :returns: an xarray.Dataset containing arrays with mean, quantiles.

    Usage:
    ```
    from covid_pipeline import PosteriorPredictiveFunctions
    from covid_pipeline import make_summary
    posterior_predictive = PosteriorPredictiveFunctions("/path/to/results", aggegate=False)
    summary = make_summary(mean=True, quantiles=(0.025, 0.5, 0.975))
    summary_xarr = summary(posterior_predictive.relative_incidence())
    summary_xarr.to_dataframe() # Format at pandas dataframe
    ```
    """

    def fn(samples):
        data_arrays = {}
        if mean is True:
            data_arrays["mean"] = samples.mean(dim=dim)
        for q in quantiles:
            data_arrays[f"q{q}"] = samples.quantile(q=q, dim=dim).reset_coords(
                drop=True
            )
        return xarray.Dataset(data_arrays)

    return fn
