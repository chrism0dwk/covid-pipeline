# Lancaster Stochastic SEIR COVID model pipeline

This repository contains pipeline code to run the [Lancaster University
Bayesian stochastic SEIR COVID19 model for the UK](https://github.com/chrism0dwk/covid19uk).

## Running the Docker container

The recommended method of running the pipeline is as a Docker container which may be bound to a local results directory, local config file, and optionally takes a `--date-range` parameter.  

For example, to run pipeline version `v0.1.1-alpha.2` over an 84 day time period ending 3 days ago, and saving results to a local directory `/local_results` use:

```
$ docker run --rm \
    -v /local_results:/results chrism0dwk/lancs-covid-pipeline:0.1.1-alpha.2 \
    --date-range `date -d "-87 days" +%F` `date -d "-3 days" +%F` 
```

The latest version of the pipeline may be determined from the [Dockerhub](https://hub.docker.com/repository/docker/chrism0dwk/lancs-covid-bayesstm) repository.

The default `config.yaml` file is set up to configure the MCMC for an approximate 84 day window.  However, the file may be edited and supplied to the container like so:
```
$ docker run --rm \
    -v /local_results:/results chrism0dwk/lancs-covid-pipeline:0.1.1-alpha.2 \
    -v /absolute/path/to/config.yaml:/app/config.yaml \
	--date-range `date -d "-87 days" +%F` `date -d "-3 days" +%F` 
```

__Note__: The Monte Carlo algorithms run by the pipeline are computationally intensive.  Using an 84 day time window and an NVIDIA Tesla V100 GPU accelerator card, expect approximately 11 hours runtime.

## Running the pipeline from the source repository

The pipeline may, for development purposes, be run from the source repository.  The pipeline requires [Poetry](https://python-poetry.org) and optionally a CUDA installation for GPU use (very much recommended!).

Once Poetry (and CUDA) is installed, package dependencies may be installed via poetry. 

```
$ git clone https://github.com/chrism0dwk/covid-pipeline.git
$ cd covid-pipeline
$ poetry install
```

To run the pipeline
```
$ poetry run python -m covid_pipeline.pipeline \
    --config config.yaml \
	--results-directory /path/to/results/directory \
	--date-range  `date -d "-87 days" +%F` `date -d "-3 days" +%F`
```

__Note__: The Monte Carlo algorithms run by the pipeline are computationally intensive.  Using an 84 day time window and an NVIDIA Tesla V100 GPU accelerator card, expect approximately 11 hours runtime.

