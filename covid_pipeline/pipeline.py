"""A Ruffus-ised pipeline for COVID-19 analysis"""

from os.path import expandvars
import yaml
import datetime
import ruffus as rf

from covid_pipeline.ruffus_pipeline import run_pipeline


def _import_global_config(config_file):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


if __name__ == "__main__":

    # Ruffus wrapper around argparse used to give us ruffus
    # cmd line switches as well as our own config
    argparser = rf.cmdline.get_argparse(description="COVID-19 pipeline")
    data_args = argparser.add_argument_group(
        "Data options", "Options controlling input data"
    )

    data_args.add_argument(
        "-c", "--config", type=str, help="global configuration file", required=True,
    )
    data_args.add_argument(
        "-r",
        "--results-directory",
        type=str,
        help="pipeline results directory",
        required=True,
    )
    data_args.add_argument(
        "--date-range",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
        nargs=2,
        help="Date range [low high)",
        metavar="ISO6801",
    )

    cli_options = argparser.parse_args()
    global_config = _import_global_config(cli_options.config)

    if cli_options.date_range is not None:
        global_config["ProcessData"]["date_range"][0] = cli_options.date_range[0]
        global_config["ProcessData"]["date_range"][1] = cli_options.date_range[1]

    run_pipeline(global_config, cli_options.results_directory, cli_options)
