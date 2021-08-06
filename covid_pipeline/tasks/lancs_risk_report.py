"""Script which produces 7- and 14-day predictions of cases 
and prevalence in Lancashire LTLAs"""

import argparse
import numpy as np
import pandas as pd
import geopandas as gp

LANCS_LTLA = pd.DataFrame(
    data={
        "LTLA17CD": [
            "E06000008",
            "E06000009",
            "E07000117",
            "E07000118",
            "E07000119",
            "E07000120",
            "E07000121",
            "E07000122",
            "E07000123",
            "E07000124",
            "E07000125",
            "E07000126",
            "E07000127",
            "E07000128",
        ],
        "LTLA17NM": [
            "Blackburn with Darwen",
            "Blackpool",
            "Burnley",
            "Chorley",
            "Fylde",
            "Hyndburn",
            "Lancaster",
            "Pendle",
            "Preston",
            "Ribble Valley",
            "Rossendale",
            "South Ribble",
            "West Lancashire",
            "Wyre",
        ],
    }
)


def metadata(date):

    return pd.DataFrame(
        [
            ["Prediction date", date],
            ["lad19cd", "LTLA code, 2019"],
            ["lad19nm", "LTLA name, 2019"],
            ["popsize", "ONS predicted population size December 2019"],
            ["Rt_mean", f"Estimated reproduction number, {date}"],
            ["Rt_0.025", "Reproduction number lower 2.5% percentile"],
            ["Rt_0.975", "Reproduction number upper 97.5% percentile"],
            ["prev_mean", f"Estimated prevalence / 10000 people, {date}"],
            ["prev_0.025", f"Prevalence lower / 10000 people 2.5% percentile"],
            ["prev_0.975", f"Prevalence upper / 10000 people 97.5% percentile"],
            ["cases7_mean", f"Expected cases to {date+7}"],
            ["cases7_0.025", f"Expected cases to {date+7} 2.5% percentile"],
            ["cases7_0.975", f"Expected cases to {date+7} 97.5% percentile"],
            ["prev7_0.mean", f"Estimated prevalence / 10000 people on {date+7}"],
            [
                "prev7_0.025",
                f"Estimated prevalence / 10000 people on {date+7} 2.5% percentile",
            ],
            [
                "prev7_0.975",
                f"Estimated prevalence / 10000 people on {date+7} 97.5% percentile",
            ],
            ["cases14_mean", f"Expected cases to {date+14}"],
            ["cases14_0.025", f"Expected cases to {date+14} 2.5% percentile"],
            ["cases14_0.975", f"Expected cases to {date+14} 97.5% percentile"],
            ["cases21_mean", f"Expected cases to {date+21}"],
            ["cases21_0.025", f"Expected cases to {date+21} 2.5% percentile"],
            ["cases21_0.975", f"Expected cases to {date+21} 97.5% percentile"],
            ["cases28_mean", f"Expected cases to {date+28}"],
            ["cases28_0.025", f"Expected cases to {date+28} 2.5% percentile"],
            ["cases28_0.975", f"Expected cases to {date+28} 97.5% percentile"],
            ["prev14_mean", f"Estimated prevalence / 10000 people on {date+14}"],
            [
                "prev14_0.025",
                f"Estimated prevalence / 10000 people on {date+14} 2.5% percentile",
            ],
            [
                "prev14_0.975",
                f"Estimated prevalence / 10000 people on {date+14} 97.5% percentile",
            ],
            ["prev21_mean", f"Estimated prevalence / 10000 people on {date+21}"],
            [
                "prev21_0.025",
                f"Estimated prevalence / 10000 people on {date+21} 2.5% percentile",
            ],
            [
                "prev21_0.975",
                f"Estimated prevalence / 10000 people on {date+21} 97.5% percentile",
            ],
            ["prev28_mean", f"Estimated prevalence / 10000 people on {date+28}"],
            [
                "prev28_0.025",
                f"Estimated prevalence / 10000 people on {date+28} 2.5% percentile",
            ],
            [
                "prev28_0.975",
                f"Estimated prevalence / 10000 people on {date+28} 97.5% percentile",
            ],
        ]
    )


def report_lancs(source_gpkg, dest_xlsx, date):

    if not isinstance(date, np.datetime64):
        date = np.datetime64(date)
       
    gpkg = gp.read_file(source_gpkg)

    lancs = gpkg.loc[gpkg["lad19cd"].isin(LANCS_LTLA["LTLA17CD"])]
    lancs = lancs[
        [
            "lad19cd",
            "lad19nm",
            "popsize",
            "Rt_mean",
            "Rt_0.025",
            "Rt_0.975",
            "prev_mean",
            "prev_0.025",
            "prev_0.975",
            "cases7_mean",
            "cases7_0.025",
            "cases7_0.975",
            "prev7_mean",
            "prev7_0.025",
            "prev7_0.975",
            "cases14_mean",
            "cases14_0.025",
            "cases14_0.975",
            "prev14_mean",
            "prev14_0.025",
            "prev14_0.975",
            "cases21_mean",
            "cases21_0.025",
            "cases21_0.975",
            "prev21_mean",
            "prev21_0.025",
            "prev21_0.975",
            "cases28_mean",
            "cases28_0.025",
            "cases28_0.975",
            "prev28_mean",
            "prev28_0.025",
            "prev28_0.975",
        ]
    ]

    lancs.loc[:, lancs.columns.str.startswith("prev")] *= 10000

    print(f"Writing to '{dest_xlsx}'")
    with pd.ExcelWriter(dest_xlsx) as writer:
        metadata(date).to_excel(
            writer, sheet_name="metadata", index=False, header=False
        )
        lancs.to_excel(writer, sheet_name="prediction", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract Lancashire LTLAs from COVID-19 prediction GeoPkg"
    )
    parser.add_argument(
        "geopkg",
        type=str,
        nargs=1,
        help="path to GeoPackage file containing prediction.",
    )
    parser.add_argument("--output", dest="output", type=str, help="path to output xlsx")
    parser.add_argument(
        "--date", dest="date", type=np.datetime64, help="date of prediction"
    )
    args = parser.parse_args()

    report_lancs(args.geopkg[0], args.output, args.date)
