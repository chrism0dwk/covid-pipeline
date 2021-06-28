#
# pip install git+https://gitlab.com/achale/dhaconfig.git#egg=dhaconfig
#
# This module needs config["base_layer"]=config["UK2019mod_pop_xgen"]
#
# ci_list = [0.05, 0.95]; num_weeks = 8; output_folder="C:/inetpub/wwwroot/COVID19UK/temp/"; url="https://fhm-chicas-storage.lancs.ac.uk/bayesstm/latest/"
# input_path = "H:/Downloads/2021-06-17_uk/"; input_files = [input_path + "inferencedata.nc", input_path + "insample7.nc", input_path + "insample14.nc", input_path + "medium_term.nc", input_path + "reproduction_number.nc"]
# config = {"base_geopackage":"data/UK2019mod_pop.gpkg", "base_layer":"UK2019mod_pop_xgen"}


from covid_pipeline.tasks.summary_longformat import xarray2summarydf
from covid_pipeline.tasks.summary_longformat import prevalence
from covid_pipeline.tasks import case_exceedance

from pathlib import Path

import datetime
import dhaconfig
import geopandas as gp
import json
import numpy as np
import pandas as pd
import re
import shapely  # shapely can be omitted on Linux?
import xarray

utils = dhaconfig.utilities()
dha = dhaconfig.dha()


def dha_format_dict():
    colour_scheme_a = [
        "#fff7db",
        "#ffeda0",
        "#fed976",
        "#fd8d3c",
        "#fc4e2a",
        "#e31a1c",
        "#b10026",
        "#800026",
    ]
    colour_scheme_b = [
        "#181A49",
        "#313695",
        "#4575b4",
        "#74add1",
        "#e0f3f8",
        "#ffffb2",
        "#fed976",
        "#feb24c",
        "#fd8d3c",
        "#fc4e2a",
        "#e31a1c",
        "#b10026",
        "#72001a",
    ]
    colour_scheme_c = ["#5e4fa2", "#66c2a5", "#ffffbf", "#fdae61", "#d7191c"]
    userDefined_scheme_b = [
        0,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
    ]
    userDefined_scheme_c = [0, 0.03, 0.05, 0.95, 0.97, 1]
    dha_format = {
        "Daily_case_incidence": {
            "colgrades_colours": colour_scheme_a,
            "colgrades_Ints": "15",
            "colgrades_userDefined": "undefined",
            "colgrades_legtitle": "undefined",
            "timeData_timeseries": "true",
            "timePlot_ymax": "undefined",
            "popupBox": "false",
        },
        "Cumulative_case_incidence": {
            "colgrades_colours": colour_scheme_a,
            "colgrades_Ints": "200",
            "colgrades_userDefined": "undefined",
            "colgrades_legtitle": "undefined",
            "timeData_timeseries": "true",
            "timePlot_ymax": "undefined",
            "popupBox": "false",
        },
        "Daily_case_incidence_per_100k": {
            "colgrades_colours": colour_scheme_a,
            "colgrades_Ints": "10",
            "colgrades_userDefined": "undefined",
            "colgrades_legtitle": "undefined",
            "timeData_timeseries": "true",
            "timePlot_ymax": "undefined",
            "popupBox": "false",
        },
        "Prevalence_per_100k": {
            "colgrades_colours": colour_scheme_a,
            "colgrades_Ints": "75",
            "colgrades_userDefined": "undefined",
            "colgrades_legtitle": "undefined",
            "timeData_timeseries": "true",
            "timePlot_ymax": "undefined",
            "popupBox": "false",
        },
        "Reproduction_number_Rt": {
            "colgrades_colours": colour_scheme_b,
            "colgrades_Ints": "0.1",
            "colgrades_userDefined": userDefined_scheme_b,
            "colgrades_legtitle": "undefined",
            "timeData_timeseries": "true",
            "timePlot_ymax": "undefined",
            "popupBox": "false",
        },
        "Prob_Rt_exceeds_1_0": {
            "colgrades_colours": colour_scheme_c,
            "colgrades_Ints": "0.1",
            "colgrades_userDefined": userDefined_scheme_c,
            "colgrades_legtitle": "Pr(R<sub>t</sub> &#62; 1.0)",
            "timeData_timeseries": "true",
            "timePlot_ymax": "1",
            "popupBox": "false",
        },
        "Insample_7_days": {
            "colgrades_colours": colour_scheme_c,
            "colgrades_Ints": "0.1",
            "colgrades_userDefined": userDefined_scheme_c,
            "colgrades_legtitle": "Pr(pred &#60; obs)",
            "timeData_timeseries": "false",
            "timePlot_ymax": "1",
            "popupBox": "true",
        },
        "Insample_14_days": {
            "colgrades_colours": colour_scheme_c,
            "colgrades_Ints": "0.1",
            "colgrades_userDefined": userDefined_scheme_c,
            "colgrades_legtitle": "Pr(pred &#60; obs)",
            "timeData_timeseries": "false",
            "timePlot_ymax": "1",
            "popupBox": "true",
        },
    }
    return dha_format


def summarydf_tidy(df, dp, num_weeks, cis):
    """
    :param df: (dataframe)
    :param dp: (int) number of decimal places
    :param num_weeks: (int) number of weeks
    :param cis: (list) list of quantile values e.g. [0.05, 0.95] or None
    :return: (dataframe)
    """
    if cis is None:
        df = df.loc[:, ["location", "time", "value"]]
        df = df.round({"value": dp})
        df = df.rename({"value": "mean"}, axis="columns")
    else:
        df = df.loc[:, ["location", "time", "value", str(cis[0]), str(cis[1])]]
        df = df.round({"value": dp, str(cis[0]): dp, str(cis[1]): dp})
        df = df.rename(
            {"value": "mean", str(cis[0]): ".L", str(cis[1]): ".U"},
            axis="columns",
        )
    df["idx"] = df["time"]
    if len(df["time"].unique()) > 1:
        start_date = min(df.time) + datetime.timedelta(days=1)
        times = [
            start_date + datetime.timedelta(days=x)
            for x in range(0, 7 * (num_weeks + 1), 7)
        ]
        df = df[df["time"].isin(times)]
    else:
        times = [df.iloc[0]["time"]]
    di = dict(zip(times, list(range(len(times)))))
    df = df.replace({"idx": di})
    return df.reset_index()


def insample_tidy(df, dp, num_days, cis, cases):
    """
    :param df: (dataframe)
    :param dp: (int) number of decimal places
    :param num_days: (int) number of weeks
    :param cis: (list) list of quantile values e.g. [0.05, 0.95] or None
    :param cases: (dataframe) actual number of recorded cases per day
    :return: (dataframe)
    """
    df = df.loc[:, ["location", "time", "value", str(cis[0]), str(cis[1])]]
    df = df.round({"value": dp, str(cis[0]): dp, str(cis[1]): dp})
    times = [min(df.time) + datetime.timedelta(days=x) for x in range(num_days)]
    df = df[df["time"].isin(times)]
    return pd.merge(
        pd.DataFrame(df),
        pd.DataFrame(cases),
        how="left",
        on=["location", "time"],
    )


def dflong2wide(df, cis):
    """
    :param df: (dataframe) wide format
    :param cis: (list) list of quantile values e.g. [0.05, 0.95] or None
    :return: (dataframe) long format
    """

    def subset_df(dx, col_name, postfix):
        dy = dx.loc[:, ["location", "idx", col_name]]
        dy["idx"] = dy["idx"].astype(str) + postfix
        dy = dy.rename({col_name: "value"}, axis="columns")
        dy = dy.groupby(["location", "idx"]).sum().unstack("idx")
        dy.columns = dy.columns.droplevel()
        return dy.reset_index()

    dd = subset_df(df, "mean", "")
    if cis is not None:
        dl = subset_df(df, ".L", ".L")
        du = subset_df(df, ".U", ".U")
        dj = dl.join(du.set_index("location"), "location")
        dd = dd.join(dj.set_index("location"), "location")
    dd = dd.sort_values("location")
    return dd.rename({"location": "lad19cd"}, axis="columns")


def make_geodf(df_geo, df):
    """
    :param df_geo: (geodataframe)
    :param df: (dataframe)
    :return: (geodataframe)
    """
    spdf = df_geo[df_geo["lad19cd"].isin(np.array(df["lad19cd"]))]
    spdf = spdf.sort_values(by="lad19cd")
    spdf = spdf.merge(df, on="lad19cd")
    spdf["lad19cd"] = spdf["lad19cd"].str.replace(",", "_")
    return spdf.rename(
        {"lad19cd": "LAD code", "lad19nm": "LAD name"}, axis="columns"
    )


def write_csv(x, cis, folder, file_name):
    """
    :param x: (dataframe) thing to be saved as csv
    :param cis: (list) list of quantile values e.g. [0.05, 0.95] or None
    :param folder: (str) output folder for geojson files e.g. "z:/dha_website_root/data/"
    :param file_name: (str) file name of csv
    """
    if cis is None:
        y = x.loc[:, ["location", "time", "mean"]]
        y = y.rename({"mean": "value"}, axis="columns")
    else:
        y = x.loc[:, ["location", "time", "mean", ".L", ".U"]]
        y = y.rename(
            {".L": str(cis[0]) + " quantile", ".U": str(cis[1]) + " quantile"},
            axis="columns",
        )
    y = pd.DataFrame(y)
    y.to_csv(folder + file_name + ".csv")


def write_xls(df1, df2, web_folder_data, name):
    """
    :param df1: (dataframe) insample dataframe
    :param df2: (dataframe) case exceedance dataframe
    :param web_folder_data: (str) output folder for geojson files e.g. "z:/dha_website_root/data/"
    :parma name: (str) name of layer
    """
    writer = pd.ExcelWriter(web_folder_data + name + ".xlsx", engine="openpyxl")
    df1.to_excel(writer, sheet_name="Insample")
    df2.to_excel(writer, sheet_name="Pr(pred<obs)")
    writer.save()


def insample_write_json(df, web_folder_data, ci_list, name):
    """
    :param df: (dataframe)
    :param web_folder_data: (str) output folder for geojson files e.g. "z:/dha_website_root/data/"
    :param ci_list: (list) list of quantile values e.g. [0.05, 0.95] or None
    :parma name: (str) name of layer
    """
    lst = (
        '{"labels":'
        + json.dumps(
            (
                pd.DatetimeIndex(list(df.loc[:, "time"].unique())).strftime(
                    "%d %b"
                )
            ).to_list()
        )
        + ","
    )
    lst = (
        lst
        + '"dateFrom":"'
        + min(df.loc[:, "time"]).strftime("%d %b %Y")
        + '",'
    )
    locations = df["LAD code"].unique()
    for x in locations:
        dd = df[df["LAD code"] == str(x)]
        lst = (
            lst + '"' + x + '": {'
            '"LADname":"' + dd.iloc[0]["LAD name"] + '",'
            '"mean":' + str(dd["value"].tolist()) + ","
            '"quantLo":' + str(dd[str(ci_list[0])].tolist()) + ","
            '"quantUp":' + str(dd[str(ci_list[1])].tolist()) + ","
            '"cases":' + str(dd["cases"].tolist())
        )
        lst_line_end = "}" if x == locations[len(locations) - 1] else "},"
        lst = lst + lst_line_end
    lst = lst + "}"
    lst = lst.replace(", ", ",")
    utils.write_text_file(web_folder_data, name + ".json", lst)


def make_layer(web_folder_data, name, dates, cis, pars, xlsx, url):
    """
    :param web_folder_data: (str) output folder for geojson files e.g. "z:/dha_website_root/data/"
    :parma name: (str) name of layer
    :param dates: (list) list of dates
    :param cis: (list) list of quantiles e.g. [0.05, 0.95]
    :parma pars: (dict) dictionary of dha parameters: for details see dhaconfig at https://gitlab.com/achale/dhaconfig.git#egg=dhaconfig
    :param xlsx: (boolean) when False downloadData is a csv otherwise xlsx
    :param url: (str) url of json and geojson data if it is external to the web server
    :return: (str) dha map layer as dictionary
    """
    downloadData_ext = ".xlsx" if xlsx else ".csv"
    ci_names = (
        " " if cis is None else str(cis[0]) + "-" + str(cis[1]) + " quantiles"
    )
    layer = dha.build_single_layer(
        geoJsonFile=url + "data/" + name + ".geojson",
        friendlyName=name.replace("_", " ").replace("1 0", "1.0"),
        radioButtonValue=name.replace("_", " ").replace("1 0", "1.0")
        + dates[0].strftime(" %d %b"),
        layerName=str(
            (
                name.replace("_", " ").replace("1 0", "1.0")
                + dates.strftime(" %d %b")
            ).to_list()
        ),
        geojsonName=str(list(map(str, range(len(dates))))),
        geojsonGeom="LAD name",
        geojsonExtraInfo="LAD code",
        mapPosition_centerLatLng=str([55.5, -2.7]),
        mapPosition_zoom="5.5",
        regionNames_country="Click region in UK to view its graph",
        regionNames_area="LAD name",
        colgrades_colours=str(pars["colgrades_colours"]),
        colgrades_legtitle=pars["colgrades_legtitle"],
        colgrades_Ints=pars["colgrades_Ints"],
        colgrades_Inis="0",
        colgrades_Num="undefined",
        colgrades_userDefined=str(pars["colgrades_userDefined"]),
        legend="true",
        sliderlabel="undefined",
        mapStyles_weight="1",
        mapStyles_opacity="undefined",
        mapStyles_color="#000",
        mapStyles_fillOpacity="0.8",
        mapStyles_smoothFactor="1",
        mapStyles_radius="6",
        noDataColour="rgba(0, 0, 0, 0.3)",
        featurehltStyle_weight="3",
        featurehltStyle_color="#000",
        timeData_xlabs=str((dates.strftime("%d %b")).to_list()),
        timeData_timeseries=pars["timeData_timeseries"],
        timeData_CIname=ci_names,
        timeData_highlight="undefined",
        timeData_timeseriesMin="0",
        timeData_timeseriesMax=str(len(dates) - 1),
        timeData_timeseriesStep="1",
        meandata=(str(["null" for x in range(len(dates))])).replace("'", ""),
        timePlot_Background1Colour="#FFF",
        timePlot_Line1Colour="#FFF",
        timePlot_Background2Colour="#007ac3",
        timePlot_Line2Colour="#007ac3",
        timePlot_Background3Colour="undefined",
        timePlot_Line3Colour="undefined",
        timePlot_MarkerSize="3",
        timePlot_HighlightColour="#b41019",
        timePlot_HighlightSize="5",
        timePlot_ymax=pars["timePlot_ymax"],
        timePlot_beginYAtZero="true",
        layerMarker="undefined",
        mapBoundary="undefined",
        units_html=name.replace("_", " ").replace("1 0", "1.0"),
        units_unicode="predicted "
        + name.replace("_", " ").lower().replace("1 0", "1.0"),
        units_xlab="date from " + dates[0].strftime("%d %b %Y"),
        downloadData=url + "data/" + name + downloadData_ext,
        popupBox=pars["popupBox"],
    )
    return layer


def geo_round(match):
    return "{:.4f}".format(float(match.group()))


def do_dha_things(
    df,
    num_weeks,
    cis,
    geo,
    web_folder_data,
    name,
    dha_format,
    dp,
    url,
    extra_df=None,
):
    """
    :param df: (dataframe)
    :param num_weeks: (int) number of weeks into the future for predictions: currently only works if num_weeks<10, if num_weeks>9 dataframe will be sorted incorrectly in dfwide2long()
    :param cis: (list) list of quantiles e.g. [0.05, 0.95]
    :geo: (geodataframe)
    :param web_folder_data: (str) output folder for geojson files e.g. "z:/dha_website_root/data/"
    :parma name: (str) name of layer
    :parma dha_format: (dict) dictionary of dha parameters: for details see dhaconfig at https://gitlab.com/achale/dhaconfig.git#egg=dhaconfig
    :param dp: (int) number of decimal places to keep
    :param url: (str) url of json and geojson data if it is external to the web server
    :param extra_df: (dataframe) additional dataframe for xlsx
    :return: (str) dha map layer as dictionary
    """
    df_tidy = summarydf_tidy(df, dp, num_weeks, cis)
    df_wide = dflong2wide(df_tidy, cis)
    geodf = make_geodf(geo, df_wide)
    utils.write_geojson(
        web_folder_data, name + ".geojson", geodf
    )  # remove bbox from geojson?
    if extra_df is None:
        write_csv(df_tidy, cis, web_folder_data, name)
        xlsx = False
    else:
        extra_df = extra_df.rename({"value": "mean"}, axis="columns")
        write_xls(extra_df, df, web_folder_data, name)
        xlsx = True
    dates = pd.DatetimeIndex(df_tidy["time"].unique())
    return make_layer(web_folder_data, name, dates, cis, dha_format, xlsx, url)


def summary_dha(input_files, output_folder, num_weeks, ci_list, config, url=""):
    """Draws together pipeline results into files for DHA

    :param input_files: (list) filename list [inferencedata_nc,
                                              insample7_nc
                                              insample14_nc,
                                              medium_term_nc,
                                              reproduction_number_nc]
    :param output_folder: (str) output folder e.g. "Z:/folder/"
    :param num_weeks: (int) number of weeks for predictions (untested for over 9 so beware ordering might break)
    :param ci_list: (list) list of quantiles e.g. [0.05, 0.95]
    :param config: SummaryGeopackage configuration information
    :param url: (str) url of json and geojson data if it is external to DHA host web server.
    """

    # initialise
    shapely.speedups.disable()  # this line can be omitted on Linux
    data = xarray.open_dataset(input_files[0], group="constant_data")
    cases = xarray.open_dataset(input_files[0], group="observations")["cases"]
    cases = cases.to_dataframe().reset_index()
    dha_format = dha_format_dict()
    layers = {}
    output_folders = {"web_folder_data": output_folder + "data/",
                      "web_folder_js": output_folder + "js/"}
    Path(output_folders["web_folder_data"]).mkdir(parents=True, exist_ok=True)
    Path(output_folders["web_folder_js"]).mkdir(parents=True, exist_ok=True)

    # geopackage: load, select, transform and round
    dec = re.compile(r"\d*\.\d+")
    geo = gp.read_file(config["base_geopackage"], layer=config["base_layer"])
    geo = geo.loc[:, ["geometry", "lad19cd", "lad19nm"]]
    geo = geo.to_crs("epsg:4326")  # geo.plot()
    geo.geometry = geo.geometry.apply(
        lambda x: shapely.wkt.loads(re.sub(dec, geo_round, x.wkt))
    )

    # Medium term absolute incidence
    name = default_name = "Daily_case_incidence"
    medium_term = xarray.open_dataset(input_files[3], group="predictions")
    medium_df = xarray2summarydf(
        medium_term["events"][..., 2].reset_coords(drop=True)
    )
    layers[name] = do_dha_things(
        medium_df,
        num_weeks,
        ci_list,
        geo,
        output_folders["web_folder_data"],
        name,
        dha_format[name],
        1,
        url,
    )

    # Cumulative cases
    name = "Cumulative_case_incidence"
    medium_df = xarray2summarydf(
        medium_term["events"][..., 2].cumsum(dim="time").reset_coords(drop=True)
    )
    layers[name] = do_dha_things(
        medium_df,
        num_weeks,
        ci_list,
        geo,
        output_folders["web_folder_data"],
        name,
        dha_format[name],
        1,
        url,
    )

    # Medium term incidence per 100k
    name = "Daily_case_incidence_per_100k"
    medium_df = xarray2summarydf(
        (
            medium_term["events"][..., 2].reset_coords(drop=True)
            / np.array(data["N"])[np.newaxis, :, np.newaxis]
        )
        * 100000
    )
    layers[name] = do_dha_things(
        medium_df,
        num_weeks,
        ci_list,
        geo,
        output_folders["web_folder_data"],
        name,
        dha_format[name],
        1,
        url,
    )

    # Medium term prevalence
    name = "Prevalence_per_100k"
    prev_df = prevalence(medium_term, data["N"])
    layers[name] = do_dha_things(
        prev_df,
        num_weeks,
        ci_list,
        geo,
        output_folders["web_folder_data"],
        name,
        dha_format[name],
        1,
        url,
    )

    # Rt
    name = "Reproduction_number_Rt"
    # del medium_term
    r_it = xarray.load_dataset(input_files[4], group="posterior_predictive")[
        "R_it"
    ]
    rt_summary = xarray2summarydf(r_it.isel(time=-1))
    rt_summary["time"] = r_it.coords["time"].data[-1] + np.timedelta64(1, "D")
    default_time = rt_summary.iloc[0]["time"]
    layers[name] = do_dha_things(
        rt_summary,
        0,
        ci_list,
        geo,
        output_folders["web_folder_data"],
        name,
        dha_format[name],
        2,
        url,
    )

    # Prob(Rt>1)
    name = "Prob_Rt_exceeds_1_0"
    rt = r_it.isel(time=-1).drop("time")
    rt_exceed = np.mean(rt > 1.0, axis=0)
    rt_exceed = (
        rt_exceed.to_dataframe()
        .reset_index()
        .rename({"R_it": "value"}, axis="columns")
        .round({"value": 2})
    )
    rt_exceed["time"] = default_time
    layers[name] = do_dha_things(
        rt_exceed,
        0,
        None,
        geo,
        output_folders["web_folder_data"],
        name,
        dha_format[name],
        2,
        url,
    )

    # Case exceedance - used by Insample
    exceed7 = case_exceedance((input_files[0], input_files[1]), 7)
    exceed14 = case_exceedance((input_files[0], input_files[2]), 14)
    case_exceed = pd.DataFrame(
        {"Pr(pred<obs)_7": exceed7, "Pr(pred<obs)_14": exceed14},
        index=exceed7.coords["location"],
    )
    case_exceed = (
        case_exceed.reset_index()
    )  # in line above if location is the index then this line is needed?
    # del r_it; case_exceed=pd.read_csv('H:/Downloads/2021-06-17_uk/exceedance_summary.csv')
    case_exceed["time"] = default_time

    # Insample predictive incidence
    name = "Insample_7_days"
    insample = xarray.open_dataset(
        input_files[1], group="predictions"
    )  # insample7_Cases
    insample_df = xarray2summarydf(
        insample["events"][..., 2].reset_coords(drop=True)
    )
    # del insample; cases = xarray.open_dataset(input_files[0], group="observations")["cases"]; cases = cases.to_dataframe().reset_index()
    insample_df_tidy = insample_tidy(insample_df, 1, 7, ci_list, cases)
    startdate7 = pd.DatetimeIndex(insample_df_tidy["time"])[0].strftime(
        " %d %b"
    )
    geodf = make_geodf(
        geo, insample_df_tidy.rename({"location": "lad19cd"}, axis="columns")
    )
    insample_write_json(
        geodf.drop(columns="geometry"),
        output_folders["web_folder_data"],
        ci_list,
        name,
    )
    case_exceed7 = case_exceed.drop("Pr(pred<obs)_14", 1).rename(
        {"Pr(pred<obs)_7": "value"}, axis="columns"
    )
    layers[name] = do_dha_things(
        case_exceed7,
        0,
        None,
        geo,
        output_folders["web_folder_data"],
        name,
        dha_format[name],
        2,
        url,
        insample_df_tidy,
    )

    name = "Insample_14_days"
    insample = xarray.open_dataset(
        input_files[2], group="predictions"
    )  # insample14_Cases
    insample_df = xarray2summarydf(
        insample["events"][..., 2].reset_coords(drop=True)
    )
    insample_df_tidy = insample_tidy(insample_df, 1, 14, ci_list, cases)
    startdate14 = pd.DatetimeIndex(insample_df_tidy["time"])[0].strftime(
        " %d %b"
    )
    geodf = make_geodf(
        geo, insample_df_tidy.rename({"location": "lad19cd"}, axis="columns")
    )
    insample_write_json(
        geodf.drop(columns="geometry"),
        output_folders["web_folder_data"],
        ci_list,
        name,
    )
    case_exceed14 = case_exceed.drop("Pr(pred<obs)_7", 1).rename(
        {"Pr(pred<obs)_14": "value"}, axis="columns"
    )
    layers[name] = do_dha_things(
        case_exceed14,
        0,
        None,
        geo,
        output_folders["web_folder_data"],
        name,
        dha_format[name],
        2,
        url,
        insample_df_tidy,
    )

    # build config file
    postfix = pd.DatetimeIndex([default_time])[0].strftime(" %d %b")
    config_data = dha.event_listener(
        default_name.replace("_", " ") + postfix,
        postfix=postfix,
        startdate7=startdate7,
        startdate14=startdate14,
    )
    config_data += dha.global_vars(
        radioNotDropdown="false",
        InitialMapCenterLatLng="[55.5, -2.7]",
        InitialmapZoom="5.5",
        mapUnits="false",
        secondMap="true",
        userDefinedBaseMaps="true",
    )
    config_data += dha.build_list_of_layers(
        [
            layers["Reproduction_number_Rt"],
            layers["Prob_Rt_exceeds_1_0"],
            layers["Prevalence_per_100k"],  # xxx
            layers["Daily_case_incidence"],
            layers["Daily_case_incidence_per_100k"],  # xxx
            layers["Cumulative_case_incidence"],
            layers["Insample_7_days"],
            layers["Insample_14_days"],
        ]
    )
    utils.write_text_file(
        output_folders["web_folder_js"], "allMetadata.js", config_data
    )

    # write file containing the current time-date
    utils.write_last_updated_time(
        output_folders["web_folder_js"], "lastupdated.js", "last updated: "
    )

