from pathlib import Path
from covid_pipeline.tasks.summary_dha import summary_dha


# Test fixtures
basedir = Path("/scratch/hpc/39/jewellcp/covid_pipeline/2021-06-20_uk")

input_files = [
    "inferencedata.nc",
    "insample7.nc",
    "insample14.nc",
    "medium_term.nc",
    "reproduction_number.nc",
]
input_files = [basedir / f for f in input_files]

output_folders = {"web_folder_data": "dha/data/", "web_folder_js": "dha/js/"}
num_weeks = 8
ci_list = [0.05, 0.95]
gpkg_config = {
    "base_geopackage": "data/UK2019mod_pop.gpkg",
    "base_layer": "UK2019mod_pop_xgen",
}

# Test
summary_dha(input_files, output_folders, num_weeks, ci_list, gpkg_config)
