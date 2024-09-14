import numpy as np
from pathlib import Path
import getpass
from datetime import datetime
from pathlib import Path
import yaml

import requests
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipyleaflet import GeoJSON, Map, basemaps
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
)
import xarray as xr

TIFF_PATHS = Path("./data_retrieval/data").glob("*.tif")

def download():
    config_data = yaml.safe_load(open("config.yml"))
    config = SHConfig()
    config.sh_client_id = config_data.sh_client_id
    config.sh_client_secret = config_data.sh_client_secret
    config.sh_token_url = config_data.sh_token_url
    config.sh_base_url = config_data.sh_base_url

    # TODO
    pass


def add_time_dim(xda):
    # This pre-processes the file to add the correct
    # year from the filename as the time dimension
    year = int(Path(xda.encoding["source"]).stem)
    return xda.expand_dims(year=[year])


def load_local():
    return xr.open_mfdataset(
        TIFF_PATHS,
        engine="rasterio",
        preprocess=add_time_dim,
        band_as_variable=True,
    )


def get_images(ds):
    # returns years, images, ndvis
    years = ds["year"].values.tolist()
    images = []
    ndvis = []
    for i in range(len(years)):
        image = np.stack([ds["band_1"][i], ds["band_2"][i], ds["band_3"][i]], 0)
        images.append(np.transpose(image / image.max(), (1, 2, 0)))
        ndvis.append(ds["band_4"][i])
    return years, images, ndvis
