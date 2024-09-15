from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
)

from data_retrieval.consts import EVALSCRIPT_CLOUDLESS


def get_request_config():
    config_data = yaml.safe_load(open("config.yml"))

    config = SHConfig()

    config.sh_client_id = config_data["sh_client_id"]
    config.sh_client_secret = config_data["sh_client_secret"]
    config.sh_token_url = config_data["sh_token_url"]
    config.sh_base_url = config_data["sh_base_url"]

    return config


def get_request(config, year, bbox_coords, key='1'):
    time_interval = (datetime(year, 6, 1), datetime(year, 9, 1))
    epsg = 3035
    bbox = BBox(bbox_coords, CRS(4326)).transform(epsg)

    return SentinelHubRequest(
        evalscript=EVALSCRIPT_CLOUDLESS,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    "s2", service_url=config.sh_base_url
                ),
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        resolution=(10, 10),
        config=config,
        data_folder=f"./data/{key}",
    )


def download(bbox_coords=None, year_start=2015, year_end=2024, key='1'):
    if bbox_coords is None:
        bbox_coords = [14.880833, 54.044444, 14.95, 54.068889]
    config = get_request_config()

    sh_requests = {}
    for year in range(year_start, year_end):
        sh_requests[year] = get_request(config, year, bbox_coords, key)
    list_of_requests = [request.download_list[0] for request in sh_requests.values()]

    data = SentinelHubDownloadClient(config=config).download(
        list_of_requests, max_threads=5
    )

    Path(f"./data/{key}").mkdir(parents=True, exist_ok=True)
    for year, request in sh_requests.items():
        Path(request.data_folder, request.get_filename_list()[0]).rename(
            f"./data/{key}/{year}.tif"
        )

    return data


def add_time_dim(xda):
    # This pre-processes the file to add the correct
    # year from the filename as the time dimension
    year = int(Path(xda.encoding["source"]).stem)
    return xda.expand_dims(year=[year])


def load_local(key='1'):
    return xr.open_mfdataset(
        Path(f"./data/{key}").glob("*.tif"),
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
        ndvi = ds["band_4"][i].values
        ndvis.append(ndvi / 10_000)
    return years, images, ndvis


def load_local_images():
    ds = load_local()
    return get_images(ds)
