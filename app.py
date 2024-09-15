import folium as fl
from streamlit_folium import st_folium
import streamlit as st
import numpy as np
import pandas as pd
from data_retrieval.loader import load_local_images
from imgutils import color_grayscale_img
from img_processing import close_mask_clusters
from model import DetecTree
import cv2
import torch
from torchvision.utils import draw_segmentation_masks

@st.cache_data
def streamlit_cached_data():
    return load_local_images()

@st.cache_data
def get_images(lat, lon, vis_type):
    years, images, ndvis = streamlit_cached_data()
    print(images[0])
    if vis_type == 'NDVI':
        return color_grayscale_img(ndvis[0]), color_grayscale_img(ndvis[1])
    elif vis_type == 'Forest':
        masked_1 = overlap_mask(images[0])
        masked_2 = overlap_mask(images[1])
        return masked_1, masked_2
    else:
        return images[0], images[1]

@st.cache_data
def get_area_change(lat, lon):
    years, images, ndvis = streamlit_cached_data()

    return [
        (close_mask_clusters(DetecTree().predict_img(images[year][:, :, :3]), 3, 5, 5) == 255).mean()
        for year in range(len(images))
    ]



def overlap_mask(image):
    mask = close_mask_clusters(DetecTree().predict_img(image[:, :, :3]), 3, 5, 5)
    image = torch.tensor(image).permute(2, 0, 1)

    return draw_segmentation_masks(image, torch.tensor(mask == 255), alpha=0.3, colors='red').permute(1, 2, 0).numpy()


def show_images_comparison(lat, lon, vis_type):
    img1, img2 = get_images(lat, lon, vis_type)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Title 1")
        st.image(img1, caption="Image 1", use_column_width=True, clamp=True)
    with col2:
        st.subheader("Title 2")
        st.image(img2, caption="Image 2", use_column_width=True, clamp=True)

    area_data = pd.DataFrame(get_area_change(lat, lon), columns=["Forest Area"])
    st.area_chart(area_data)

    line_data = pd.DataFrame(np.random.randn(20, 2), columns=["Temperature", "CO2"])
    st.line_chart(line_data)


def main():
    bounds_poland = [[49.0, 14.0], [55.0, 24.0]]
    m = fl.Map(location=[52.0, 19.0], zoom_start=4, min_zoom=3, max_bounds=True)
    m.fit_bounds(bounds_poland)
    m.add_child(fl.LatLngPopup())

    # marked_points = get_important_points()

    # for point in marked_points:
    #     fl.Marker(
    #         location=[point['lat'], point['lon']],
    #         popup=point['name'],
    #         icon=fl.Icon(color='red', icon='info-sign')
    #     ).add_to(m)

    map = st_folium(m, height=400, width=700)

    point = None
    if map.get("last_clicked"):
        point = (map["last_clicked"]["lat"], map["last_clicked"]["lng"])

    if point is not None:
        vis_type = st.radio(
            "Choose visualization",
            ["Image", "NDVI", "Forest"],
            index=0,
        )
        lat, lon = point
        show_images_comparison(lat, lon, vis_type)


if __name__ == "__main__":
    main()
