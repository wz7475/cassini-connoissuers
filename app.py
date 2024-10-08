import folium as fl
from streamlit_folium import st_folium
import streamlit as st
import numpy as np
import pandas as pd
from data_retrieval.loader import load_local_images
from imgutils import color_grayscale_img

@st.cache_data
def get_data(lat, lon):
    return load_local_images()

def show_images_comparison(lat, lon, show_ndvi):
    years, images, ndvis = get_data(lat, lon)
    years_range = st.slider('Select a range of years', min(years), max(years), (min(years), max(years)))
    # img1, img2 = get_images(lat, lon, show_ndvi)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Title 1")
        idx = years.index(years_range[0])
        st.image(images[idx] if not show_ndvi else color_grayscale_img(ndvis[idx]), caption="Image 1", use_column_width=True, clamp=True)
    with col2:
        st.subheader("Title 2")
        idx = years.index(years_range[1])
        st.image(images[idx] if not show_ndvi else color_grayscale_img(ndvis[idx]), caption="Image 2", use_column_width=True, clamp=True)

    area_data = pd.DataFrame(np.random.randn(20, 1), columns=["Forest Area"])
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
        show_ndvi = st.button("Show NDVI")
        lat, lon = point
        show_images_comparison(lat, lon, show_ndvi)


if __name__ == "__main__":
    main()
