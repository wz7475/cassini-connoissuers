import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from data_retrieval.loader import load_local, get_rgb_images


def load_images_for_coords(lat, lon):
    ds = load_local()
    years, images = get_rgb_images(ds)
    return images[0], images[1]


def get_data(lat, lon):
    pass


def show_images_comparison(data):
    lat, lon = data["coords"]
    img1, img2 = load_images_for_coords(lat, lon)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Title 1")
        st.image(img1, caption="Image 1", use_column_width=True, clamp=True)
    with col2:
        st.subheader("Title 2")
        st.image(img2, caption="Image 2", use_column_width=True, clamp=True)

    plot_data = get_data(lat, lon)

    area_data = pd.DataFrame(np.random.randn(20, 1), columns=["Forest Area"])
    st.area_chart(area_data)

    line_data = pd.DataFrame(np.random.randn(20, 2), columns=["Temperature", "CO2"])
    st.line_chart(line_data, x="Year")

    quality_data = pd.DataFrame(
        np.random.randn(20, 1),
        columns=[
            "nvdi",
        ],
    )
    st.line_chart(quality_data, x="Year")


# def get_important_points():
#     points = [
#         {'lat': 51.1074, 'lon': 17.0382, 'name': 'Wroclaw'},
#         {'lat': 52.2298, 'lon': 21.0118, 'name': 'Warsaw'},
#         {'lat': 50.0647, 'lon': 19.9450, 'name': 'Cracow'}
#     ]
#     return points


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

    show_images = st.button("Show/Hide Images")

    if not show_images and point is not None:
        data = {
            "coords": point,
        }
        show_images_comparison(data)


if __name__ == "__main__":
    main()
