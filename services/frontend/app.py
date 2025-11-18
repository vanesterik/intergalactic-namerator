import sys
from pathlib import Path

import requests  # type: ignore
import streamlit as st
from loguru import logger


def generate_names(
    backend_url: str,
    n: int = 10,
    max_length: int = 20,
    temperature: float = 1.0,
):
    try:
        response = requests.post(
            backend_url,
            json={
                "n": n,
                "max_length": max_length,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        result = response.json()
        st.session_state["names"] = result
        logger.success("Names generated successfully")

    except requests.exceptions.ConnectionError as event:
        logger.error(f"HTTP error from backend {event}")
        st.error("Error: Could not connect to the backend. Is it running?")

    except requests.exceptions.RequestException as event:
        logger.critical(f"An unexpected error {event}")
        st.error(f"An error occurred: {event}")


BACKEND_URL = "http://backend:443/generate"
LOG_FILEPATH = Path("logs/app.log")

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(LOG_FILEPATH, level="INFO", rotation="1 MB")


st.image(
    "https://raw.githubusercontent.com/vanesterik/intergalactic-namerator/refs/heads/main/references/rick-and-morty.png"
)
st.title("Intergalactic Namerator")
st.text(
    "Generate names based on the Rick and Morty universe. Use the controls below to adjust results."
)
st.divider()

col1, col2, col3, col4 = st.columns(4, vertical_alignment="bottom")

n = col1.number_input(
    "Number of names",
    min_value=1,
    max_value=50,
    value=10,
    step=5,
)

max_length = col2.slider(
    "Max length",
    min_value=3,
    max_value=30,
    value=20,
    step=1,
)

temperature = col3.slider(
    "Temperature",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.05,
)

col4.button(
    "Namerate",
    type="primary",
    width="stretch",
    args=[BACKEND_URL, n, max_length, temperature],
    on_click=generate_names,
)

if "names" in st.session_state:
    st.space()
    st.pills(
        "Names",
        options=st.session_state["names"],
        label_visibility="collapsed",
        selection_mode="multi",
    )
