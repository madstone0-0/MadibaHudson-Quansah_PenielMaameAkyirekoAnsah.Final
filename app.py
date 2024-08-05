from os import environ
from pathlib import Path

import __main__
import base64
import pandas as pd
from dill import load
import tensorflow as tf
import numpy as np
import tensorflow
import scipy

# Fix pandas not found https://stackoverflow.com/a/65318623
__main__.pandas = pd
__main__.scipy = scipy
__main__.tensorflow = tensorflow

MODE = environ.get("MODE", "dev")
modelPath = Path("./model/GPA_ANN_Wide.keras")
preprocessorPath = Path("./model/GPA_ANN_Wide_pre.pkl")
explainerPath = Path("./model/GPA_ANN_Wide_explainer.pkl")
ciPath = Path("./model/ci.pkl")
scalerPath = Path("./model/GPA_ANN_Wide_scaler.pkl")


def loadPkl(path: Path):
    obj = load(open(path, mode="rb"))
    return obj


model = tf.keras.models.load_model(modelPath)
ci = loadPkl(ciPath)
pipe = loadPkl(preprocessorPath)
explainer = loadPkl(explainerPath)
scaler = loadPkl(scalerPath)


import streamlit as st

c = st.container()
st.markdown(
    """
    <style>


    .main {
        font-family: 'Orbitron', sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        width: 100%;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# From https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/6
@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = (
        """
    <style>
    .main {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    """
        % bin_str
    )

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg("./assets/Ashesi's_Archer_Cornfield_Courtyard.jpg")

c.title("GPA Predictor")


edu = (
    "None",
    "Primary Education (4th grade)",
    "5th to 9th grade",
    "Secondary Education or Higher Education",
)

study = (
    "Less than 15 minutes",
    "15 to 30 minutes",
    "30 minutes to 1 hour",
    "More than 1 hour",
)

travel = ("Less than 2 hours", "2 to 5 hours", "5 to 10 hours", "More than 10 hours")
fail = [0, 1, 2, "4+"]
go = (
    "Very Low",
    "Low",
    "Moderate",
    "High",
    "Very High",
)
d = (
    "Very Low",
    "Low",
    "Moderate",
    "High",
    "Very High",
)
w = (
    "Very Low",
    "Low",
    "Moderate",
    "High",
    "Very High",
)


# Input fields
failures = c.selectbox("Failures", fail)

medu = c.selectbox("Mother's Education", edu)
fedu = c.selectbox("Father's Education", edu)
studytime = c.selectbox("Weekly Study Time", study)
traveltime = c.selectbox("Travel Time", travel)
age = c.number_input("Age", min_value=18, max_value=72)
goout = c.selectbox(
    "Going Out",
    go,
)
dalc = c.selectbox(
    "Weekday Alcohol Consumption",
    d,
)
walc = c.selectbox(
    "Weekend Alcohol Consumption",
    w,
)
famsup = c.selectbox("Family Support", ["yes", "no"])
schoolsup = c.selectbox("School Support", ["yes", "no"])


def enc(l, v, add1=True):
    if add1:
        return l.index(v) + 1
    else:
        return l.index(v)


def predict():
    data = {
        "failures": enc(fail, failures, False),
        "Medu": enc(edu, medu),
        "Fedu": enc(edu, fedu),
        "studytime": enc(study, studytime),
        "traveltime": enc(travel, traveltime),
        "age": age,
        "goout": enc(go, goout),
        "Dalc": enc(d, dalc),
        "Walc": enc(w, walc),
        "famsup": famsup,
        "schoolsup": schoolsup,
    }
    if not all(
        [
            age,
        ]
    ):
        st.error(
            "No inputs should be 0",
        )
        return

    data = pd.DataFrame(data, index=[0])
    dataTrans = pipe.transform(data)
    dataTrans = scaler.transform(dataTrans)
    prediction = model.predict(dataTrans)
    ciStr = ci.ci(prediction[0])
    vals, imp, dis, expected = explainer.explain(data, pipe)
    ints = explainer.interpret(vals, imp, dis, expected)
    predString = str(np.round(prediction[0], 2))
    predString = predString.replace("[", "").replace("]", "")
    c.markdown(
        f"""<h1 style='text-align: center'>{predString}</h1>
        <h4 style='text-align: center'>{ciStr}</h4>
        """,
        unsafe_allow_html=True,
    )
    for i in ints:
        c.markdown(
            f"""<h4 style='text-align: justify'>{i}</h4>""",
            unsafe_allow_html=True,
        )


c.button("Predict", on_click=predict)
