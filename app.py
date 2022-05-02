import streamlit as st
from fastai.vision.all import *
import plotly.express as px

# PosixPath
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title("Transportni klassifikatsiya qiluvchi model")

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['jpg','png','jpeg','gif','svg'])
if file:
    st.image(file)

    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner('transport_model.pkl')

    # prediction
    # prediction = model.predict(img)
    pred, pred_id, prob = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {prob[pred_id]*100:.1f}%")

    # plotting
    fig = px.bar(x=prob*100, y=model.dls.vocab)
    st.plotly_chart(fig)