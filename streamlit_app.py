import streamlit as st
import pandas as pd
import altair as alt

st.title("Let's analyze some Yacht Data 🐧📊.")

@st.cache  # add caching so we load the data only once
def load_data():
    # Load the Auto MPG data from https://github.com/allisonhorst/palmerpenguins.
    yacht_url = "https://raw.githubusercontent.com/CMU-IDS-2020/a3-yet-another-streamlit/master/yacht_hydrodynamics.data"
    return pd.read_csv(yacht_url, delim_whitespace=True, names=[
        'buoyancy position',
        'Prismatic coefficient',
        'Length-displacement ratio',
        'Beam-draught ratio',
        'Length-beam ratio',
        'Froude number',
        'Residuary resistance'
    ])

df = load_data()

st.write("Let's look at raw data in the Pandas Data Frame.")

st.write(df)

st.write("Hmm 🤔, is there some correlation between residuary resistance and froude number? Let's make a scatterplot with [Altair](https://altair-viz.github.io/) to find.")

pts = alt.selection(type="interval", encodings=["x"])

points = alt.Chart().mark_point().encode(
    x='Froude number',
    y='Residuary resistance'
).transform_filter(
    pts
).properties(
    width=300,
    height=300
)

mag = alt.Chart().mark_bar().encode(
    x=alt.X('Prismatic coefficient:N', axis=alt.Axis(format='.3f')),
    y='count()',
    color=alt.condition(pts, alt.value("lightblue"), alt.value("lightgray"))
).properties(
    width=300,
    height=300
).add_selection(pts)

chart = alt.hconcat(
    points,
    mag,
    data=df
).transform_bin(
    "Prismatic coefficient",
    field="Prismatic coefficient",
    bin=alt.Bin(step=0.01)
)

st.write(chart)
