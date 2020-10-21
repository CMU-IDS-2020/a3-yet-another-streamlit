import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from interactive_polynomial import interactive_polynomial

st.title("Let's analyze some Yacht Data ⛴📊.")

@st.cache  # add caching so we load the data only once
def load_data():
    # Yacht data source: http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics.
    yacht_url = "https://raw.githubusercontent.com/CMU-IDS-2020/a3-yet-another-streamlit/master/yacht_hydrodynamics.data"
    return pd.read_csv(yacht_url, delim_whitespace=True, names=[
        'Buoyancy position',
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

st.write("Hmm 🤔, how is the correlation between columns?")

corr = df.corr()
x, y = np.meshgrid(corr.index, corr.columns)
corr_df = pd.DataFrame({
    'column-x': x.ravel(),
    'column-y': y.ravel(),
    'correlation': corr.values.ravel(),
    'text': ['%.2f' % v for v in corr.values.ravel()]
})

base = alt.Chart(corr_df).encode(
    x='column-x:O',
    y='column-y:O'
)

text = base.mark_text().encode(
    text='text:O'
)

select = alt.selection_single(fields=['column-x', 'column-y'], init={'column-x': 'Froude number', 'column-y': 'Residuary resistance'})

chart = base.mark_rect().encode(
    color='correlation:Q'
).properties(
    width=240,
    height=240
).add_selection(select)

arr = []
for x in corr.columns:
    for y in corr.columns:
        for i in range(df.shape[0]):
            arr.append([x, y, df[x][i], df[y][i]])
arr_df = pd.DataFrame(np.array(arr), columns=['column-x', 'column-y', 'value-x', 'value-y'])

scat_plot = alt.Chart(arr_df).transform_filter(
    select
).mark_circle().encode(
    alt.X('value-x:Q', scale=alt.Scale(zero=False, padding=0.1)),
    alt.Y('value-y:Q', scale=alt.Scale(zero=False, padding=0.1))
).properties(
    width=240,
    height=240
)

chart = alt.hconcat(
    chart+text,
    scat_plot
).resolve_scale(color='independent')

st.write(chart)

st.write("Hmm 🤔, is there some correlation between residuary resistance and other features?")

test = st.selectbox(options=df.columns[:-2], label='Feature selection')

pts = alt.selection(type="interval", encodings=["x"])

points = alt.Chart().mark_point().encode(
    x='Froude number',
    y='Residuary resistance'
).transform_filter(
    pts
).properties(
    width=300,
    height=300
).interactive()

mag = alt.Chart().mark_bar().encode(
    x=alt.X(test+':N', axis=alt.Axis(format='.3f')),
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
    test,
    field=test,
    bin=alt.Bin(step=0.01)
).resolve_scale(color='independent')

st.write(chart)

st.write("Let's have fun with linear regression!")

interactive_polynomial(df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy(), df.columns[:-1], df.columns[-1])
