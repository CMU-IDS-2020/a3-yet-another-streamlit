import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from interactive_polynomial import interactive_polynomial

st.title("Polynomial Regression on Yacht Hydrodynamics Data ⛴📊")

st.markdown('''Let's see how we can predict the residuary resistance of sailing yachts from our features.

In our [Yacht Hydrodynamics Data](https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics),
the prediction target is `Residuary resistance`, and the other 6 features are

* Buoyancy position
* Prismatic coefficient
* Length-displacement ratio
* Beam-draught ratio
* Length-beam ratio
* Froude number
''')

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

if st.checkbox('Raw Data'):
    st.write(df)

st.markdown('''## **Data Exploration**''')

st.markdown('''Let's first explore the relation between different columns!
### *Click* on each correlation block to view the scatter plot for it.
''')

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

select = alt.selection_single(fields=['column-x', 'column-y'], init={'column-x': 'Froude number', 'column-y': 'Residuary resistance'}, empty='none')

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

st.markdown('''Hmm 🤔, our target has a strong correlation with `Froude number`.
               Let's also visualize the patterns with other features together. ''') 
st.markdown('''### Select a feature to *brush & zoom in*.''')

test = st.selectbox(options=df.columns[:-2], label='Select a feature', index=1)

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

st.markdown('''## **Polynomial Regression**''')
st.markdown('''Let's use polynomial regression to make use of our features!''')
st.markdown('''* Use the side bar to select the degree, max epoch, and learning rate of our models.
* Simply move your mouse along X-axis on the chart below to see how our models fit the data.''')

interactive_polynomial(df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy(), df.columns[:-1], df.columns[-1])
