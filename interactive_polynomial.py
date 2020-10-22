import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from multivariate_LR import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# time-series data for losses
def get_loss_df(train_losses, test_losses):
    #  losses should be [float]
    num_epoch = len(test_losses)
    return pd.DataFrame({'Epoch' : range(num_epoch), 'test MSE' : test_losses, 'train MSE' : train_losses})

# multivariate-polynomial plot line points for each epoch each variable
def get_plotline_df(x_range, num_var, num_exp, weights_on_epoch, variable_names, measure_name):
    num_epoch = len(weights_on_epoch)
    num_samples = len(x_range)
    assert num_epoch > 0
    assert num_var * num_exp == len(weights_on_epoch[0]) - 1
    assert len(variable_names) > 0

    epoch_dfs = []
    for epoch_id in range(num_epoch):
        plotline_df = pd.DataFrame({'Epoch' : [epoch_id] * num_samples})
        for var_id in range(len(variable_names)):
            var_name = variable_names[var_id]
            # x fields for each plot
            plotline_df[var_name] = x_range
            # y fields for each plot - supposed polynomials of the model.
            y_values = np.zeros(num_samples)
            prod = np.ones(num_samples)
            weight = weights_on_epoch[epoch_id]
            for idx in range(var_id, num_var * num_exp, num_var):
                prod = prod * x_range
                y_values = y_values + prod * weight[idx]
            # add interception
            y_values = y_values + weight[-1]
            plotline_df[measure_name + " Residual on " + var_name] = y_values
        
        epoch_dfs.append(plotline_df)

    return pd.concat(epoch_dfs)

# residual points for each epoch each variable, varying in time-series.
# X is scaled and concatenated with different exponants
def get_residual_xy_df(X, y, num_var, num_exp, weights_on_epoch, variable_names, measure_name):
    num_epoch = len(weights_on_epoch)
    num_samples = len(y)
    assert num_epoch > 0
    assert num_var * num_exp == len(weights_on_epoch[0]) - 1
    assert len(variable_names) > 0
    assert X.shape[1] == num_var * num_exp

    # calculate the residual masks for each variable, which will be subtracted by ground truth y.
    masks = []
    for var_id in range(len(variable_names)):
        mask_ = np.ones(num_var * num_exp)
        mask_[list(range(var_id, num_var * num_exp, num_var))] = 0.0
        masks.append(mask_)

    epoch_dfs = []
    for epoch_id in range(num_epoch):
        plotscat_df = pd.DataFrame({'Epoch' : [epoch_id] * num_samples})
        for var_id in range(len(variable_names)):
            var_name = variable_names[var_id]
            # x fields for each plot
            plotscat_df[var_name] = X[:, var_id].reshape(num_samples)
            # y fields for each plot - real y subtract the residuals
            weight = weights_on_epoch[epoch_id][0:-1]
            redisuals = X @ (masks[var_id] * weight)
            y_values = y - redisuals
            plotscat_df[measure_name + " Residual on " + var_name] = y_values
            
        epoch_dfs.append(plotscat_df)

    return pd.concat(epoch_dfs)


def interactive_polynomial(feat_X, feat_y, variable_names, measure_name):
    '''
    feat_X: np.array(n_sample, n_feature), not need to be scaled
    feat_y: np.array(n_sample)
    variable_names: [str], size equal to n_feature, name of each column.
    measure_name: str, name of the column to regression on.
    '''
    num_exp = st.slider('Polynomial Exponantial', 1, 3, 2, 1)
    n_iter = st.slider('Maximum Training Epoch', 0, 200, 100, 10)
    learning_rate = st.slider('Learning Rate', 0.01, 0.25, 0.01, 0.01)
    # lam = st.slider('Lambda', 0.0, 2e-4, 1e-4, 1e-5)

    # scale the feature matrix.
    # then apply polynomial tranformation. Concatenate high-exps
    scaler = StandardScaler()
    feat_X = scaler.fit_transform(feat_X)
    Xs = []
    for num in range(1, num_exp + 1):
        Xs.append(feat_X ** num)
    feat_X = np.concatenate(Xs, axis=1)

    lr_model = LinearRegression(lam=1e-4)


    X_train, X_test, y_train, y_test = train_test_split(feat_X, feat_y, test_size=0.5, random_state=0)
    losses, test_losses, weights_epochs = lr_model.fit(X_train, y_train, X_test, y_test, n_iter, learning_rate)

    # selector based on epoch
    epoch_selection = alt.selection_single(nearest=True, on='mouseover',
                        fields=['Epoch'], empty='none')

    # times-series for the loss
    loss_df = get_loss_df(losses, test_losses)
    loss_df = loss_df.melt('Epoch', var_name='loss', value_name='MSE')
    # selector layer
    selector_layer = alt.Chart(loss_df).mark_point().encode(
        alt.X('Epoch'),
        opacity=alt.value(0)
    ).add_selection(epoch_selection)
    loss_line = alt.Chart(loss_df).mark_line().encode(
        alt.X('Epoch'),
        alt.Y('MSE'),
        color='loss:N'
    )
    ruleline = alt.Chart(loss_df).mark_rule().encode(
        alt.X('Epoch'),
        color=alt.value('grey')
    ).transform_filter(epoch_selection)
    tooltip = alt.Chart(loss_df).mark_text(align='left', dx=5, dy=-5).encode(
        alt.X('Epoch'),
        alt.Y('MSE'),
        alt.Text('MSE')
    ).transform_filter(epoch_selection)

    # st.write(alt.layer(selector_layer, loss_line, tooltip, ruleline).properties(width=700, height=300))
    curr_chart = alt.layer(selector_layer, loss_line, tooltip, ruleline).properties(width=700, height=300).resolve_scale(color='independent')

    # get the layered visualization of residual line plot and residual X-Y points.
    residual_xy_df = get_residual_xy_df(X_test, y_test, len(variable_names), num_exp, weights_epochs, variable_names, measure_name)
    plotline_df = get_plotline_df(np.arange(-2.0, 2.0, 0.05), len(variable_names), num_exp, weights_epochs, variable_names, measure_name)

    # list the residual plot on each dimension, three in a row to look better.
    curr_list = []
    for var_name in variable_names:
        # residual points and the line plot together.
        residual_xy_plot = alt.Chart(residual_xy_df).mark_point().encode(
                alt.X(var_name),
                alt.Y(measure_name + " Residual on " + var_name)
            ).transform_filter(epoch_selection)
        plotline_plot = alt.Chart(plotline_df).mark_line().encode(
            alt.X(var_name),
            alt.Y(measure_name + " Residual on " + var_name),
            color = alt.value('red')
        ).transform_filter(epoch_selection)
        curr_list.append(alt.layer(plotline_plot, residual_xy_plot).properties(width=200, height=200))
        if len(curr_list) == 3:
            curr_chart = curr_chart & (curr_list[0] | curr_list[1] | curr_list[2])
            curr_list = []
    
    if curr_list != []:
        last_row = curr_list[0]
        for idx in range(1, len(curr_list)):
            last_row = last_row | curr_list[idx]
        curr_chart = curr_chart & last_row

    # at last, write everything to streamlit UI.
    st.write(curr_chart)

if __name__ == '__main__':
    n_sample = 100
    X = np.random.randn(n_sample, 2)
    y = (X ** 2).sum(axis=1) + np.random.normal(scale=0.2, size=n_sample)
    interactive_polynomial(X, y, ['X1', 'X2'], 'y')
                

