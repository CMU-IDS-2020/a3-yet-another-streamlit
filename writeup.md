# Let's analyze some Yacht Data

![A screenshot of your application. Could be a GIF.](screenshot.png)

In this application our goal is to present the exploratory visualization and quantitative approaches such as the machine learning process, in an interactive way. We first perform some data analysis to understand the correlation between our target and features, then present a machine learning panel that visualizes the training process interactively.

## Project Goals

Question: how to build a machine learning model on [Yacht dataset](https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics) to predict the residuary resistance per unit weight of displacement?

Our goal in this assignment is to present the exploratory visualization the machine learning process ***interactively***. A sufficient combination of those elements can yield a data analysis application with decent breadth and depth. Data visualization is very helpful in discovering the patterns and relations of data; Mathematic methods such as machine learning can further capture those patterns quantitatively; Friendly interactions that cover both processes make the user easily perceive and understand.

Machine learning is a powerful tool to build predictive models. For example, in the Yacht dataset, we can use linear regression to predict the residuary resistance per unit weight of displacement. However, the process to build a machine learning model is hard to documented and presented because:

* We cannot estimate the quality of features, learning rates, training epochs, and other parameters without experiments, therefore the process of building a machine learning model is interactive.
* The common optimization algorithms such as Stochastic Gradient Descent are iterative, so the learning process itself contains new data, which are in the time-series format, and dynamically constructed from the dataset.

Therefore, it is compelling to visualize both the original dataset and the new data from the learning process in an interactive way. Our solution is to create an interactive playground to present them, so users can get a better understanding about the process of building predictive models through exploring our UI.

## Design

To achieve the goal, we use Streamlit to build a simple yet comprehensive application for visualizing the machine learning process on the Yacht dataset. We first perform some exploratory visualization through the Altair API. We visualize the correlation matrix between different columns of the data, so users exploring our UI can have an idea about the relationship between different features, and between target and features. We can see that there is a strong correlation between residuary resistance and froude number. Users can further explore whether other dimensions can influence the relationship between residuary resistance and froude number. We then build an interactive panel to show the prediction of residuary resistance based on all the features we have. Users can play with the learning rate and epoch to understand the behavior of the models. During exploratory visualization, users may find that although residuary resistance has a strong correlation with froude number, their relation is not linear but rather polynomial. In our panel, users can experiment with the feature engineering part and understand how feature engineering impacts the performance of our models on the Yacht dataset.

Interactions are present in both steps to help users actively acquire insights from data. In the exploratory visualization step, variable selection and coordination techniques are leveraged. In the machine learning step, our app brings some visualizations for the time-series training status of machine learning (highlight and tooltip), and even some Streamlit widgets for users to manually tune the model based on the visual effects. Users can perform various analyses based on our app to understand the choice of features and hyperparameters.

As to the visual encodings and interactions in detail, there are three groups of Altair charts coordinating with certain interactions. 
- Firstly, to effectively show the correlations between every pair of numeric columns, scatter plot matrix seems too crowded since there are 7 columns; so we instead plot the correlation matrix (correlation encoded as color) and one single scatter plot. The key is interaction: by clicking on a block in the correlation matrix, corresponding scatter plot is shown. So the various relations are still delivered without being too crowded.
- To explore more about other features not so correlated with the target value, we use brushing. So for each feature, an interval brush can be applied to the univariate bar chart while filtering on a certain scatter plot. That enables users to discover subtle patterns.
- The visualizations and interactions in the machine learning panel focus on making the dynamic regression process easily accessible to users. One chart is the time-series line plot of train and test loss on epochs. The other is the polynomial residual plot on each scaled variate, layering a line plot and a residual scatter plot, *which vary in different epochs*.
Particularly we design the interaction to show a smooth animation of the regression training. By moving mouse over the loss chart on X-axis, rule and tooltip appears, and residual plot of a certain epoch is selected onto the plot; that's how it becomes an animation by sliding the mouse.


## Development

We have a productive collaboration. Lichen developed the machine learning panel section and Yijie developed the data exploration section. Roughly each of us spent 30 hours on this application. As we learned more about the dataset, we found better ways to hightlight our findings to users, so we iteratively improve our visualization through development.

About the most time-consuming part of our development, we would talk about the two parts respectively. For the data exploration section, it took some time to explore the insights underlying the yacht data; as to the ML panel, some tricky mathematics underlying the *residual plot* on polynomial-multivariate regression also makes up of part of the development. Both sections took some time correctly developing our designed interactions among Altair charts, since we are not familiar with Altair at the beginning.

## Reference

The dataset is hosted in UCI Machine Learning Repository. Dataset URL: http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

J. Gerritsma, R. Onnink, and A. Versluis. Geometry, resistance and stability of the delft systematic yacht hull series. In International Shipbuilding
Progress, volume 28, pages 276-297, 1981.

I. Ortigosa, R. Lopez and J. Garcia. A neural networks approach to residuary resistance of sailing
yachts prediction. In Proceedings of the International Conference on Marine Engineering MARINE
2007, 2007.

Streamlit, https://www.streamlit.io/

Altair, https://altair-viz.github.io/index.html

How to describe or visualize a multiple linear regression model? https://stats.stackexchange.com/questions/89747/how-to-describe-or-visualize-a-multiple-linear-regression-model

