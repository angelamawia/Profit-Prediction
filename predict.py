import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

sns.set_style('ticks')

# TITLE
st.title("Profit Prediction ")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.write(data.head(3))
    # Exploratory Data Analysis - # Analyzing the dataset
    # print(data.tail())
    # print(data.info())  # Dtype evaluation
    # print(data.shape)  # Dataset shape
    # print(data.describe())  # how broad spread data is
    # print(data.columns)  # check the columns available in the dataset
    # print(data.isnull().sum().values)  # check for NULL values
    # correlation = data.corr()  # How independent variables relate to dependent variable
    # print(correlation)
    # data.corr()['Profit'].sort_values().to_frame()
    # print(df.State.value_counts()) # total number of startups per State
#################################################################
    # DATA PREPROCESSING.
    # Dealing with categorical data
    data.replace({'State': {'California': 0, 'New York': 1, 'Florida': 2}}, inplace=True)

###################################################################
    # RESTRUCTURING DATASET

    X = data.iloc[:, [0, 1, 2, 3]].values
    y = data.iloc[:, [-1]].values

    # SPLITTING DATASET TO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # TRAINING MODEL
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.subheader("Profit Predicted")
    st.write(y_pred)
    score = r2_score(y_pred, y_test)

#####################################################################
    # Show relationship between each independent variable and the target variable.

    FeaturesName = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
    fig_col1, fig_col2 = st.columns(2)
    # with fig_col1:
    if st.checkbox('Show the relation between "Target" vs each variable'):
        checked_variable = st.selectbox('Select one variable:', FeaturesName)
        # Plot
        fig, ax = plt.subplots(figsize=(5, 3))
        # ax.scatter(x=data[checked_variable], y=data["Profit"])
        sns.regplot(x=data[checked_variable], y=data["Profit"], color='orange')
        plt.xlabel("checked_variable")
        plt.ylabel("Profit")
        st.pyplot(fig)
######################################################################
        # PREDICT PROFIT
        # with fig_col2:
        st.subheader("Single Profit Prediction")
        with st.form("my_form", clear_on_submit=True):
            name = st.number_input("Enter R&D Expenses", min_value=1315.46, max_value=165349.2)
            name = st.number_input("Enter Administration expenses", min_value=51283.14, max_value=182645.6)
            name = st.number_input("Enter Marketing Expenses", min_value=14681.4, max_value=192261.8)
            state = st.selectbox("State", options=[0.0000, 1.0000, 2.0000])
            st.write('0.0000: California', '1.0000: New York', '2.0000: Florida')

            submitted = st.form_submit_button("submit")
            if submitted:
                input_data = np.array([[name, name, name, state]])
                pre = model.predict(input_data)
                st.subheader("Predict Single Profit")
                st.write(pre)

#########################################################################
                # Model line fit
        # with fig_col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.regplot(x=y_test, y=y_pred, color='green')
        plt.xlabel("Actual Profit")
        plt.ylabel("Predicted Profit")
        st.pyplot(fig)

        st.subheader("Evaluating Performance Based On Metrics")
        st.write("Model Accuracy = {}".format(round(score, 2)))

        # Model Optimization and Evaluation of the result
        st.subheader("Optimization and evaluation of the Results")
        X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

        X_opt = X[:, [0, 1, 2, 3]]
        model_OLS = sm.OLS(endog=y, exog=X_opt).fit()
        model_OLS.summary()
        st.write(model_OLS.summary())
