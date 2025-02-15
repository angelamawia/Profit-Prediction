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


# Function to load and display dataset
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset")
    st.write(data.head(3))
    return data


# Function for preprocessing the dataset
def preprocess_data(data):
    data.replace({'State': {'California': 0, 'New York': 1, 'Florida': 2}}, inplace=True)
    X = data.iloc[:, [0, 1, 2, 3]].values
    y = data.iloc[:, [-1]].values
    return X, y


# Function to train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Function to predict and display results
def display_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.subheader("Profit Predicted")
    st.write(y_pred)
    score = r2_score(y_test, y_pred)
    return y_pred, score


# Function to plot regression between variables
def plot_relation(data, FeaturesName):
    if st.checkbox('Show the relation between "Target" vs each variable'):
        checked_variable = st.selectbox('Select one variable:', FeaturesName)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.regplot(x=data[checked_variable], y=data["Profit"], color='orange')
        plt.xlabel(checked_variable)
        plt.ylabel("Profit")
        st.pyplot(fig)


# Function for single profit prediction
def single_profit_prediction(model):
    st.subheader("Single Profit Prediction")
    with st.form("my_form", clear_on_submit=True):
        rd_expense = st.number_input("Enter R&D Expenses", min_value=1315.46, max_value=165349.2)
        admin_expense = st.number_input("Enter Administration expenses", min_value=51283.14, max_value=182645.6)
        marketing_expense = st.number_input("Enter Marketing Expenses", min_value=14681.4, max_value=192261.8)
        state = st.selectbox("State", options=[0.0000, 1.0000, 2.0000])
        st.write('0.0000: California', '1.0000: New York', '2.0000: Florida')

        submitted = st.form_submit_button("Submit")
        if submitted:
            input_data = np.array([[rd_expense, admin_expense, marketing_expense, state]])
            prediction = model.predict(input_data)
            st.subheader("Predicted Single Profit")
            st.write(prediction)


# Function to evaluate the model and display summary
def evaluate_model(X, y):
    X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
    X_opt = X[:, [0, 1, 2, 3]]
    model_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    st.write(model_OLS.summary())


# Function to plot actual vs predicted profit
def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.regplot(x=y_test, y=y_pred, color='green')
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    st.pyplot(fig)


# Main function to run the app
def main():
    st.title("Profit Prediction")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        model = train_model(X_train, y_train)
        y_pred, score = display_predictions(model, X_test, y_test)

        FeaturesName = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
        plot_relation(data, FeaturesName)
        single_profit_prediction(model)
        plot_actual_vs_predicted(y_test, y_pred)

        st.subheader("Evaluating Performance Based On Metrics")
        st.write(f"Model Accuracy = {round(score, 2)}")

        st.subheader("Optimization and Evaluation of the Results")
        evaluate_model(X, y)


if __name__ == "__main__":
    main()
