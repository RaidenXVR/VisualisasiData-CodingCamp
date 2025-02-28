import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def load_data():
    df = pd.read_csv("./dashboard/day.csv")
    # Convert normalized temperature values back to actual values
    if "temp" in df.columns:
        df["temp_actual"] = df["temp"] * 41

    if "windspeed" in df.columns:
        df["windspeed_actual"] = df["windspeed"] * 67
    return df


def run_regression(df):
    features = [
        "temp_actual",
        "hum",
        "windspeed_actual",
        "season",
        "weathersit",
    ]
    X = df[features]
    X = sm.add_constant(X)
    y = df["cnt"]
    model = sm.OLS(y, X).fit()
    return model


def main():
    st.title("Bike Sharing Analysis Dashboard")
    st.markdown(
        """
    This dashboard showcases the results of an OLS regression model predicting bike rentals based on weather.
    """
    )

    # Load data and preview
    data = load_data()
    st.header("Data Preview")
    st.dataframe(data.head())

    st.header("Data Summary")
    st.write(data.describe())

    # Run the regression model
    model = run_regression(data)
    st.header("Regression Model Summary")
    st.markdown(f"```\n{model.summary()}\n```")

    # Actual vs. Predicted Plot
    features = [
        "temp_actual",
        "hum",
        "windspeed",
        "season",
        "weathersit",
    ]
    X_full = sm.add_constant(data[features])
    data["predicted_cnt"] = model.predict(X_full)

    # scatter plot using matplotlib
    plt.figure(figsize=(8, 6))
    plt.scatter(data["cnt"], data["predicted_cnt"], alpha=0.5, color="blue")
    plt.xlabel("Actual Rentals")
    plt.ylabel("Predicted Rentals")
    plt.title("Actual vs. Predicted Bike Rentals")
    plt.plot(
        [data["cnt"].min(), data["cnt"].max()],
        [data["cnt"].min(), data["cnt"].max()],
        color="red",
        linestyle="--",
    )
    st.pyplot(plt.gcf())
    plt.clf()

    # Plot relationships for variables
    st.header("Relationship Plots")
    fig2, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.heatmap(
        data=data[features + ["cnt"]].corr(), cmap="coolwarm", annot=True, ax=ax
    )
    ax.set_title("Data Relationship Heatmap")
    st.pyplot(fig2)

    # Sidebar for interactive predictions
    st.sidebar.header("Predict Bike Rentals")
    temp_input = st.sidebar.slider(
        "Temperature (°C)",
        float(data["temp_actual"].min()),
        float(data["temp_actual"].max()),
        float(data["temp_actual"].mean()),
    )
    hum_input = st.sidebar.slider(
        "Humidity",
        float(data["hum"].min()),
        float(data["hum"].max()),
        float(data["hum"].mean()),
    )
    wind_input = st.sidebar.slider(
        "Windspeed",
        float(data["windspeed_actual"].min()),
        float(data["windspeed_actual"].max()),
        float(data["windspeed_actual"].mean()),
    )

    # For categorical variables
    season_options = sorted(data["season"].unique())
    season_input = st.sidebar.selectbox("Season", season_options)

    weathersit_options = sorted(data["weathersit"].unique())
    weathersit_input = st.sidebar.selectbox("Weather Situation", weathersit_options)

    input_dict = {
        "const": 1,
        "temp_actual": temp_input,
        "hum": hum_input,
        "windspeed_actual": wind_input,
        "season": season_input,
        "weathersit": weathersit_input,
    }
    input_df = pd.DataFrame(input_dict, index=[0])
    prediction = model.predict(input_df)

    st.sidebar.markdown("### Predicted Bike Rentals")
    st.sidebar.write(f"**{prediction.iloc[0]:.2f} bikes**")


if __name__ == "__main__":
    main()
