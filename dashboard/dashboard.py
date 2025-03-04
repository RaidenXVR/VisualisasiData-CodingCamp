import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np


def run_regression(temp, hum, wind, season, weathersit):
    reg_daily_data = pd.read_csv("./dashboard/day_cleaned.csv")
    features = ["temp", "hum", "windspeed", "season", "weathersit"]
    target1 = ["casual"]
    target2 = ["registered"]
    x_data = reg_daily_data[features + target1 + target2]
    x_data["temp_actual"] = x_data["temp"].mul(41)
    x_data["hum_actual"] = x_data["hum"].mul(100)
    x_data["windspeed_actual"] = x_data["windspeed"].mul(67)

    features_actual = [
        "temp_actual",
        "hum_actual",
        "windspeed_actual",
        "season",
        "weathersit",
    ]
    x = sm.add_constant(x_data[features_actual])
    y1 = daily_data[target1]
    y2 = daily_data[target2]
    model_casual = sm.OLS(y1, x).fit()
    model_registered = sm.OLS(y2, x).fit()

    input_dict = {
        "const": 1,
        "temp": temp,
        "hum": hum,
        "windspeed_actual": wind,
        "season": season,
        "weathersit": weathersit,
    }
    input_df = pd.DataFrame(input_dict, index=[0])
    casual_out = model_casual.predict(input_df).iloc[0]
    registered_out = model_registered.predict(input_df).iloc[0]

    return [
        0 if casual_out < 0 else casual_out,
        0 if registered_out < 0 else registered_out,
    ]


# Load data
hourly_data = pd.read_csv("dashboard/hour_cleaned.csv")
daily_data = pd.read_csv("dashboard/day_cleaned.csv")

st.set_page_config(page_title="Bike Sharing Analysis", layout="wide")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("Dashboard ini dibuat untuk menganalisis pola perentalan sepeda.")

temp_input = st.sidebar.slider(
    "Temperature (°C)",
    daily_data["temp"].min() * 41,
    daily_data["temp"].max() * 41,
    daily_data["temp"].mean() * 41,
)
hum_input = st.sidebar.slider(
    "Humidity",
    daily_data["hum"].min() * 100,
    daily_data["hum"].max() * 100,
    daily_data["hum"].mean() * 100,
)
wind_input = st.sidebar.slider(
    "Wind Speed",
    daily_data["windspeed"].min() * 67,
    daily_data["windspeed"].max() * 67,
    daily_data["windspeed"].mean() * 67,
)
# For categorical variables

season_options = sorted(daily_data["season"].unique())
season_input = st.sidebar.selectbox("Season", season_options)

weathersit_options = sorted(daily_data["weathersit"].unique())
weathersit_input = st.sidebar.selectbox("Weather Situation", weathersit_options)

prediction = run_regression(
    temp_input, hum_input, wind_input, season_input, weathersit_input
)

st.sidebar.markdown("### Predicted Bike Rentals")

st.sidebar.write(f"**Casual: {prediction[0]:.2f} bikes**")
st.sidebar.write(f"**Registered: {prediction[1]:.2f} bikes**")

st.title("Bike Sharing Analysis Dashboard")

# Dataset Preview
st.header("Dataset Preview")
tab1, tab2 = st.tabs(["Hourly Data", "Daily Data"])

with tab1:
    st.subheader("Hourly Data Preview")
    st.dataframe(hourly_data.head())

with tab2:
    st.subheader("Daily Data Preview")
    st.dataframe(daily_data.head())

# Q1: Libur vs Kerja
tab1.header("1. Hari Libur Vs. Hari Kerja")
tab2.header("1. Hari Libur Vs. Hari Kerja")

tab1.subheader("Weekend Vs. Weekday")
tab2.subheader("Weekend Vs. Weekday")


hourly_means = (
    hourly_data.groupby("weekday")[["registered", "casual"]].mean().reset_index()
)
daily_means = (
    daily_data.groupby("weekday")[["registered", "casual"]].mean().reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

colors = ["#1f77b4", "#ff7f0e"]  # Blue for registered, Orange for casual
weekday_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# Convert numeric weekday to labels
hourly_means["weekday"] = hourly_means["weekday"].map(lambda x: weekday_labels[x])
daily_means["weekday"] = daily_means["weekday"].map(lambda x: weekday_labels[x])

sns.barplot(x="weekday", y="registered", data=hourly_means, ax=axes[0], color=colors[0])
axes[0].set_title("Hourly - Registered Users (Mean)")
sns.barplot(x="weekday", y="casual", data=hourly_means, ax=axes[1], color=colors[1])
axes[1].set_title("Hourly - Casual Users (Mean)")

for ax in axes.flat:
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Mean Count")
    ax.set_xticklabels(weekday_labels, rotation=45)

tab1.pyplot(fig)


fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

sns.barplot(x="weekday", y="registered", data=daily_means, ax=axes2[0], color=colors[0])
axes2[0].set_title("Daily - Registered Users (Mean)")
sns.barplot(x="weekday", y="casual", data=daily_means, ax=axes2[1], color=colors[1])
axes2[1].set_title("Daily - Casual Users (Mean)")


for ax in axes2.flat:
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Mean Count")
    ax.set_xticklabels(weekday_labels, rotation=45)
tab2.pyplot(fig2)

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

hourly_means = (
    hourly_data.groupby("workingday")[["registered", "casual"]].mean().reset_index()
)
daily_means = (
    daily_data.groupby("workingday")[["registered", "casual"]].mean().reset_index()
)

tab1.subheader("Hari Libur vs. Hari Kerja")
sns.barplot(
    x="workingday", y="registered", data=hourly_means, ax=axes3[0], color=colors[0]
)
axes3[0].set_title("Hourly - Registered Users (Mean)")
sns.barplot(x="workingday", y="casual", data=hourly_means, ax=axes3[1], color=colors[1])
axes3[1].set_title("Hourly - Casual Users (Mean)")

tab1.pyplot(fig3)

fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4))

sns.barplot(
    x="workingday", y="registered", data=daily_means, ax=axes4[0], color=colors[0]
)
axes4[0].set_title("Daily - Registered Users (Mean)")
sns.barplot(x="workingday", y="casual", data=daily_means, ax=axes4[1], color=colors[1])
axes4[1].set_title("Daily - Casual Users (Mean)")

tab2.pyplot(fig4)

tab1.markdown(
    """
**Insight:**

Dari grafik di atas, didapatkan pola:
- Pengguna teregistrasi lebih sedikit merental sepeda pada saat hari libur, namun lebih banyak merental sepeda saat hari kerja.
- Berkebalikan dengan pengguna teregistrasi, pengguna kasual lebih banyak merental pada hari libur dan lebih sedikit pada hari kerja.

            """
)
tab2.markdown(
    """
**Insight:**

Dari grafik di atas, didapatkan pola:
- Pengguna teregistrasi lebih sedikit merental sepeda pada saat hari libur, namun lebih banyak merental sepeda saat hari kerja.
- Berkebalikan dengan pengguna teregistrasi, pengguna kasual lebih banyak merental pada hari libur dan lebih sedikit pada hari kerja.

            """
)
# Q2: Peak Hour
tab1.header("2. Frekuensi Jam Puncak Sibuk")
tab2.header("2. Frekuensi Jam Puncak Sibuk")

tab1.subheader("Jam Puncak Sibuk Total")
tab2.subheader("Jam Puncak Sibuk Total")

peak_hours_registered = hourly_data.loc[
    hourly_data.groupby("dteday")["registered"].idxmax(), ["dteday", "hr"]
]
peak_counts_registered = peak_hours_registered["hr"].value_counts().sort_index()
peak_hours_casual = hourly_data.loc[
    hourly_data.groupby("dteday")["casual"].idxmax(), ["dteday", "hr"]
]
peak_counts_casual = peak_hours_casual["hr"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.barplot(data=peak_counts_registered, ax=axes[0], color=colors[0])
axes[0].set_title("Peak Hour Frequency - Registered Users")
sns.barplot(data=peak_counts_casual, ax=axes[1], color=colors[1])
axes[1].set_title("Peak Hour Frequency - Casual Users (Mean)")

for ax in axes.flat:
    ax.set_xlabel("Hour")
    ax.set_ylabel("Frequency")

tab1.pyplot(fig)
tab2.pyplot(fig)

tab1.subheader("Jam Puncak Sibuk Hari Kerja")
tab2.subheader("Jam Puncak Sibuk Hari Kerja")
data_workingday_registered = hourly_data[hourly_data["workingday"].between(1, 1)]
peak_hour_workingday_registered = data_workingday_registered.loc[
    data_workingday_registered.groupby("dteday")["registered"].idxmax(),
    ["dteday", "hr", "registered"],
]

working_day_registered = (
    peak_hour_workingday_registered["hr"].value_counts().sort_index()
)

data_workingday_casual = hourly_data[hourly_data["workingday"].between(1, 1)]
peak_hour_workingday_casual = data_workingday_casual.loc[
    data_workingday_casual.groupby("dteday")["casual"].idxmax(),
    ["dteday", "hr", "casual"],
]

working_day_casual = peak_hour_workingday_casual["hr"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

colors = ["#1f77b4", "#ff7f0e"]  # Blue for registered, Orange for casual

sns.barplot(data=working_day_registered, ax=axes[0], color=colors[0])
axes[0].set_title("Peak Hour Frequency - Registered Users")
sns.barplot(data=working_day_casual, ax=axes[1], color=colors[1])
axes[1].set_title("Peak Hour Frequency - Casual Users (Mean)")

for ax in axes.flat:
    ax.set_xlabel("Hour")
    ax.set_ylabel("Frequency")

tab1.pyplot(fig)
tab2.pyplot(fig)

tab1.subheader("Jam Puncak Sibuk Hari Libur")
tab2.subheader("Jam Puncak Sibuk Hari Libur")

data_workingday_registered = hourly_data[hourly_data["workingday"].between(0, 0)]
peak_hour_workingday_registered = data_workingday_registered.loc[
    data_workingday_registered.groupby("dteday")["registered"].idxmax(),
    ["dteday", "hr", "registered"],
]

not_working_day_registered = (
    peak_hour_workingday_registered["hr"].value_counts().sort_index()
)

data_workingday_casual = hourly_data[hourly_data["workingday"].between(0, 0)]
peak_hour_workingday_casual = data_workingday_casual.loc[
    data_workingday_casual.groupby("dteday")["casual"].idxmax(),
    ["dteday", "hr", "casual"],
]

not_working_day_casual = peak_hour_workingday_casual["hr"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

colors = ["#1f77b4", "#ff7f0e"]  # Blue for registered, Orange for casual

sns.barplot(data=not_working_day_registered, ax=axes[0], color=colors[0])
axes[0].set_title("Peak Hour Frequency - Registered Users")
sns.barplot(data=not_working_day_casual, ax=axes[1], color=colors[1])
axes[1].set_title("Peak Hour Frequency - Casual Users (Mean)")

for ax in axes.flat:
    ax.set_xlabel("Hour")
    ax.set_ylabel("Frequency")

tab1.pyplot(fig)
tab2.pyplot(fig)


tab1.markdown(
    """
**Insight:**

Dari ketiga grafik di atas, didapatkan bahwa:
- Perentalan paling banyak terjadi pada jam berangkat kerja (Pukul 8) dan jam pulang kerja (Pukul 17-18).
- Pada hari kerja, pengguna teregistrasi paling banyak merental pada jam berangkat kerja dan jam pulang kerja, sementara pengguna kasual memiliki puncak di jam 17, namun merata di jam lainnya.
- Pada hari libur, pengguna teregistrasi dan kasual sama-sama merata dalam hal waktu perentalan sepeda.
- Pada hari libur, pengguna teregistrasi memiliki puncak pada pukul 12 dan 13, sementara pengguna kasual memiliki puncak pada pukul 10. 
              """
)
tab2.markdown(
    """
**Insight:**

Dari ketiga grafik di atas, didapatkan bahwa:
- Perentalan paling banyak terjadi pada jam berangkat kerja (Pukul 8) dan jam pulang kerja (Pukul 17-18).
- Pada hari kerja, pengguna teregistrasi paling banyak merental pada jam berangkat kerja dan jam pulang kerja, sementara pengguna kasual memiliki puncak di jam 17, namun merata di jam lainnya.
- Pada hari libur, pengguna teregistrasi dan kasual sama-sama merata dalam hal waktu perentalan sepeda.
- Pada hari libur, pengguna teregistrasi memiliki puncak pada pukul 12 dan 13, sementara pengguna kasual memiliki puncak pada pukul 10. 
              """
)

# Q3: Correlation
tab1.header("3. Korelasi Keadaan Cuaca dengan Jumlah Perental")

fig, axes = plt.subplots(1, 1, figsize=(10, 4))
corr_features = ["temp", "hum", "weathersit", "season", "windspeed"]
targets = ["casual", "registered"]

sns.heatmap(
    daily_data[corr_features + targets].corr(), cmap="coolwarm", annot=True, ax=axes
)
tab1.pyplot(fig)
tab2.pyplot(fig)

tab1.markdown(
    """
**Insight:**
- Korelasi antara keadaan cuaca terhadap banyak pengguna kasual dan teregistrasi bisa dibilang sama.
- Korelasi terbesar ada pada temperatur dan musim.
- Korelasi `hum`, `weathersit`, dan `windspeed` terhadap banyak pengguna adalah korelasi terbalik, sehingga jika `hum`, `weathersit`, dan `windspeed` bertambah maka jumlah pengguna berkurang.
              """
)
tab2.markdown(
    """
**Insight:**
- Korelasi antara keadaan cuaca terhadap banyak pengguna kasual dan teregistrasi bisa dibilang sama.
- Korelasi terbesar ada pada temperatur dan musim.
- Korelasi `hum`, `weathersit`, dan `windspeed` terhadap banyak pengguna adalah korelasi terbalik, sehingga jika `hum`, `weathersit`, dan `windspeed` bertambah maka jumlah pengguna berkurang.
              """
)

tab1.header("Analisis Lanjutan")
tab2.header("Analisis Lanjutan")
tab1.subheader("OLS Regression Casual Renter")
tab2.subheader("OLS Regression Casual Renter")


features = ["temp", "hum", "windspeed", "season", "weathersit"]
target1 = ["casual"]
target2 = ["registered"]
x_data = daily_data[features + target1 + target2]
x_data["temp_actual"] = x_data["temp"].mul(41)
x_data["hum_actual"] = x_data["hum"].mul(100)
x_data["windspeed_actual"] = x_data["windspeed"].mul(67)

features_actual = [
    "temp_actual",
    "hum_actual",
    "windspeed_actual",
    "season",
    "weathersit",
]
x = sm.add_constant(x_data[features_actual])
y1 = daily_data[target1]
y2 = daily_data[target2]

model = sm.OLS(y1, x).fit()
tab1.code(model.summary())
tab2.code(model.summary())


tab1.subheader("OLS Regression Registered Renter")
tab2.subheader("OLS Regression Registered Renter")
model = sm.OLS(y2, x).fit()
tab1.code(model.summary())
tab2.code(model.summary())

tab1.markdown(
    """
**Insight:**
- Untuk Regresi dengan data perental kasual, nilai R-squared adalah 0.380, artinya variabel independen memiliki pengaruh terhadap variabel dependen sebesar 38%.
- Untuk Regresi dengan data perental teregistrasi, nilai R-squared adalah 0.419, artinya variabel independen memiliki pengaruh terhadap variabel dependen sebesar 41.9%.
- Untuk perental kasual, musim tidak memiliki pengaruh signifikan terhadap jumlah perental kasual berdasarkan nilai `p` yang lebih dari 0.05.
- Untuk perental teregistrasi, semua parameter cuaca memiliki pengaruh yang signifikan terhadap jumlah pengguna teregistrasi."""
)
tab2.markdown(
    """
**Insight:**
- Untuk Regresi dengan data perental kasual, nilai R-squared adalah 0.380, artinya variabel independen memiliki pengaruh terhadap variabel dependen sebesar 38%.
- Untuk Regresi dengan data perental teregistrasi, nilai R-squared adalah 0.419, artinya variabel independen memiliki pengaruh terhadap variabel dependen sebesar 41.9%.
- Untuk perental kasual, musim tidak memiliki pengaruh signifikan terhadap jumlah perental kasual berdasarkan nilai `p` yang lebih dari 0.05.
- Untuk perental teregistrasi, semua parameter cuaca memiliki pengaruh yang signifikan terhadap jumlah pengguna teregistrasi."""
)

tab1.markdown(
    """
### Conclusion
- Dari data yang ada, perbedaan perentalan sepeda pada hari libur dan hari kerja terdapat pada apakah pengguna tersebut pengguna teregistrasi atau kasual. Pengguna kasual lebih banyak merental pada weekend dan hari libur, sementara pengguna teregristrasi lebih banyak merental pada hari kerja.
- Dari data yang ada, peak hour terbanyak adalah pada pukul 8, 17, dan 18 karena merupakan jam berangkat dan pulang kerja. Sementara, pengguna kasual lebih merata dalam perihal banyaknya peak hour.
- Dari data yang ada, korelasi antara keadaan cuaca mulai dari temperatur, kelembaban, kondisi cuaca, musim, dan kecepatan angin memiliki korelasi yang cukup signifikan dengan temperatur yang memiliki korelasi sangat tinggi dibandingkan parameter cuaca lain.
              """
)

tab2.markdown(
    """
### Conclusion
- Dari data yang ada, perbedaan perentalan sepeda pada hari libur dan hari kerja terdapat pada apakah pengguna tersebut pengguna teregistrasi atau kasual. Pengguna kasual lebih banyak merental pada weekend dan hari libur, sementara pengguna teregristrasi lebih banyak merental pada hari kerja.
- Dari data yang ada, peak hour terbanyak adalah pada pukul 8, 17, dan 18 karena merupakan jam berangkat dan pulang kerja. Sementara, pengguna kasual lebih merata dalam perihal banyaknya peak hour.
- Dari data yang ada, korelasi antara keadaan cuaca mulai dari temperatur, kelembaban, kondisi cuaca, musim, dan kecepatan angin memiliki korelasi yang cukup signifikan dengan temperatur yang memiliki korelasi sangat tinggi dibandingkan parameter cuaca lain.
              """
)

st.header("Dataset Statistics")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Hourly Data")
    st.write(f"Records: {len(hourly_data)}")
    st.write(
        f"Date Range: {hourly_data['dteday'].min()} to {hourly_data['dteday'].max()}"
    )

with col2:
    st.subheader("Daily Data")
    st.write(f"Records: {len(daily_data)}")
    st.write(
        f"Date Range: {daily_data['dteday'].min()} to {daily_data['dteday'].max()}"
    )

# Add footer
st.markdown("---")
st.markdown(
    "**Analysis by Fitran Alfian Nizar** | [GitHub](https://github.com/RaidenXVR)"
)
