import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np


# Load data
hourly_data = pd.read_csv("dashboard/hour_cleaned.csv")
daily_data = pd.read_csv("dashboard/day_cleaned.csv")

# Convert date columns to datetime
hourly_data["dteday"] = pd.to_datetime(hourly_data["dteday"])
daily_data["dteday"] = pd.to_datetime(daily_data["dteday"])

st.set_page_config(page_title="Bike Sharing Analysis", layout="wide")

# Filter
st.sidebar.header("Filter Data")

# Date range filter
start_date = st.sidebar.date_input(
    "Start Date",
    value=hourly_data["dteday"].min(),
    min_value=hourly_data["dteday"].min(),
    max_value=hourly_data["dteday"].max(),
)
end_date = st.sidebar.date_input(
    "End Date",
    value=hourly_data["dteday"].max(),
    min_value=hourly_data["dteday"].min(),
    max_value=hourly_data["dteday"].max(),
)

# Working Day filter
workingday_options = st.sidebar.multiselect(
    "Select Day Type",
    options=["Working Day", "Holiday"],
    default=["Working Day", "Holiday"],
)
# Map the selection to the values in the dataset (1 for working day, 0 for holiday)
workingday_filter = []
if "Working Day" in workingday_options:
    workingday_filter.append(1)
if "Holiday" in workingday_options:
    workingday_filter.append(0)

# Hour filter for hourly data
hour_range = st.sidebar.slider("Hour Range (Hourly Data)", 0, 23, (0, 23))

# Apply filters to the datasets
filtered_hourly = hourly_data[
    (hourly_data["dteday"] >= pd.to_datetime(start_date))
    & (hourly_data["dteday"] <= pd.to_datetime(end_date))
    & (hourly_data["workingday"].isin(workingday_filter))
    & (hourly_data["hr"].between(hour_range[0], hour_range[1]))
]

filtered_daily = daily_data[
    (daily_data["dteday"] >= pd.to_datetime(start_date))
    & (daily_data["dteday"] <= pd.to_datetime(end_date))
    & (daily_data["workingday"].isin(workingday_filter))
]

# Alert if there is no data after filtering
if filtered_hourly.empty or filtered_daily.empty:
    st.warning(
        "No data available for the selected filters. Please adjust your filter options."
    )

# Sidebar About section remains as before
st.sidebar.title("About")
st.sidebar.info("Dashboard ini dibuat untuk menganalisis pola perentalan sepeda.")

# -------------------------------
# Main Dashboard Title and Dataset Preview
# -------------------------------
st.title("Bike Sharing Analysis Dashboard")

st.header("Dataset Preview")
tab1, tab2 = st.tabs(["Hourly Data", "Daily Data"])

with tab1:
    st.subheader("Hourly Data Preview")
    st.dataframe(filtered_hourly.head())

with tab2:
    st.subheader("Daily Data Preview")
    st.dataframe(filtered_daily.head())


# Q1: Libur vs Kerja
tab1.header("1. Hari Libur Vs. Hari Kerja")
tab2.header("1. Hari Libur Vs. Hari Kerja")

tab1.subheader("Weekend Vs. Weekday")
tab2.subheader("Weekend Vs. Weekday")


hourly_means = (
    filtered_hourly.groupby("weekday")[["registered", "casual"]].mean().reset_index()
)
daily_means = (
    filtered_daily.groupby("weekday")[["registered", "casual"]].mean().reset_index()
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
    filtered_hourly.groupby("workingday")[["registered", "casual"]].mean().reset_index()
)
daily_means = (
    filtered_daily.groupby("workingday")[["registered", "casual"]].mean().reset_index()
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

peak_hours_registered = filtered_hourly.loc[
    filtered_hourly.groupby("dteday")["registered"].idxmax(), ["dteday", "hr"]
]
peak_counts_registered = peak_hours_registered["hr"].value_counts().sort_index()
peak_hours_casual = filtered_hourly.loc[
    filtered_hourly.groupby("dteday")["casual"].idxmax(), ["dteday", "hr"]
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
data_workingday_registered = filtered_hourly[
    filtered_hourly["workingday"].between(1, 1)
]
peak_hour_workingday_registered = data_workingday_registered.loc[
    data_workingday_registered.groupby("dteday")["registered"].idxmax(),
    ["dteday", "hr", "registered"],
]

working_day_registered = (
    peak_hour_workingday_registered["hr"].value_counts().sort_index()
)

data_workingday_casual = filtered_hourly[filtered_hourly["workingday"].between(1, 1)]
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

data_workingday_registered = filtered_hourly[
    filtered_hourly["workingday"].between(0, 0)
]
peak_hour_workingday_registered = data_workingday_registered.loc[
    data_workingday_registered.groupby("dteday")["registered"].idxmax(),
    ["dteday", "hr", "registered"],
]

not_working_day_registered = (
    peak_hour_workingday_registered["hr"].value_counts().sort_index()
)

data_workingday_casual = filtered_hourly[filtered_hourly["workingday"].between(0, 0)]
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
    filtered_daily[corr_features + targets].corr(), cmap="coolwarm", annot=True, ax=axes
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
x_data = filtered_daily[features + target1 + target2]
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
y1 = filtered_daily[target1]
y2 = filtered_daily[target2]

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
    st.write(f"Records: {len(filtered_hourly)}")
    st.write(
        f"Date Range: {filtered_hourly['dteday'].min()} to {filtered_hourly['dteday'].max()}"
    )

with col2:
    st.subheader("Daily Data")
    st.write(f"Records: {len(filtered_daily)}")
    st.write(
        f"Date Range: {filtered_daily['dteday'].min()} to {filtered_daily['dteday'].max()}"
    )

# Add footer
st.markdown("---")
st.markdown(
    "**Analysis by Fitran Alfian Nizar** | [GitHub](https://github.com/RaidenXVR)"
)
