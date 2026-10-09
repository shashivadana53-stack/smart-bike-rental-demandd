import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/hour.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data types
print("\nData Types:")
print(df.dtypes)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display basic statistical summary
print("\nStatistical Summary:")
print(df.describe())
# ==========================================
# MODULE I - DESCRIPTIVE STATISTICS
# ==========================================

print("\n====================================")
print("MODULE I - DESCRIPTIVE STATISTICS")
print("====================================")

# Select bike rental demand column
demand = df["cnt"]

# Mean
mean_demand = demand.mean()

# Median
median_demand = demand.median()

# Mode
mode_demand = demand.mode()[0]

# Variance
variance_demand = demand.var()

# Standard Deviation
std_demand = demand.std()

# Minimum
min_demand = demand.min()

# Maximum
max_demand = demand.max()

print("\nMean:", mean_demand)
print("Median:", median_demand)
print("Mode:", mode_demand)
print("Variance:", variance_demand)
print("Standard Deviation:", std_demand)
print("Minimum Demand:", min_demand)
print("Maximum Demand:", max_demand)


# ==========================================
# HISTOGRAM
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(df["cnt"], bins=30)

plt.title("Distribution of Bike Rental Demand")
plt.xlabel("Number of Bike Rentals")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("graphs/demand_histogram.png")
plt.show()


# ==========================================
# BOX PLOT
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["cnt"])

plt.title("Box Plot of Bike Rental Demand")
plt.xlabel("Number of Bike Rentals")

plt.tight_layout()
plt.savefig("graphs/demand_boxplot.png")
plt.show()


# ==========================================
# DEMAND BY HOUR
# ==========================================

hour_demand = df.groupby("hr")["cnt"].mean()

plt.figure(figsize=(10, 6))

plt.plot(hour_demand.index, hour_demand.values, marker="o")

plt.title("Average Bike Rental Demand by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Bike Rentals")
plt.xticks(range(0, 24))

plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/demand_by_hour.png")
plt.show()


# ==========================================
# DEMAND BY SEASON
# ==========================================

season_demand = df.groupby("season")["cnt"].mean()

plt.figure(figsize=(8, 5))

plt.bar(season_demand.index, season_demand.values)

plt.title("Average Bike Rental Demand by Season")
plt.xlabel("Season")
plt.ylabel("Average Bike Rentals")

plt.tight_layout()
plt.savefig("graphs/demand_by_season.png")
plt.show()
# ==========================================
# MODULE II - PROBABILITY
# ==========================================

print("\n====================================")
print("MODULE II - PROBABILITY")
print("====================================")

# Define high demand using the median rental demand
demand_threshold = df["cnt"].median()

df["high_demand"] = df["cnt"] > demand_threshold

# ------------------------------------------
# 1. Probability of High Demand
# ------------------------------------------

total_cases = len(df)
high_demand_cases = df["high_demand"].sum()

prob_high_demand = high_demand_cases / total_cases

print("\n1. Probability of High Demand:")
print("Total Cases:", total_cases)
print("High Demand Cases:", high_demand_cases)
print("Probability:", prob_high_demand)
print("Percentage:", prob_high_demand * 100, "%")


# ------------------------------------------
# 2. Probability of High Demand
#    given Clear Weather
# ------------------------------------------

# In this dataset:
# Weather Situation 1 = Clear / Few clouds

clear_weather = df[df["weathersit"] == 1]

clear_total = len(clear_weather)
clear_high_demand = clear_weather["high_demand"].sum()

prob_high_given_clear = clear_high_demand / clear_total

print("\n2. Probability of High Demand given Clear Weather:")
print("Clear Weather Cases:", clear_total)
print("High Demand on Clear Weather:", clear_high_demand)
print("Probability:", prob_high_given_clear)
print("Percentage:", prob_high_given_clear * 100, "%")


# ------------------------------------------
# 3. Probability of High Demand
#    given Working Day
# ------------------------------------------

working_day = df[df["workingday"] == 1]

working_total = len(working_day)
working_high_demand = working_day["high_demand"].sum()

prob_high_given_working = working_high_demand / working_total

print("\n3. Probability of High Demand given Working Day:")
print("Working Day Cases:", working_total)
print("High Demand on Working Days:", working_high_demand)
print("Probability:", prob_high_given_working)
print("Percentage:", prob_high_given_working * 100, "%")


# ------------------------------------------
# 4. Probability of Low Demand
# ------------------------------------------

low_demand_cases = (~df["high_demand"]).sum()

prob_low_demand = low_demand_cases / total_cases

print("\n4. Probability of Low Demand:")
print("Low Demand Cases:", low_demand_cases)
print("Probability:", prob_low_demand)
print("Percentage:", prob_low_demand * 100, "%")


# ------------------------------------------
# Probability Bar Chart
# ------------------------------------------

probability_names = [
    "High Demand",
    "High | Clear Weather",
    "High | Working Day",
    "Low Demand"
]

probability_values = [
    prob_high_demand,
    prob_high_given_clear,
    prob_high_given_working,
    prob_low_demand
]

plt.figure(figsize=(10, 6))

plt.bar(probability_names, probability_values)

plt.title("Bike Rental Demand Probabilities")
plt.xlabel("Probability Event")
plt.ylabel("Probability")

plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("graphs/probability_analysis.png")
plt.show()
# ==========================================
# MODULE VII - HYPOTHESIS TESTING
# ==========================================

from scipy.stats import ttest_ind

print("\n====================================")
print("MODULE VII - HYPOTHESIS TESTING")
print("====================================")

# Separate working days and non-working days

working_demand = df[df["workingday"] == 1]["cnt"]

non_working_demand = df[df["workingday"] == 0]["cnt"]

# Calculate means

mean_working = working_demand.mean()
mean_non_working = non_working_demand.mean()

print("\nMean Demand on Working Days:", mean_working)
print("Mean Demand on Non-Working Days:", mean_non_working)

# Hypotheses

print("\nH0: Mean demand is the same for working and non-working days.")
print("H1: Mean demand is different for working and non-working days.")

# Perform independent t-test

t_statistic, p_value = ttest_ind(
    working_demand,
    non_working_demand,
    equal_var=False
)

print("\nT-Statistic:", t_statistic)
print("P-Value:", p_value)

# Significance level

alpha = 0.05

print("Significance Level:", alpha)

# Decision

if p_value < alpha:
    print("\nDecision: Reject H0")
    print("Conclusion: There is a statistically significant difference")
    print("in average bike rental demand between working and non-working days.")
else:
    print("\nDecision: Fail to Reject H0")
    print("Conclusion: There is no statistically significant difference")
    print("in average bike rental demand between working and non-working days.")

# ==========================================
# MODULE IX - CORRELATION
# ==========================================

print("\n====================================")
print("MODULE IX - CORRELATION")
print("====================================")

# Select important numerical variables
correlation_columns = [
    "hr",
    "temp",
    "hum",
    "windspeed",
    "cnt"
]

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Correlation with bike rental demand

print("\nCorrelation with Bike Rental Demand:")
print(correlation_matrix["cnt"].sort_values(ascending=False))


# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Bike Rental Demand")

plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png")
plt.show()


# ==========================================
# TEMPERATURE VS DEMAND
# ==========================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="temp",
    y="cnt"
)

plt.title("Temperature vs Bike Rental Demand")
plt.xlabel("Temperature")
plt.ylabel("Bike Rental Demand")

plt.tight_layout()
plt.savefig("graphs/temp_vs_demand.png")
plt.show()


# ==========================================
# HUMIDITY VS DEMAND
# ==========================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="hum",
    y="cnt"
)

plt.title("Humidity vs Bike Rental Demand")
plt.xlabel("Humidity")
plt.ylabel("Bike Rental Demand")

plt.tight_layout()
plt.savefig("graphs/humidity_vs_demand.png")
plt.show()
# ==========================================
# MODULE X - REGRESSION & PREDICTION
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n====================================")
print("MODULE X - REGRESSION & PREDICTION")
print("====================================")


# ------------------------------------------
# Select input features
# ------------------------------------------

features = [
    "hr",
    "season",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed"
]

X = df[features]
y = df["cnt"]


# ------------------------------------------
# Split data into training and testing
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))


# ------------------------------------------
# Create Linear Regression Model
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ------------------------------------------
# Make Predictions
# ------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------
# Model Evaluation
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation:")
print("-------------------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ------------------------------------------
# Regression Coefficients
# ------------------------------------------

print("\nRegression Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)


# ------------------------------------------
# Actual vs Predicted Graph
# ------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, alpha=0.4)

plt.xlabel("Actual Bike Rentals")
plt.ylabel("Predicted Bike Rentals")

plt.title("Actual vs Predicted Bike Rental Demand")

plt.tight_layout()

plt.savefig("graphs/actual_vs_predicted.png")

plt.show()


# ------------------------------------------
# Prediction Example
# ------------------------------------------

sample_data = pd.DataFrame({
    "hr": [17],
    "season": [2],
    "workingday": [1],
    "weathersit": [1],
    "temp": [0.6],
    "hum": [0.5],
    "windspeed": [0.2]
})

predicted_demand = model.predict(sample_data)[0]

print("\nExample Bike Rental Demand Prediction:")
print("Predicted Rentals:", round(predicted_demand))