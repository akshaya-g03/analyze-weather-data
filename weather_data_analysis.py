import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
df = pd.read_csv('weather.csv')

# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 3: Data Visualization
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

# Step 4:  Data Analysis
# Calculate average MaxTemp
avg_max_temp = df['MaxTemp'].mean()
print(f'Average Max Temperature: {avg_max_temp}')

# Step 5: Data Visualization (Part 2)
# Visualize the distribution of MaxTemp
plt.figure(figsize=(10, 5))
sns.histplot(df['MaxTemp'], bins=30, kde=True)
plt.xlabel('Max Temperature')
plt.ylabel('Frequency')
plt.title('Distribution of Max Temperature')
plt.grid(True)
plt.show()

# Step 6: Advanced Analysis (e.g., predict Rainfall)
# Prepare the data for prediction
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and calculate the Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')

# Step 7: Conclusions and Insights
# Example: Identify the highest and lowest temperatures
highest_max_temp = df['MaxTemp'].max()
lowest_max_temp = df['MaxTemp'].min()
print(f'Highest Max Temperature: {highest_max_temp}, Lowest Max Temperature: {lowest_max_temp}')
