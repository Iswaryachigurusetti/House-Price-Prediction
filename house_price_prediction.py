# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('house_data.csv')  # Make sure your CSV is in the same folder

# Check first few rows
print(data.head())

# Features and target
X = data[['area', 'bedrooms', 'bathrooms']]  # Use the columns you have
y = data['price']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Predict for new data example
new_house = [[2500, 4, 3]]  # area=2500, bedrooms=4, bathrooms=3
predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: {predicted_price[0]:.2f}")
