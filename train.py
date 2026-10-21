from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score 
import joblib

data = fetch_california_housing()
df= pd.DataFrame(data.data, columns=data.feature_names)
df["price"] = data.target

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae * 100000:,.0f}")
print(f"R-squared: {r2}")

joblib.dump(model, "house_model.joblib")
joblib.dump(list(X.columns), "house_features.joblib")