import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# =========================
# STEP 1: LOAD DATA
# =========================
df = pd.read_csv("train.csv")

print("Dataset Preview:")
print(df.head())


# =========================
# STEP 2: SELECT FEATURES
# =========================
X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]


# =========================
# STEP 3: SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# STEP 4: TRAIN MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully ✅")


# =========================
# STEP 5: PREDICT
# =========================
y_pred = model.predict(X_test)


# =========================
# STEP 6: EVALUATE
# =========================
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))


# =========================
# STEP 7: NEW PREDICTION
# =========================
new_house = pd.DataFrame([[2000, 3, 2]],
                         columns=["GrLivArea", "BedroomAbvGr", "FullBath"])

predicted_price = model.predict(new_house)

print("\nPredicted House Price:", predicted_price[0])


# =========================
# STEP 8: GRAPHS
# =========================

# 📊 1. Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()


# 📊 2. Error Distribution
errors = y_test - y_pred

plt.figure(figsize=(6,4))
sns.histplot(errors, bins=30, kde=True)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.show()


# 📊 3. Area vs Price
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["GrLivArea"], y=df["SalePrice"])
plt.title("GrLivArea vs SalePrice")
plt.xlabel("Living Area")
plt.ylabel("Price")
plt.show()


# 📊 4. Correlation Heatmap
plt.figure(figsize=(6,4))
corr = df[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()