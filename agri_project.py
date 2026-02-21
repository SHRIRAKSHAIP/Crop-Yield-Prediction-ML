# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 09:09:55 2025

@author: Shri Raksha
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


df=pd.read_csv("C:/Users/Shri Raksha/anaconda3/crop_yield.csv")
print(df.head())

print(df.isnull().sum())

print(df.info())

print("Duplicates:",df.duplicated().sum())

print(df['Crop'].unique())


num_cols=["Rainfall_mm","Temperature_Celsius","Days_to_Harvest"]
corr = df[num_cols + ['Yield_tons_per_hectare']].corr()['Yield_tons_per_hectare'].sort_values(ascending=False)
print(corr)



#Visualization
#Fertilizer vs Yield

df.groupby('Fertilizer_Used')['Yield_tons_per_hectare'].mean().sort_values(ascending=False)

sns.barplot(x='Fertilizer_Used',y='Yield_tons_per_hectare',data=df)
plt.show()

df.groupby('Irrigation_Used')['Yield_tons_per_hectare'].mean().sort_values(ascending=False)

sns.barplot(x='Irrigation_Used',y='Yield_tons_per_hectare',data=df)
plt.show()

df.groupby('Soil_Type')['Yield_tons_per_hectare'].mean().sort_values(ascending=False)

sns.barplot(x='Soil_Type',y='Yield_tons_per_hectare',data=df)
plt.show()

df.groupby('Weather_Condition')['Yield_tons_per_hectare'].mean().sort_values(ascending=False)

sns.barplot(x='Weather_Condition',y='Yield_tons_per_hectare',data=df)
plt.show()

crop_mean = df.groupby('Crop')[['Rainfall_mm', 'Yield_tons_per_hectare']].mean().reset_index()

sns.scatterplot(
    x='Rainfall_mm',
    y='Yield_tons_per_hectare',
    hue='Crop',
    data=crop_mean,
    s=100
)

plt.title("Average Rainfall vs Average Yield by Crop")
plt.show()

pivot = df.pivot_table(
    values='Yield_tons_per_hectare',
    index='Region',
    columns='Crop',
    aggfunc='mean'
)


sns.heatmap(pivot, annot=True, cmap='YlGnBu')
plt.title("Average Yield by Region and Crop")
plt.show()


X_all = df.drop(columns=['Yield_tons_per_hectare'])
X_all = pd.get_dummies(X_all, drop_first=True)
y = df['Yield_tons_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(
    X_all.iloc[:100000], y.iloc[:100000], test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

rf.fit(X_train, y_train)



importance = pd.Series(
    rf.feature_importances_,
    index=X_all.columns
).sort_values(ascending=False)

print("Feature:",importance)

y_pred=rf.predict(X_test)
rmse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print("RMSE:",rmse)
print("r2_score:",r2)

plt.figure(figsize=(10,5))

importance.head(10).plot(kind='bar')

plt.title("Top 10 Important Features Affecting Crop Yield")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

rmse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression RMSE:", rmse_lr)
print("Linear Regression R2:", r2_lr)

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [rmse_lr, rmse],
    "R2 Score": [r2_lr, r2]
})

print(comparison)






