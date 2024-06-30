import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('./csvs/train.csv')
test_data = pd.read_csv('./csvs/test.csv')

# print(train_data.head())

print(f"Shape of training data: {train_data.shape}")
print(f"Shape of test data: {test_data.shape}")

train_X = train_data.drop(columns=['Item_Outlet_Sales'], axis=1)
train_y = train_data['Item_Outlet_Sales']
test_X = test_data.drop(columns=['Item_Outlet_Sales'], axis=1)
test_y = test_data['Item_Outlet_Sales']

model = LinearRegression()
model.fit(train_X, train_y)

print(f"Coefficient of the model: {model.coef_}")
print(f"Intercept of the model: {model.intercept_}")

predicted = model.predict(train_X)

rmse_train = mean_squared_error(train_y, predicted)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

predicted_test = model.predict(test_X)
rmse_test = mean_squared_error(test_y, predicted_test)**(0.5)
print('\nRMSE on train dataset : ', rmse_test)
