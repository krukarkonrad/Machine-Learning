#TODO: write python script to train linear regression model
# that learn price estimation on Boston market data.
# Check learned model with mean squared error and variance score (r2).
# Check predicted value for house with id=5 from test data.
# How this estimation differ from real value? Evaluate linear model with cross validation.

from sklearn.datasets import load_boston
boston_market_data = load_boston()

from sklearn.model_selection import train_test_split
train_data, test_data, \
train_target, test_target = \
train_test_split(boston_market_data['data'], boston_market_data['target'], test_size=0.1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_data, train_target)

id = 5
print(lr.predict(test_data[id,:].reshape(1,-1)))

lrp = lr.predict(test_data)
print(lrp)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(test_target, lr.predict(test_data)))

from sklearn.metrics import r2_score
print(r2_score(test_target, lr.predict(test_data)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(), boston_market_data['data'], boston_market_data['target'], cv=4)
print(scores)