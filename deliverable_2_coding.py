import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class PredictiveModelFF:
    def __init__(self, data):
        self.fastfood = pd.read_csv(data)

        # Getting X and Y variable. Here, based on X variable we will predict y variable.
        X = self.fastfood[
            ['cal_fat', 'total_fat', 'sat_fat', 'trans_fat', 'cholesterol', 'sodium', 'total_carb', 'fiber', 'sugar',
             'protein']]
        y = self.fastfood['calories']


        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                random_state=25)

        # Evaluate the linear regression model
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.mse_l = mean_squared_error(self.y_test, y_pred)
        self.r2_l = r2_score(self.y_test, y_pred)
        print('Linear Regression')
        print('Mean squared error:', self.mse_l)
        print('R^2 score:', round(self.r2_l * 100, 2), "%")
        print('\n')

        # Evaluate the SVM regression model
        model = SVR(kernel='linear')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.mse_s = mean_squared_error(self.y_test, y_pred)
        self.r2_s = r2_score(self.y_test, y_pred)
        print('SVM Regression')
        print('Mean squared error:', self.mse_s)
        print('R^2 score:', round(self.r2_s * 100, 2), "%")
        print('\n')

        # Evaluate the decision tree regression model
        model = DecisionTreeRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.mse_dct = mean_squared_error(self.y_test, y_pred)
        self.r2_dct = r2_score(self.y_test, y_pred)
        print('Decision Tree Regression')
        print('Mean squared error:', self.mse_dct)
        print('R^2 score:', round(self.r2_dct * 100, 2), "%")
        print('\n')

        # Evaluate the random forest regression model
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.mse_rfr = mean_squared_error(self.y_test, y_pred)
        self.r2_rfr = r2_score(self.y_test, y_pred)
        print('Random Forest Regression')
        print('Mean squared error:', self.mse_rfr)
        print('R^2 score:', round(self.r2_rfr * 100, 2), "%")
        print('\n')

        # Evaluate the gradient boosting regression model
        gbr = GradientBoostingRegressor()
        gbr.fit(self.X_train, self.y_train)
        self.y_pred = gbr.predict(self.X_test)
        self.mse_gbr = mean_squared_error(self.y_test, y_pred)
        self.r2_gbr = r2_score(self.y_test, y_pred)
        print('Gradient Boosting Regression')
        print('Mean squared error:', self.mse_gbr)
        print('R^2 score:', round(self.r2_gbr * 100, 2), "%")
        print('\n')

        # Saving all R2 value so that we can plot with it to see all model's score at once.
        r2_scores = {'Linear Regression': self.r2_l, 'SVM Regression': self.r2_s,
                     'Decision Tree Regression': self.r2_dct,
                     'Random Forest Regression': self.r2_rfr, 'Gradient Boosting Regression': self.r2_gbr}

        # Create a dataframe from the dictionary
        r2_df = pd.DataFrame.from_dict(r2_scores, orient='index', columns=['R2 Score'])

        # Create a bar plot from the dataframe
        plt.figure(figsize=(8, 6))
        sns.set_style('whitegrid')
        sns.barplot(x=r2_df.index, y='R2 Score', data=r2_df, color='lightblue')

        # Set the x-axis and y-axis labels
        plt.xlabel('Model')
        plt.ylabel('R2 Score')
        plt.title('Different Model and R2 Scores')
        plt.xticks(rotation=15)
        plt.show()


if __name__ == '__main__':
    predictive_model = PredictiveModelFF("fastfood_cleaned.csv")
