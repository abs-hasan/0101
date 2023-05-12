import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class EDAAnalysis:
    def __init__(self, eda_df):
        self.eda_df = eda_df

    def explore_visualization_s(self):
        st.title('Exploratory Data Analysis and Predictive Model Building on Fast Food Data')

        st.write("\n")
        st.write("\n")
        st.subheader("Project Deliverable 1: Exploratory Data Analysis and Visualization Task ")
        st.write("\n")
        st.write("\n")

        st.write('This notebook analyses the nutritional data of the most well-known fast food restaurants.'
                 'Based on the number of calories and other nutritional components of the food items, the goal is to '
                 'dentify better alternatives. We hope that this analysis may help anyone wanting to choose healthier '
                 'foods at fast food outlets by offering insights and advice.')



        # Show first 5 rows
        st.write("\n")
        st.subheader('First 5 Rows:')
        st.write(self.eda_df.head())

        st.write("\n")
        st.subheader('Data Info:')
        st.write('restaurant: Name of restaurant')
        st.write('item: Name of item')
        st.write('calories: Number of calories (Cal)')
        st.write('cal_fat: Calories from fat')
        st.write('total_fat: Total fat')
        st.write('sat_fat: Saturated fat (g)')
        st.write('trans_fat: Trans fat (g)')
        st.write('cholesterol: Cholesterol (mg)')
        st.write('sodium: Sodium (mg)')
        st.write('total_carb: Total carbs')
        st.write('fiber: Fiber')
        st.write('sugar: Suger (g)')
        st.write('protein: Protein')
        st.write('vit_a: Vitamin A')
        st.write('vit_c: Vitamin C')
        st.write('calcium: Calcium')
        st.write('salad: Salad or not')

        st.write("\n")
        # check for missing values
        st.subheader('Missing Values:')
        st.write(self.eda_df.isnull().sum())
        st.write("We have missing value only in Fiber, protein, vit_a, vit_c and calcium")
        st.write("\n")

        # calculate the percentage of missing values in each column
        missing_value_ratio = (self.eda_df.isna().sum() / len(self.eda_df)) * 100
        st.write('Percentage of Missing Values in Each Column:')
        st.bar_chart(missing_value_ratio)
        st.write("vit_a, vit_c and calcium has massing values more than 40%, which is a lot. So, we will remove these."
                 "three.However, for Fiber, protein as the number is insignificant, we will use mean value to fill up "
                 "the."
                 "missing values.")
        st.write("\n")

        # drop columns with more than 40% missing values
        fast_food = self.eda_df.drop(['vit_a', 'vit_c', 'calcium'], axis=1)

        # Mean Computation when missing value is few [2]
        mean_fiber = fast_food['fiber'].mean()
        fast_food['fiber'].fillna(mean_fiber, inplace=True)

        mean_protein = fast_food['protein'].mean()
        fast_food['protein'].fillna(mean_protein, inplace=True)

        # looking for outliers by using box plots
        fig, ax = plt.subplots()
        sns.boxplot(data=fast_food, ax=ax)
        plt.xticks(rotation=45)
        plt.title("Box plot for Outliers")
        plt.xlabel("Columns")
        plt.ylabel("Value")
        st.subheader('Check outliers:')
        st.pyplot(fig)
        fig.savefig("fast_food_boxplot.png")
        st.write("For this analysis, removing outliers will impact the dataset, as few cases contain a lot of food "
                 "items."
                 "high nutrition's values compare to other food items")

        st.write("\n")

        # Check for duplicate rows
        st.subheader("Checking Duplicate")
        st.write('Checking for Duplicate Rows:')
        duplicates = fast_food.duplicated()
        st.write('Number of duplicate rows:', duplicates.sum())
        st.write("\n")

        # Drop duplicate rows
        fast_food.drop_duplicates(inplace=True)

        # Check for duplicate rows after dropping
        st.write('Checking for Duplicate Rows After Dropping:')
        duplicates_dropped = fast_food.duplicated()
        st.write('Number of duplicate rows:', duplicates_dropped.sum())
        st.write("\n")

        # display the summary statistics of the data
        st.subheader('Summary Statistics:')
        st.write(fast_food.describe())
        st.write("\n")

        # check the data types of the columns
        st.subheader('Data Types:')
        st.write(fast_food.dtypes)
        st.write("\n")

        self.fast_food = fast_food

        # Question 1
        st.subheader("Question 1 : Which restaurant has larger menu size?")

        # create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))

        # plot the bar chart in the left subplot
        data = self.fast_food['restaurant'].value_counts()
        x = data.index
        y = data.values
        axs[0].bar(x, y)
        axs[0].set_title('Q1 : Total count of items by Restaurant')
        axs[0].set_xlabel('Restaurant')
        axs[0].set_ylabel('Item count')
        axs[0].tick_params(axis='x', rotation=45)

        # plot the pie chart in the right subplot [3]
        axs[1].pie(self.fast_food['restaurant'].value_counts().values,
                   labels=self.fast_food['restaurant'].value_counts().index,
                   shadow=True, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
        axs[1].set_title('Q1: Percentage of Menu Items in Each Restaurant')

        # adjust the layout and display the figure
        fig.tight_layout()
        st.pyplot(fig)
        st.write("When comparing fast food menus, it is apparent that Taco Bell boasts a vast selection, with over 100 "
                 "items making up 22% of all items in the dataset. On the other hand, Chick Fil-A has a smaller menu, "
                 "accounting for only 5.3% of items.")
        st.write("\n")

        # Question 2
        st.subheader("Question 2: What is the average calorie count per restaurant ?")

        # compute the average calorie count per restaurant
        average_value = self.fast_food.groupby('restaurant')['calories'].mean().sort_values()

        # create a bar plot
        fig = plt.figure(figsize=(8, 6))
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'lightblue'
        plt.bar(average_value.index, average_value.values)
        plt.title('Q2 : Average Calorie Count per Restaurant')
        plt.xlabel('Restaurant')
        plt.ylabel('Calories')
        plt.xticks(rotation=45)  # rotate the x-axis labels by 45 degrees
        st.pyplot(fig)
        st.write("McDonald, Sonic and Burger king, all have around 600 as average calories")

        st.write("\n")

        # Question 3
        st.subheader("Question 03: What is the correlation between the calories and the fat ?")

        # Scatter plot of Sodium vs Total Fat
        fig = plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.fast_food, x="calories", y="total_fat")
        plt.title('Q3: Correlation between the calories and the fat')
        plt.xlabel('Calories')
        plt.ylabel('Total Fat')
        st.pyplot(fig)
        st.write("We can see a positive co-relation betwwen fat anf Calories.This result is not unexpected given that "
                 "fat is a macronutrient with a high caloric value and that foods heavy in fat frequently have high "
                 "caloric values as well. Because of this, it is crucial to consider calorie and fat intake while "
                 "choosing foods, especially when consuming fast food.")

        st.write("\n")

        # Question 4
        st.subheader("Question 4 : What is the distribution of total fat content ?")

        # Distribution of Total Fat Content
        fig = plt.figure(figsize=(8, 6))
        sns.histplot(data=self.fast_food, x="total_fat", kde=True)
        plt.title('Q4: Distribution of Total Fat Content')
        plt.xlabel('Total Fat (grams)')
        st.pyplot(fig)
        st.write("According to the plot, the majority of fast food items have a total fat level of between 0 and 50 "
                 "grammes. Furthermore, several of the outliers have extremely high fat content readings, "
                 "which suggests that some fast food items have a markedly greater fat content than others. Overall, "
                 "the plot shows that there is a long tail on the right side and a positive skew in the distribution "
                 "of total fat content in fast food.")

        st.write("\n")

        # Question 5
        st.subheader("Question 5 : Which restaurant has highest and lowest of total fat ?")

        # Restaurant with highest and lowest total fat
        max_fat = self.fast_food.groupby('restaurant')['total_fat'].max()
        min_fat = self.fast_food.groupby('restaurant')['total_fat'].min()
        st.write("\nRestaurant with highest total fat:")
        st.write(max_fat.idxmax())
        st.write("\nRestaurant with lowest total fat:")
        st.write(min_fat.idxmin())

        st.write("\n")

        # Saving the clean dataframe for later use.
        self.fast_food = fast_food
        #self.fast_food.to_csv('fastfood_cleaned.csv', index=False)

    def predictive_model(self):
        self.explore_visualization_s()

        # Define the features (X) and target variable (y)
        X = self.fast_food[
            ['cal_fat', 'total_fat', 'sat_fat', 'trans_fat', 'cholesterol', 'sodium', 'total_carb', 'fiber', 'sugar',
             'protein']]
        y = self.fast_food['calories']

        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                random_state=42)

        st.write('\n')
        st.write("\n")
        st.subheader("Project Deliverable 2: Predictive Analytics Task ")
        st.write("Lets build a model to see : Calorie amount of a food item based on other nutrition amounts.")
        st.write('\n')
        st.write('\n')

        # Linear Regression
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.mse_l = mean_squared_error(self.y_test, y_pred)
        self.r2_l = r2_score(self.y_test, y_pred)
        st.subheader('Linear Regression')
        st.write('Mean squared error:', self.mse_l)
        st.write('R^2 score:', round(self.r2_l * 100, 2), "%")
        st.write('\n')

        # SVM Regression
        model = SVR(kernel='linear')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.mse_s = mean_squared_error(self.y_test, y_pred)
        self.r2_s = r2_score(self.y_test, y_pred)
        st.subheader('SVM Regression')
        st.write('Mean squared error:', self.mse_s)
        st.write('R^2 score:', round(self.r2_s * 100, 2), "%")
        st.write('\n')

        # Decision Tree Regression
        model = DecisionTreeRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.mse_dct = mean_squared_error(self.y_test, y_pred)
        self.r2_dct = r2_score(self.y_test, y_pred)
        st.subheader('Decision Tree Regression')
        st.write('Mean squared error:', self.mse_dct)
        st.write('R^2 score:', round(self.r2_dct * 100, 2), "%")

        st.write('\n')

        # Evaluate the random forest regression model
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.mse_rfr = mean_squared_error(self.y_test, y_pred)
        self.r2_rfr = r2_score(self.y_test, y_pred)
        st.subheader('Random Forest Regression')
        st.write('Mean squared error:', self.mse_rfr)
        st.write('R^2 score:', round(self.r2_rfr * 100, 2), "%")
        st.write('\n')

        # Evaluate the gradient boosting regression model
        gbr = GradientBoostingRegressor()
        gbr.fit(self.X_train, self.y_train)
        self.y_pred = gbr.predict(self.X_test)
        self.mse_gbr = mean_squared_error(self.y_test, y_pred)
        self.r2_gbr = r2_score(self.y_test, y_pred)
        st.subheader('Gradient Boosting Regression')
        st.write('Mean squared error:', self.mse_gbr)
        st.write('R^2 score:', round(self.r2_gbr * 100, 2), "%")
        st.write('\n')

        # Saving all R2 value so that we can plot with it to see all model's score at once.
        r2_scores = {'Linear Regression': self.r2_l, 'SVM Regression': self.r2_s,
                     'Decision Tree Regression': self.r2_dct,
                     'Random Forest Regression': self.r2_rfr, 'Gradient Boosting Regression': self.r2_gbr}

        # Create a dataframe from the dictionary
        r2_df = pd.DataFrame.from_dict(r2_scores, orient='index', columns=['R2 Score'])

        st.write('**R2 Scores:**')

        # Create a bar plot from the dataframe
        fig = plt.figure(figsize=(8, 6))
        sns.set_style('whitegrid')
        sns.barplot(x=r2_df.index, y='R2 Score', data=r2_df, color='lightblue')

        # Set the x-axis and y-axis labels
        plt.xlabel('Model')
        plt.ylabel('R2 Score')
        plt.title('Different Model and R2 Scores')
        plt.xticks(rotation=45)

        # Show the plot
        st.pyplot(fig)
        st.write('After analyzing the performance of different models to identify the calorie amount of a food based '
                 'on other nutritional amounts, it can be concluded that Linear Regression and SVM Regression '
                 'performed better compared to others. These models can be utilized to make predictions about calorie '
                 'amounts based on other nutritional values, which can be useful in various applications such as '
                 'health monitoring and meal planning. ')


if __name__ == '__main__':
    # load the CSV file into a pandas DataFrame
    fast_food_df = pd.read_csv('fastfood.csv')
    fast_food_analysis = EDAAnalysis(fast_food_df)
    fast_food_analysis.explore_visualization_s()
    fast_food_analysis.predictive_model()

