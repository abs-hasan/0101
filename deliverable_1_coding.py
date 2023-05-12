# Importing Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Creating Class
class ExploratoryDataAnalysisAndVisualization:
    def __init__(self, data):
        self.fast_food = pd.read_csv(data)

    # Creating Method
    def explore_visualization(self):

        # Show first 5 rows
        print(self.fast_food.head())
        # Printing Database Info
        print(self.fast_food.info())

        # Check for duplicate rows
        print("Before dropping duplicates:")
        duplicates = self.fast_food.duplicated()
        print('Number of duplicate rows:', duplicates.sum())

        # Drop duplicate rows
        self.fast_food.drop_duplicates(inplace=True)

        # Check for duplicate rows after dropping
        print("\nAfter dropping duplicates:")
        duplicates_dropped = self.fast_food.duplicated()
        print('Number of duplicate rows:', duplicates_dropped.sum())

        # check for missing values
        print(self.fast_food.isnull().sum())

        # calculate the total missing values in each column
        missing_value_ratio = (self.fast_food.isna().sum() / len(self.fast_food)) * 100
        print(missing_value_ratio)

        # Plotting the missing values [1]
        plt.figure(figsize=(10, 7))
        plt.bar(x=missing_value_ratio.index, height=missing_value_ratio.values)
        plt.xticks(rotation=90)
        plt.xlabel('Columns')
        plt.ylabel('Percentage of Missing Values')
        plt.title('Percentage of Missing Values in Fast Food Dataset')
        plt.show(block=False)

        # drop columns which has more than 30 % missing values
        fast_food = self.fast_food.drop(['vit_a', 'vit_c', 'calcium'], axis=1)
        fast_food = self.fast_food

        # Mean Computation when missing value is few [2]
        mean_fiber = fast_food['fiber'].mean()
        fast_food['fiber'].fillna(mean_fiber, inplace=True)

        mean_protein = fast_food['protein'].mean()
        fast_food['protein'].fillna(mean_protein, inplace=True)

        # looking for outliers by using box plots
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=fast_food)
        plt.xticks(rotation=45)
        plt.title("Box plot for Outliers")
        plt.xlabel("Columns")
        plt.ylabel("Value")
        plt.show(block=False)

        # display the summary statistics of the data
        fast_food.describe()

        # check the data types of the columns
        print(fast_food.dtypes)

        # Question 1 : Which restaurant has larger menu size

        # create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))

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
        plt.show(block=False)

        # Question 2: What is the average calorie count per restaurant ?

        average_value = self.fast_food.groupby('restaurant')['calories'].mean().sort_values()

        # create a bar plot
        plt.figure(figsize=(7, 5))
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'lightblue'
        plt.bar(average_value.index, average_value.values)
        plt.title('Q2 : Average Calorie Count per Restaurant')
        plt.xlabel('Restaurant')
        plt.ylabel('Calories')
        plt.xticks(rotation=45)  # rotate the x-axis labels by 45 degrees
        plt.show(block=False)

        # Question 03: What is the correlation between the calories and the fat ?

        # create a scatter plot
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=self.fast_food, x="calories", y="total_fat")
        plt.title('Q3: Correlation between the calories and the fat')
        plt.xlabel('Calories)')
        plt.ylabel('Total Fat)')
        plt.show(block=False)

        # Question 4 : What is the distribution of total fat content ?

        # creating a hist plot
        plt.figure(figsize=(7, 5))
        sns.histplot(data=self.fast_food, x="total_fat", kde=True)
        plt.title('Q4: Distribution of Total Fat Content')
        plt.xlabel('Total Fat (grams)')
        plt.show()

        # Question 5 : Which restaurant has highest and lowest of total fat?
        max_fat = self.fast_food.groupby('restaurant')['total_fat'].max()
        min_fat = self.fast_food.groupby('restaurant')['total_fat'].min()
        print("\nRestaurant with highest total fat:")
        print(max_fat.idxmax())
        print("\nRestaurant with lowest total fat:")
        print(min_fat.idxmin())

        # Saving the clean dataframe for later use.
        self.fast_food = fast_food
        self.fast_food.to_csv('fastfood_cleaned.csv', index=False)


if __name__ == '__main__':
    eda_data = ExploratoryDataAnalysisAndVisualization("fastfood.csv")
    eda_data.explore_visualization()
