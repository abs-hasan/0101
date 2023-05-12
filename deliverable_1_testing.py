# import the ExploratoryDataAnalysisAndVisualization class from the part1 module
from deliverable_1_coding import ExploratoryDataAnalysisAndVisualization


# define a function to test the explore_visualization method of the ExploratoryDataAnalysisAndVisualization class
def test_pytest():
    # create an instance of the ExploratoryDataAnalysisAndVisualization class with the fastfood.csv file as input
    testing = ExploratoryDataAnalysisAndVisualization('fastfood.csv')

    # assert that the explore_visualization method returns None
    assert testing.explore_visualization() is None
