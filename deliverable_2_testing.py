# importing Necessary libraries
import pytest
from deliverable_2_coding import PredictiveModelFF


# Fixture for creating instance of PredictiveModelFF
@pytest.fixture
def model():
    return PredictiveModelFF("fastfood_cleaned.csv")


# Pytest case for linear regression model
def test_linear_regression(model):
    assert model.mse_l >= 0  # assert check the condition, if the condition is true the program will ryun otherwise
    assert model.r2_l >= 0  # it will stop


# Pytest case for SVM regression model
def test_svm_regression(model):
    assert model.mse_s >= 0
    assert model.r2_s >= 0


# Pytest case for Decision Tree regression model
def test_decision_tree_regression(model):
    assert model.mse_dct >= 0
    assert model.r2_dct >= 0


# Pytest case for Random Forest regression model
def test_random_forest_regression(model):
    assert model.mse_rfr >= 0
    assert model.r2_rfr >= 0


# Pytest case for Gradient Boosting regression model
def test_gradient_boosting_regression(model):
    assert model.mse_gbr >= 0
    assert model.r2_gbr >= 0
