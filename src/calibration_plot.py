'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score
# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()


def show_calibration_results(df_arrests_test_LR,  df_arrests_test_DT):
    """ Plots the calibration plots based off of results from the logistic regression and decision tree
    
    Parameters:
    - df_arrests_test_LR: testing data that contains the predicted outcome for the logistic regression
    - df_arrests_test_DT: testing data that contains the predicted outcome for the decision tree
    
    Returns:
    - None"""

    # Calibration plot for logistic regression
    calibration_plot(df_arrests_test_LR['y'], df_arrests_test_LR['pred_lr_prob_1'], n_bins=5)

    #  Calibration plot for decision Tree
    calibration_plot(df_arrests_test_DT['y'], df_arrests_test_DT['pred_dt_prob_1'], n_bins=5)

    print("The decision tree is more calibrated based off the plots.")

    print(f"The AUC of the logistic regression is: {roc_auc_score(df_arrests_test_LR['y'], df_arrests_test_LR['pred_lr_prob_1'])}")
    print(f"The AUC of the decision tree is: {roc_auc_score(df_arrests_test_DT['y'], df_arrests_test_DT['pred_dt_prob_1'])}")