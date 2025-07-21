'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score



def create_dt_model(X_train, y_train):
    """ Creates decision tree with parameters optimal parameters set by 
    Grid Search CV
    
    Parameters;
    - X_train: training data of features
    - y_train: training labels
    
    Returns:
    - gs_cv_dt: Fitted Decision Tree model
    
    """
    
    param_grid = {'max_depth': [100, 200, 500]}
    dt_model = DTC(random_state=10)
    gs_cv_dt = GridSearchCV(estimator=dt_model,
        param_grid=param_grid,
        cv=5)

    gs_cv_dt.fit(X_train, y_train)
    optimal_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"What was the optimal value for max depth? Answer: {optimal_max_depth}")

    if optimal_max_depth == min(param_grid['max_depth']):
        reg_strength = "Most regularization"
    elif optimal_max_depth == max(param_grid['max_depth']):
        reg_strength = "Least regularization"
    else:
        reg_strength = "Medium regularization"

    print(f"How much regularization? Answer: {reg_strength}")

    return gs_cv_dt




def predict_dt_model(df_arrests_train, df_arrests_test):
    """Imports data and uses trained decision tree for prediction. Saves decision tree results as a .csv
    
    Parameters:
    - df_arrests_train: training data that was created from the train/test/split in logistic_regression module
    - df_arrests_test: testing data that was created from the train/test/split in the logisitc_regression module
    
    Returns:
    - None
    """
    
    features = ['num_fel_arrests_last_year', 'current_charge_felony']

    X_train = df_arrests_train[features]
    y_train = df_arrests_train['y']
    X_test = df_arrests_test[features]
    y_test = df_arrests_test['y']

    # train and test a decision tree
    gs_cv_dt = create_dt_model(X_train, y_train)
    df_arrests_test['pred_dt'] = gs_cv_dt.predict(X_test)

    # new features that represent predicted probability of having a future felony crime in the next 365 days (1) or not (0)
    df_arrests_test['pred_dt_prob_0'] = gs_cv_dt.predict_proba(X_test)[:,0]
    df_arrests_test['pred_dt_prob_1'] = gs_cv_dt.predict_proba(X_test)[:,1]
    df_arrests_test.drop(columns = ['pred_lr_prob_0', 'pred_lr_prob_1'], inplace = True)
    
    print(f"Testing accuracy for decision tree: {accuracy_score(y_test, pd.Series(df_arrests_test['pred_lr']))}")

    # export decision tree test results
    df_arrests_test.to_csv('data/df_arrests_test_DT.csv', index = False)