'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score


# Your code here
def create_lr_model(X_train, y_train):
    """ Creates logistic regression with parameters optimal parameters set by 
    Grid Search CV
    
    Parameters;
    - X_train: training data of features
    - y_train: training labels
    
    Returns:
    - gs_cv: Fitted LR model
    
    """
    param_grid = {'C': [0.01, 0.10, 10]}
    lr_model = lr()

    gs_cv = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy')

    gs_cv.fit(X_train, y_train)
    optimal_c = gs_cv.best_params_['C']
    print(f"What was the optimal value for C? Answer: {optimal_c}")

    if optimal_c == min(param_grid['C']):
        reg_strength = "Most regularization"
    elif optimal_c == max(param_grid['C']):
        reg_strength = "Least regularization"
    else:
        reg_strength = "Medium regularization"

    print(f"How much regulariation? Answer: {reg_strength}")

    return gs_cv



def predict_lr_model():
    """Imports data and uses trained decision tree for prediction"""

    df_arrests = pd.read_csv('data/df_arrests.csv')

    # get relavant features from df_arrests and perform train/test/split
    df_arrests_train, df_arrests_test = train_test_split(df_arrests, test_size = 0.3, shuffle = True, stratify=df_arrests['y'])
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    X_train = df_arrests_train[features]
    y_train = df_arrests_train['y']
    X_test = df_arrests_test[features]
    y_test = df_arrests_test['y']

    # train and test a logistic regression
    gs_cv = create_lr_model(X_train, y_train)
    df_arrests_test['pred_lr'] = gs_cv.predict(X_test)
    df_arrests_test['pred_lr_prob_0'] = gs_cv.predict_proba(X_test)[:,0]
    df_arrests_test['pred_lr_prob_1'] = gs_cv.predict_proba(X_test)[:,1]

    print(f"Testing accuracy for logistic regression: {accuracy_score(y_test, pd.Series(df_arrests_test['pred_lr']))}")

    # export training data and results of logistic regression on testing data
    df_arrests_train.to_csv('data/df_arrests_train.csv', index = False)
    df_arrests_test.to_csv('data/df_arrests_test_LR.csv', index = False)
