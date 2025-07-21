'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    preprocessing.run_preprocessing()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests = pd.read_csv('data/df_arrests.csv')
    logistic_regression.predict_lr_model(df_arrests)

    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
    df_arrests_test_lr = pd.read_csv('data/df_arrests_test_LR.csv')
    decision_tree.predict_dt_model(df_arrests_train, df_arrests_test_lr)

    # PART 5: Call functions/instanciate objects from calibration_plot
    df_arrests_test_LR = pd.read_csv('data/df_arrests_test_LR.csv')
    df_arrests_test_DT = pd.read_csv('data/df_arrests_test_DT.csv')
    calibration_plot.show_calibration_results(df_arrests_test_LR, df_arrests_test_DT)


if __name__ == "__main__":
    main()