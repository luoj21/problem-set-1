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
    logistic_regression.predict_lr_model()

    # PART 4: Call functions/instanciate objects from decision_tree
    decision_tree.predict_dt_model()

    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.show_calibration_results()


if __name__ == "__main__":
    main()