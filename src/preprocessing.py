'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding

- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`

- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?

- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?

- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta


# Your code here
def create_y(row):
    """ Checks if a person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`
    
    Parameters:
    - row: a row of the `df_arrests` dataframe
    
    Returns:
    - 1 if a person was arrested for a felony crime 365 days after initial arrest date, 0 else"""

    lower_date = row['arrest_date_current'] + relativedelta(days=1)
    upper_date = row['arrest_date_current'] + relativedelta(years=1)
    
    if lower_date <= row['arrest_date_event'] <= upper_date and row['charge_degree'] == 'felony':
        return 1
    else:
        return 0
    

def create_arrested_last_year(row):
    """ Checks if a person was arrested for a felony crime in the 365 days prior to their current charge`
    
    Parameters:
    - row: a row of the `df_arrests` dataframe
    
    Returns:
    - 1 if a person was arrested for a felony crime 365 days prior to their current charge, 0 else"""

    lower_date = row['arrest_date_current'] - relativedelta(years = 1)
    upper_date = row['arrest_date_current'] - relativedelta(days = 1)
    
    if lower_date <= row['arrest_date_event'] <= upper_date and row['charge_degree'] == 'felony':
        return 1
    else:
        return 0


def run_preprocessing():
    """Preprocesses data, creating new features `y` and `num_fel_arrests_last_year` for 
    each person in `pred_universe_raw`. Exports transformed data as `df_arrests.csv`"""

    # read in csv files
    pred_universe = pd.read_csv('data/pred_universe_raw.csv')
    arrest_events= pd.read_csv('data/arrest_events_raw.csv')

    # perform full outer join and change column names for readability
    df_arrests = pd.merge(arrest_events, pred_universe, on = 'person_id', how = 'outer', suffixes=('_event', '_current'))
    df_arrests.rename(columns={'arrest_date_univ': 'arrest_date_current'}, inplace= True)
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])
    df_arrests['arrest_date_current'] = pd.to_datetime(df_arrests['arrest_date_current'])

    # create 'y' feature column
    df_arrests['y'] = df_arrests.apply(create_y, axis = 1)
    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? Answer: {np.sum(df_arrests.groupby('person_id')['y'].max())} \n")

    # create 'current_charge_feloy' feature column
    df_arrests['current_charge_felony'] = df_arrests.apply(lambda row: 1 if row['charge_degree'] == "felony" and row['arrest_date_event'] == row['arrest_date_current'] else 0, axis = 1)
    print(f"What share of current charges are felonies? Answer: {np.sum(df_arrests.groupby('person_id')['current_charge_felony'].max())} \n")


    # create 'was_arrested_last_year' feature column, which is 1 if a person was arrested last year for a felony else 0
    df_arrests['was_arrested_last_year'] = df_arrests.apply(create_arrested_last_year, axis = 1)
    print(f"What is the average number of felony arrests in the last year? Answer: {np.sum(df_arrests.groupby('person_id')['was_arrested_last_year'].max()) / 365} \n")

    # getting relavant features
    df_arrests = df_arrests[['person_id',
                            'arrest_id_current',
                            'age_at_arrest',
                            'sex',
                            'race',
                            'arrest_date_current',
                            'y',
                            'current_charge_felony',
                            'was_arrested_last_year']]
    
    # create num_fel_arrests_last_year column and tidying df_arrests data frame

    # obtain how many times a person was arrested last year for a felony
    num_fel_arrests_last_year = df_arrests.groupby('person_id')[['was_arrested_last_year']].sum()
    df_arrests.drop(columns='was_arrested_last_year', inplace = True)

    # tidy data, adding in the new num_fel_arrests_last_year feature
    df_arrests = df_arrests.groupby('person_id').agg({'person_id': 'first',
                                                    'arrest_id_current': 'first',
                                                    'age_at_arrest': 'first',
                                                    'sex': 'first',
                                                    'race': 'first',
                                                    'arrest_date_current': 'first',
                                                    'y': 'max',
                                                    'current_charge_felony': 'max'})
    df_arrests['num_fel_arrests_last_year'] = num_fel_arrests_last_year

    print(df_arrests.head())

    # export df_arrests to data/
    df_arrests.to_csv('data/df_arrests.csv', index = False)