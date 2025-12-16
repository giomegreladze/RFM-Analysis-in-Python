"""
RFM Analysis for Car Dealers

This script fetches car purchase and pricing data from a MySQL database,
cleans it up, calculates RFM metrics (Recency, Frequency, Monetary) for each dealer,
and outputs the results to an Excel file.

Logging is set up to track progress and errors in 'rfm_analysis.log'.

Make sure there is a 'config.json' in the working directory with the database
connection details before running this script.
"""

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import json
from pathlib import Path
from sqlalchemy.exc import ProgrammingError
import logging
import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column
from collections.abc import Mapping


CURRENT_DIR = Path(__file__).resolve().parent


logging.basicConfig(
    level = logging.INFO,
    format = "%(message)s. - %(lineno)d - %(asctime)s - %(levelname)s",
    filename = CURRENT_DIR / "rfm_analysis.log",
    filemode = "w"
)


class RfmAnalysis:
    """
    Handles fetching data, cleaning it, calculating RFM metrics, and validating schemas.

    You can use this class to get an overview of dealer performance and segment dealers
    based on their purchase behavior.
    """
    def __init__(self):
        """
        Loads database settings from config.json.
    
        Will raise an error if the config file isn't found.
        """
        if Path('config.json').exists():
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        else:
            logging.error("Configuration file 'config.json' not found.")
            raise FileNotFoundError("Configuration file 'config.json' not found.")

    
    def get_data_from_database(self, query, db_name) -> pd.DataFrame:
        """
        Runs a SQL query on the chosen database and returns the result as a DataFrame.
    
        db_name should match a key in your config.json. If the key is missing or
        there’s a problem with the query, it logs the error and returns None.
        """
        if db_name in self.config:
            db_params = self.config[db_name]
        else:
            logging.error(f"Database configuration for '{db_name}' not found in config.json.")
            raise KeyError(f"Database configuration for '{db_name}' not found in config.json.")

        try:
            db_url = URL.create(
                'mysql+mysqlconnector',
                host=db_params['host'],
                username=db_params['username'],
                password=db_params['password'],
                database=db_params['database'],
                port=db_params['port']
            )
            engine = create_engine(db_url)
            with engine.connect() as connection:
                logging.info(f"Successfully getting data: database: {db_name}, query: {query}.")
                return pd.read_sql(query, connection)
        except ProgrammingError as pe:
            logging.error(f"Programming error occurred: {pe}")
            return None
        
    def clean_car_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate VINs and makes sure date columns are proper datetime objects.
    
        Raises a ValueError if the dates can’t be converted.
        """
        df_cleaned = df.drop_duplicates('vin')
        try:
            df_cleaned['buy_date'] = pd.to_datetime(df_cleaned['buy_date'], errors='raise')
            df_cleaned['registration_date'] = pd.to_datetime(df_cleaned['registration_date'], errors='raise')
        except ValueError as ve:
            logging.error(f"Date format error in buy_date: {ve}")
            raise ValueError(f"Date format error in buy_date: {ve}")

        logging.info("Car info DataFrame cleaned successfully.")
        
        return df_cleaned
    

    def rfm_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Recency, Frequency, and Monetary metrics for each dealer.
    
        Returns a DataFrame sorted by overall RFM score, with columns for
        the number of cars sold, average sale value, last sale date,
        and individual RFM scores.
    
        Recency is how recently the dealer bought cars (lower is better),
        Frequency is how many cars they bought, and Monetary is their average spend.
        """
        df = (
            df.groupby(['dealer_name', 'registration_date'], as_index=False)
                .agg({
                    'buy_date': 'max',
                    'total_price': 'mean',
                    'vin': 'count'
                })
            .rename(columns={
                'buy_date': 'last_date',
                'total_price': 'avg_value',
                'vin': 'vin_count'
            })
            .round(0)
        )

        # Assign RFM scores from 1 to 5 based on quantiles, higher is better
        # Recency score: fewer days since last purchase -> higher score
        df['recency'] = pd.qcut((pd.Timestamp.now() - df['last_date']).dt.days, 5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
        df['frequency'] = pd.qcut(df['vin_count'], 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
        df['monetary'] = pd.qcut(df['avg_value'], 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)

        # Combine RFM scores into a single RFM score
        df['RFM_Score'] = df['recency'] + df['frequency'] + df['monetary']
        df = df[['dealer_name', 'registration_date', 'vin_count', 'avg_value', 'last_date', 'recency', 'frequency', 'monetary', 'RFM_Score']].copy()
        df = df.sort_values(by='RFM_Score', ascending=False)
        return df
    

    def check_schemas(self, dataframes: Mapping[str, pd.DataFrame]) -> None:
        """
        Checks that the given DataFrames match their expected schemas using Pandera.
        
        Pass in a dictionary where the keys are schema names and the values are
        the DataFrames to check. If any DataFrame doesn't match its schema,
        an error is logged and raised.
        """
        schemas = {
            'car_info': DataFrameSchema({
                'vin': Column(str),
                'dealer_name': Column(str),
                'registration_date': Column(pd.Timestamp),
                'buy_date': Column(pd.Timestamp),
                'total_price': Column(float)
            })
        }
        
        for name, df in dataframes.items():
            try:
                schemas[name].validate(df)
                logging.info(f"{name} DataFrame passed schema validation.")
            except pa.errors.SchemaError as e:
                logging.error(f"{name} DataFrame failed schema validation: {e}")
                raise pa.errors.SchemaError(f"{name} DataFrame failed schema validation: {e}")


def main():
    """
    Runs the full workflow: fetches data, cleans it, validates it,
    calculates RFM scores, and saves the results to Excel.

    Make sure no other program is using the Excel file, or it will fail.
    """
    rfm = RfmAnalysis()
    car_info = rfm.get_data_from_database(
        """
        Select ci.vin, 
               ut.name as dealer_name, 
               cast(ut.created_at as date) as registration_date,
               cast(ci.buy_date as date) as buy_date,
               p.total_price
        from car_info AS ci
        left join prices AS p ON ci.id = p.car_id
        left join users as ut ON ut.id = ci.user_id
        where cast(ci.buy_date as date) >= '2025-06-01'
        """
        , "glob")

    car_info = rfm.clean_car_info(car_info)

    # Validate schema
    rfm.check_schemas({"car_info": car_info})
    rfm_result = rfm.rfm_analysis(car_info)

    try:
        # Export to Excel without headers or index for a cleaner appearance
        rfm_result.T.reset_index().T.to_excel(CURRENT_DIR / "rfm_analysis_result.xlsx", index=False, header=False)
    except PermissionError as pe:
        logging.error(f"Permission error when saving Excel file: {pe}")
        raise PermissionError(f"Permission error when saving Excel file: {pe}")


if __name__=="__main__":
    main()
