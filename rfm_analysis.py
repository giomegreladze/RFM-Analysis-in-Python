"""
This module performs RFM (Recency, Frequency, Monetary) analysis on car dealers.

Transaction and vehicle purchase data are retrieved from a MySQL database using
SQLAlchemy. The data is cleaned, validated, and aggregated to calculate RFM
metrics for each dealer. Based on these metrics, dealers are segmented to help
analyze purchasing behavior and customer value.

The final RFM results are exported to an Excel file for further analysis or
reporting.

Logging:
    All execution logs are written to the 'rfm_analysis.log' file.

Configuration:
    Database connection parameters are stored in the 'config.json' file, which
    must be present in the working directory.
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
    Performs RFM Analysis on car dealers.

    The class retrieves transaction data from a MySQL database using SQLAlchemy,
    calculates Recency, Frequency and Monetary (RFM) metrics segments dealers
    based on these metrics, and exports the results to an Excel file.
    """
    def __init__(self):
        """
        Initializes the class by loading MySQL database connection parameters from a 'config.json' file.

        Raises:
            FileNotFoundError: If the 'config.json' file does not exist in the current directory.

        Notes:
            The 'config.json' file must be present in the working directory and contain the necessary
            configuration details for connecting to the MySQL database. If the file is missing,
            an error is logged and an exception is raised.
        """
        if Path('config.json').exists():
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        else:
            logging.error("Configuration file 'config.json' not found.")
            raise FileNotFoundError("Configuration file 'config.json' not found.")

    
    def get_data_from_database(self, query, db_name) -> pd.DataFrame:
        """
        Executes a SQL query on the specified database and returns the result as a pandas DataFrame.
    
        This method retrieves data from a MySQL database using SQLAlchemy. It reads the 
        database connection parameters from the loaded configuration (`config.json`) and 
        executes the provided SQL query. Execution details are logged.
    
        Parameters:
        query: 
            str: A valid SQL query string to execute on the database.
        db_name: 
            str: The key in the configuration dictionary that specifies which database 
            connection parameters to use.
    
        Returns:
            pd.DataFrame: A pandas DataFrame containing the query results. Returns None if a 
            programming error occurs during query execution.
    
        Raises:
            KeyError: If the provided `db_name` does not exist in the configuration.
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
        Cleans and preprocesses car information data.
    
        This method removes duplicate vehicle records based on VIN and ensures
        that date-related columns are properly converted to pandas datetime
        objects.
    
        Parameters:
            df (pd.DataFrame): Input DataFrame containing car information.
                Required columns:
                    - 'vin'
                    - 'buy_date'
                    - 'registration_date'
    
        Returns:
            pd.DataFrame: Cleaned DataFrame with duplicate VINs removed and
            standardized datetime columns.
    
        Raises:
            ValueError: If date conversion fails for 'buy_date' or
            'registration_date'.
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
        Performs RFM (Recency, Frequency, Monetary) analysis on a transactional 
        DataFrame to segment dealers based on their purchasing behavior.

        Parameters
        df: pd.DataFrame
            Input DataFrame containing the following columns:
                - 'dealer_name': Identifier for each dealer.
                - 'registration_date': Date when the dealer registered (datetime).
                - 'buy_date': Date of each purchase (datetime).
                - 'total_price': Monetary value of each transaction.
                - 'vin': Unique identifier for each vehicle purchased.
        Returns
        pd.DataFrame
            DataFrame with aggregated RFM metrics for each dealer, including:
                - 'dealer_name': Dealer identifier.
                - 'registration_date': Dealer registration date.
                - 'vin_count': Total number of vehicles purchased.
                - 'avg_value': Average transaction value.
                - 'last_date': Date of most recent purchase.
                - 'recency': Recency score (1-5, higher is better).
                - 'frequency': Frequency score (1-5, higher is better).
                - 'monetary': Monetary score (1-5, higher is better).
                - 'RFM_Score': Combined RFM score (sum of recency, frequency, and monetary).
        Notes
        - Recency is calculated as the number of days since the last purchase; lower values are better and receive higher scores.
        - Dealers registered within the last 30 days are assigned the lowest frequency score.
        - Frequency is based on the average number of purchases per day since registration; higher values are better.
        - Monetary is based on the average transaction value; higher values are better.
        - RFM scores are assigned using quintiles (pd.qcut) to ensure balanced segmentation.
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
        Validates the schemas of provided DataFrames against predefined rules.
    
        This method uses Pandera to ensure that each DataFrame matches the expected
        schema. If a DataFrame does not conform, an error is logged and a SchemaError
        is raised.
    
        Parameters:
            dataframes (Mapping[str, pd.DataFrame]): A dictionary mapping names
                (str) to pandas DataFrames to be validated. The keys should match
                the predefined schema names.
    
        Raises:
            pandera.errors.SchemaError: If any DataFrame does not match its
                predefined schema.
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
    Executes the end-to-end RFM analysis workflow and saves results to Excel.

    Workflow:
        1. Initializes the RfmAnalysis class, loading database configuration.
        2. Retrieves car purchase data from the MySQL database using a SQL query.
        3. Cleans and preprocesses the retrieved data.
        4. Validates the DataFrame schema to ensure correctness.
        5. Performs RFM (Recency, Frequency, Monetary) analysis to segment dealers.
        6. Saves the resulting RFM metrics to an Excel file ('rfm_analysis_result.xlsx').

    Outputs:
        An Excel file with aggregated RFM metrics and dealer segmentation.

    Raises:
        PermissionError: If the Excel file cannot be written due to file access issues.
        Any exceptions raised by RfmAnalysis methods (e.g., database errors, schema validation errors).
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
