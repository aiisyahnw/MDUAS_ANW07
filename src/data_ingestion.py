"""
data_ingestion.py

Data ingestion pipeline for the Credit Score dataset.
"""

import pandas as pd


class DataIngestion:

    def __init__(
        self,
        path: str = "data/data_D.csv"
    ):

        self.path = path


    def load_data(self) -> pd.DataFrame:

        df = pd.read_csv(
            self.path
        )

        return df