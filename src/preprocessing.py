"""
preprocessing.py

Preprocessing pipeline for the Credit Score dataset.

This module performs data cleaning, missing value handling,
anomaly handling, feature engineering, encoding, and feature
scaling based on the final preprocessing steps used in EDA.ipynb.

Data ingestion and train-test splitting are handled separately
to keep this module focused on data transformation only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Applies all preprocessing steps used in EDA.ipynb
    to ensure consistency between training and inference.
    """

    DROP_COLS = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN']

    COLS_TO_CONVERT = [
        'Age',
        'Annual_Income',
        'Num_of_Loan',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit',
        'Outstanding_Debt',
        'Amount_invested_monthly',
        'Monthly_Balance'
    ]

    MEDIAN_IMPUTE_COLS = [
        'Monthly_Inhand_Salary',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit',
        'Num_Credit_Inquiries',
        'Credit_History_Age',
        'Amount_invested_monthly',
        'Monthly_Balance'
    ]

    ANOMALY_COLS = [
        'Age',
        'Num_Bank_Accounts',
        'Num_Credit_Card',
        'Interest_Rate',
        'Num_of_Loan',
        'Num_Credit_Inquiries'
    ]

    IQR_COLS = [
        'Num_Bank_Accounts',
        'Num_Credit_Card',
        'Interest_Rate',
        'Num_of_Loan',
        'Num_Credit_Inquiries'
    ]

    NUMERICAL_COLS = [
        'Age',
        'Annual_Income',
        'Monthly_Inhand_Salary',
        'Num_Bank_Accounts',
        'Num_Credit_Card',
        'Interest_Rate',
        'Num_of_Loan',
        'Delay_from_due_date',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit',
        'Num_Credit_Inquiries',
        'Outstanding_Debt',
        'Credit_Utilization_Ratio',
        'Credit_History_Age',
        'Total_EMI_per_month',
        'Amount_invested_monthly',
        'Monthly_Balance'
    ]

    ONEHOT_COLS = ['Month', 'Occupation', 'Payment_Behaviour']

    CREDIT_MIX_MAP = {'Bad': 0, 'Standard': 1, 'Good': 2, 'Unknown': 3}
    PAYMENT_MAP = {'No': 0, 'Unknown': 1, 'Yes': 2}

    TARGET_COL = 'Credit_Score'

    def __init__(self):
        self.scaler = StandardScaler()
        self.loan_types = None      
        self.train_columns = None    

    #data cleaning
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        existing_drop_cols = [c for c in self.DROP_COLS if c in df.columns]
        df = df.drop(columns=existing_drop_cols)

        for col in self.COLS_TO_CONVERT:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[^0-9.-]', '', regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Credit_History_Age'] = df['Credit_History_Age'].apply(
            self._convert_credit_history
        )

        for col in self.MEDIAN_IMPUTE_COLS:
            df[col] = df[col].fillna(df[col].median())

        df['Type_of_Loan'] = df['Type_of_Loan'].fillna('Unknown')

        df['Occupation'] = df['Occupation'].replace('_______', 'Unknown')
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', 'Unknown')
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', 'Unknown')
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', 'Unknown')

        df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
        df.loc[df['Num_Bank_Accounts'] < 0, 'Num_Bank_Accounts'] = np.nan
        df.loc[df['Num_Credit_Card'] < 0, 'Num_Credit_Card'] = np.nan
        df.loc[df['Num_of_Loan'] < 0, 'Num_of_Loan'] = np.nan
        df.loc[df['Num_Credit_Inquiries'] < 0, 'Num_Credit_Inquiries'] = np.nan

        for col in self.IQR_COLS:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            df.loc[df[col] > upper_bound, col] = np.nan

        for col in self.ANOMALY_COLS:
            df[col] = df[col].fillna(df[col].median())

        df.loc[df['Monthly_Balance'] < -1e10, 'Monthly_Balance'] = np.nan
        df['Monthly_Balance'] = df['Monthly_Balance'].fillna(
            df['Monthly_Balance'].median()
        )

        return df

    @staticmethod
    def _convert_credit_history(value):
        if pd.isna(value):
            return np.nan
        try:
            years = int(value.split(' Years ')[0])
            months = int(value.split(' and ')[1].split(' Months')[0])
            return years * 12 + months
        except Exception:
            return np.nan

    #Feature engineering
    def feature_engineering(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()

        if fit or self.loan_types is None:
            all_loans = set()
            for value in df['Type_of_Loan'].dropna():
                value = value.replace(' and ', ', ')
                loans = value.split(', ')
                for loan in loans:
                    loan = loan.strip().rstrip(',')
                    all_loans.add(loan)

            self.loan_types = [
                loan for loan in sorted(all_loans)
                if loan not in ['Not Specified', 'Unknown']
            ]

        for loan in self.loan_types:
            col_name = loan.replace('-', '_').replace(' ', '_')
            df[col_name] = (
                df['Type_of_Loan']
                .str.contains(loan, na=False)
                .astype(int)
            )

        df = df.drop(columns='Type_of_Loan')

        return df
    
    #Encoding
    def encode(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, fit: bool = True):
        X_train = X_train.copy()
        X_train['Credit_Mix'] = X_train['Credit_Mix'].map(self.CREDIT_MIX_MAP)
        X_train['Payment_of_Min_Amount'] = X_train['Payment_of_Min_Amount'].map(self.PAYMENT_MAP)
        X_train = pd.get_dummies(X_train, columns=self.ONEHOT_COLS, drop_first=True)

        if X_test is not None:
            X_test = X_test.copy()
            X_test['Credit_Mix'] = X_test['Credit_Mix'].map(self.CREDIT_MIX_MAP)
            X_test['Payment_of_Min_Amount'] = X_test['Payment_of_Min_Amount'].map(self.PAYMENT_MAP)
            X_test = pd.get_dummies(X_test, columns=self.ONEHOT_COLS, drop_first=True)

            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

            if fit:
                self.train_columns = X_train.columns.tolist()

            return X_train, X_test


        if self.train_columns is not None:
            X_train = X_train.reindex(columns=self.train_columns, fill_value=0)

        return X_train


    #Scaling
    def scale(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, fit: bool = True):
        X_train = X_train.copy()

        if fit:
            X_train[self.NUMERICAL_COLS] = self.scaler.fit_transform(X_train[self.NUMERICAL_COLS])
        else:
            X_train[self.NUMERICAL_COLS] = self.scaler.transform(X_train[self.NUMERICAL_COLS])

        if X_test is not None:
            X_test = X_test.copy()
            X_test[self.NUMERICAL_COLS] = self.scaler.transform(X_test[self.NUMERICAL_COLS])
            return X_train, X_test

        return X_train

    #to pipeline
    def transform_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.clean(df)
        df = self.feature_engineering(df, fit=False)
        df = self.encode(df, X_test=None, fit=False)
        df = self.scale(df, X_test=None, fit=False)
        return df