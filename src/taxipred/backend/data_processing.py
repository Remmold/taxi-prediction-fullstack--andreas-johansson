import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from taxipred.utils.constants import ORIGINAL_CSV_PATH, ALTERED_CSV_PATH ,COLUMNS


class TaxiData:
    def __init__(self):
        self.df = pd.read_csv(ORIGINAL_CSV_PATH)

    def to_json(self):
        return json.loads(self.df.to_json(orient="records"))

    def repair_data_using_algebra(self):
        """
        Repairs price and distance columns using a known fare calculation formula
        and logs the changes.

        This function modifies the DataFrame in place.
        """
        self._log_repair_status('Before Repair', [COLUMNS["PRICE"], COLUMNS["DISTANCE"]])

        price_components = [
            COLUMNS["BASE_FARE"], COLUMNS["DISTANCE"], COLUMNS["KM_RATE"],
            COLUMNS["DURATION"], COLUMNS["MIN_RATE"]
        ]
        mask_repair_price = self.df[COLUMNS["PRICE"]].isnull() & self.df[price_components].notnull().all(axis=1)

        if mask_repair_price.any():
            self.df.loc[mask_repair_price, COLUMNS["PRICE"]] = (
                self.df.loc[mask_repair_price, COLUMNS["BASE_FARE"]] +
                (self.df.loc[mask_repair_price, COLUMNS["DISTANCE"]] * self.df.loc[mask_repair_price, COLUMNS["KM_RATE"]]) +
                (self.df.loc[mask_repair_price, COLUMNS["DURATION"]] * self.df.loc[mask_repair_price, COLUMNS["MIN_RATE"]])
            )

        distance_components = [
            COLUMNS["BASE_FARE"], COLUMNS["KM_RATE"], COLUMNS["MIN_RATE"],
            COLUMNS["DURATION"], COLUMNS["PRICE"]
        ]
        mask_repair_distance = self.df[COLUMNS["DISTANCE"]].isnull() & self.df[distance_components].notnull().all(axis=1)
        mask_repair_distance &= (self.df[COLUMNS["KM_RATE"]] != 0)

        if mask_repair_distance.any():
            self.df.loc[mask_repair_distance, COLUMNS["DISTANCE"]] = (
                (self.df.loc[mask_repair_distance, COLUMNS["PRICE"]] -
                 self.df.loc[mask_repair_distance, COLUMNS["BASE_FARE"]] -
                 (self.df.loc[mask_repair_distance, COLUMNS["DURATION"]] * self.df.loc[mask_repair_distance, COLUMNS["MIN_RATE"]])) /
                self.df.loc[mask_repair_distance, COLUMNS["KM_RATE"]]
            )

        self._log_repair_status('After Repair', [COLUMNS["PRICE"], COLUMNS["DISTANCE"]])

    def fill_all_null(self, max_cat_values: int = 5):
        """
        This method takes a dataframe and attempts to fill null values inside all columns
        by iteratively training models on the non-null data.
        """
        cat_columns = self._find_categorical_columns(self.df, max_cat_values)

        while self.df.isnull().sum().sum() > 0:
            total_nans_before = self.df.isnull().sum().sum()

            cols_with_nulls = self.df.columns[self.df.isnull().any()].tolist()

            for col_name in cols_with_nulls:
                print(f"Attempting to fill column: {col_name}")

                if col_name in cat_columns:
                    predictions = self._fill_one_categorical_column(self.df, col_name, max_cat_values)
                else:
                    predictions = self._fill_one_numeric_column(self.df, col_name, max_cat_values)

                if not predictions.empty:
                    self.df.loc[predictions.index, col_name] = predictions
                    print(f"Filled {len(predictions)} values in {col_name}.")

            if self.df.isnull().sum().sum() == total_nans_before:
                print("Could not fill any more NaNs. Exiting.")
                break

    def _log_repair_status(self, stage: str, repaired_cols: list):
        """Internal helper function to log the status of missing values."""
        print(f"--- {stage} ---")
        print("Missing values in key columns:")
        print(self.df[repaired_cols].isnull().sum())
        print("-" * 25)

    def _fill_one_numeric_column(self, df: pd.DataFrame, target_column: str, max_cat_values: int = 5) -> pd.Series:
        """
        This function trains a model and returns a Series containing the
        predicted values for the nulls in the target column.
        """
        df_copy = df.copy()
        features_df = df_copy.drop(target_column, axis=1)
        cat_cols = self._find_categorical_columns(features_df, max_cat_values=max_cat_values)
        encoded_df = pd.get_dummies(data=features_df, columns=cat_cols)
        encoded_df[target_column] = df_copy[target_column]
        train_df = encoded_df[encoded_df[target_column].notnull()]
        predict_df = encoded_df[encoded_df[target_column].isnull()]

        if predict_df.empty:
            return pd.Series(dtype=float)

        X_train = train_df.drop(target_column, axis=1)
        y_train = train_df[target_column]
        X_predict = predict_df.drop(target_column, axis=1)

        clean_indices = X_train.dropna().index
        X_train_clean = X_train.loc[clean_indices]
        y_train_clean = y_train.loc[clean_indices]

        if X_train_clean.empty:
            return pd.Series(dtype=float)

        best_model = self._find_best_regression_model(X_train_clean, y_train_clean)

        X_predict = X_predict.reindex(columns=X_train_clean.columns, fill_value=0)

        predict_indices = X_predict.dropna().index
        X_predict_clean = X_predict.loc[predict_indices]

        if X_predict_clean.empty:
            return pd.Series(dtype=float)

        predictions = best_model.predict(X_predict_clean)
        return pd.Series(predictions, index=predict_indices)

    def _fill_one_categorical_column(self, df: pd.DataFrame, target_column: str, max_cat_values: int = 5) -> pd.Series:
        """
        Fills null values of one categorical column using a classification model
        and returns the predictions as a Series.
        """
        df_copy = df.copy()
        train_df_orig = df_copy[df_copy[target_column].notnull()]
        predict_df_orig = df_copy[df_copy[target_column].isnull()]

        if predict_df_orig.empty:
            return pd.Series(dtype=object)

        X = train_df_orig.drop(target_column, axis=1)
        y = train_df_orig[target_column]

        forward_map = self._map_category(y)
        inverse_map = self._inverse_dict(forward_map)
        y_encoded = y.map(forward_map)

        cat_cols = self._find_categorical_columns(X, max_cat_values=max_cat_values)
        X_encoded = pd.get_dummies(X, columns=cat_cols)

        clean_indices = X_encoded.dropna().index
        X_train_clean = X_encoded.loc[clean_indices]
        y_train_clean = y_encoded.loc[clean_indices]

        if X_train_clean.empty:
            return pd.Series(dtype=object)

        best_model = self._find_best_classification_model(X_train_clean, y_train_clean)

        X_predict = predict_df_orig.drop(target_column, axis=1)

        X_predict_encoded = pd.get_dummies(X_predict, columns=self._find_categorical_columns(X_predict, max_cat_values=max_cat_values))
        X_predict_encoded = X_predict_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        predict_indices = X_predict_encoded.dropna().index
        X_predict_clean = X_predict_encoded.loc[predict_indices]

        if X_predict_clean.empty:
            return pd.Series(dtype=object)

        numeric_predictions = best_model.predict(X_predict_clean)
        text_predictions = pd.Series(numeric_predictions, index=predict_indices).map(inverse_map)
        return text_predictions

    @staticmethod
    def _find_categorical_columns(df:pd.DataFrame, max_cat_values:int = 5) -> list[str]:
        """function figures out which columns should be treated as categorical"""
        category_columns = []
        for column_name in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column_name]) or df[column_name].nunique() < max_cat_values:
                category_columns.append(column_name)
        return category_columns

    @staticmethod
    def _find_best_regression_model(X, y):
        """Function tries both linear regression/random forest to predict column values and returns
        the model with lowest rsme"""
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42, train_size=0.7)
        lr_model = LinearRegression().fit(Xtrain, ytrain)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtrain, ytrain)
        linear_regression_rsme = TaxiData._calculate_rsme(lr_model, Xtest, ytest)
        random_forest_rsme = TaxiData._calculate_rsme(rf_model, Xtest, ytest)
        print(f"{linear_regression_rsme=}")
        print(f"{random_forest_rsme=}")
        return lr_model if linear_regression_rsme < random_forest_rsme else rf_model

    @staticmethod
    def _calculate_rsme(model, Xtest, ytest):
        y_pred = model.predict(Xtest)
        return root_mean_squared_error(y_pred=y_pred, y_true=ytest)

    @staticmethod
    def _find_best_classification_model(X, y):
        """note that currently only know how to operate 1 model. potentially refactor this later"""
        rf_model = RandomForestClassifier()
        rf_model.fit(X, y)
        return rf_model

    @staticmethod
    def _map_category(series:pd.Series) -> dict:
        mapping_dict = {category: i for i, category in enumerate(series.unique())}
        return mapping_dict

    @staticmethod
    def _inverse_dict(map:dict):
        inverse_map = {value: key for key, value in map.items()}
        return inverse_map
    
if __name__ == "__main__":
    taxi_data = TaxiData()
    print("Data before cleaning")
    print(taxi_data.df.info())
    print("Data after algebra cleaning")
    taxi_data.repair_data_using_algebra()
    print(taxi_data.df.info())
    print("Data after using machinelearning repair")
    taxi_data.fill_all_null(5)
    print(taxi_data.df.info())
    