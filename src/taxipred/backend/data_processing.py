import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error,r2_score,mean_absolute_error,mean_squared_error
from taxipred.utils.constants import ORIGINAL_CSV_PATH, ALTERED_CSV_PATH ,ALGEBRA_COLUMNS,FEATURES_COLUMNS,DATA_PATH
from pydantic import BaseModel
import joblib
class Trip(BaseModel):
    Trip_Distance_km:float
    Time_of_Day:str
    Day_of_Week:str
    Passenger_Count:int
    Traffic_Conditions:str
    Weather:str

class TaxiData:
    def __init__(self,path):
        self.df = pd.read_csv(path)

    def to_json(self):
        return json.loads(self.df.to_json(orient="records"))
    def drop_columns(self,columns_to_drop:list[str]):
        self.df.drop(columns=columns_to_drop,inplace=True)
    def to_csv(self,path):
        self.df.to_csv(path,index = False)
    def predict_trip_price(self, trip: Trip):
        # Define the same fixed list of categorical features
        categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']

        # Load the trained model and the training columns
        model = joblib.load(DATA_PATH / "taxi_model.joblib")
        training_columns = joblib.load('training_columns.joblib')

        # Create a one-row DataFrame from the input
        features_df = pd.DataFrame([trip.model_dump()])
        
        # Encode only the specified categorical features
        encoded_df = pd.get_dummies(features_df, columns=categorical_features)
        
        # Reindex to ensure the columns perfectly match the training columns
        encoded_df = encoded_df.reindex(columns=training_columns, fill_value=0)
        
        # Make the prediction
        y_pred = model.predict(encoded_df)
        
        # Return the result
        return y_pred
    
    def train_model(self):
        # Define your categorical features as a fixed list
        categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
        
        # Create a features DataFrame
        features_df = self.df[FEATURES_COLUMNS]
        
        # Encode only the specified categorical features
        X_encoded = pd.get_dummies(features_df, columns=categorical_features)
        y = self.df["Trip_Price"]
        
        # Save the exact column names the model was trained on
        training_columns = X_encoded.columns
        joblib.dump(training_columns, 'training_columns.joblib')

        # Train the model using the encoded data
        model = self._find_best_regression_model(X=X_encoded, y=y)
        
        # Save the trained model
        joblib.dump(model, DATA_PATH / "taxi_model.joblib")



    def repair_data_using_algebra(self):
        """
        Repairs all fare-related columns using a known fare calculation formula.
        This function iteratively fills missing values until no more repairs can be made.

        This function modifies the DataFrame in place.
        """
        # Log the state before any repairs
        all_algebra_cols = [
            ALGEBRA_COLUMNS["PRICE"], ALGEBRA_COLUMNS["DISTANCE"], ALGEBRA_COLUMNS["BASE_FARE"],
            ALGEBRA_COLUMNS["KM_RATE"], ALGEBRA_COLUMNS["DURATION"], ALGEBRA_COLUMNS["MIN_RATE"]
        ]
        self._log_repair_status('Before Algebraic Repair', all_algebra_cols)

        while True:
            # Track the number of missing values before this pass
            nans_before_pass = self.df.isnull().sum().sum()

            # --- 1. Attempt to repair Trip_Price ---
            price_components = [
                ALGEBRA_COLUMNS["BASE_FARE"], ALGEBRA_COLUMNS["DISTANCE"], ALGEBRA_COLUMNS["KM_RATE"],
                ALGEBRA_COLUMNS["DURATION"], ALGEBRA_COLUMNS["MIN_RATE"]
            ]
            mask_repair_price = self.df[ALGEBRA_COLUMNS["PRICE"]].isnull() & self.df[price_components].notnull().all(axis=1)
            if mask_repair_price.any():
                self.df.loc[mask_repair_price, ALGEBRA_COLUMNS["PRICE"]] = (
                    self.df.loc[mask_repair_price, ALGEBRA_COLUMNS["BASE_FARE"]] +
                    (self.df.loc[mask_repair_price, ALGEBRA_COLUMNS["DISTANCE"]] * self.df.loc[mask_repair_price, ALGEBRA_COLUMNS["KM_RATE"]]) +
                    (self.df.loc[mask_repair_price, ALGEBRA_COLUMNS["DURATION"]] * self.df.loc[mask_repair_price, ALGEBRA_COLUMNS["MIN_RATE"]])
                )

            # --- 2. Attempt to repair Trip_Distance_km ---
            dist_components = [
                ALGEBRA_COLUMNS["PRICE"], ALGEBRA_COLUMNS["BASE_FARE"], ALGEBRA_COLUMNS["DURATION"],
                ALGEBRA_COLUMNS["MIN_RATE"], ALGEBRA_COLUMNS["KM_RATE"]
            ]
            mask_repair_dist = self.df[ALGEBRA_COLUMNS["DISTANCE"]].isnull() & self.df[dist_components].notnull().all(axis=1)
            mask_repair_dist &= (self.df[ALGEBRA_COLUMNS["KM_RATE"]] != 0) # Avoid division by zero
            if mask_repair_dist.any():
                self.df.loc[mask_repair_dist, ALGEBRA_COLUMNS["DISTANCE"]] = (
                    (self.df.loc[mask_repair_dist, ALGEBRA_COLUMNS["PRICE"]] -
                    self.df.loc[mask_repair_dist, ALGEBRA_COLUMNS["BASE_FARE"]] -
                    (self.df.loc[mask_repair_dist, ALGEBRA_COLUMNS["DURATION"]] * self.df.loc[mask_repair_dist, ALGEBRA_COLUMNS["MIN_RATE"]])) /
                    self.df.loc[mask_repair_dist, ALGEBRA_COLUMNS["KM_RATE"]]
                )

            # --- 3. Attempt to repair Base_Fare ---
            base_components = [
                ALGEBRA_COLUMNS["PRICE"], ALGEBRA_COLUMNS["DISTANCE"], ALGEBRA_COLUMNS["KM_RATE"],
                ALGEBRA_COLUMNS["DURATION"], ALGEBRA_COLUMNS["MIN_RATE"]
            ]
            mask_repair_base = self.df[ALGEBRA_COLUMNS["BASE_FARE"]].isnull() & self.df[base_components].notnull().all(axis=1)
            if mask_repair_base.any():
                self.df.loc[mask_repair_base, ALGEBRA_COLUMNS["BASE_FARE"]] = (
                    self.df.loc[mask_repair_base, ALGEBRA_COLUMNS["PRICE"]] -
                    (self.df.loc[mask_repair_base, ALGEBRA_COLUMNS["DISTANCE"]] * self.df.loc[mask_repair_base, ALGEBRA_COLUMNS["KM_RATE"]]) -
                    (self.df.loc[mask_repair_base, ALGEBRA_COLUMNS["DURATION"]] * self.df.loc[mask_repair_base, ALGEBRA_COLUMNS["MIN_RATE"]])
                )

            # --- 4. Attempt to repair Per_Km_Rate ---
            km_rate_components = [
                ALGEBRA_COLUMNS["PRICE"], ALGEBRA_COLUMNS["BASE_FARE"], ALGEBRA_COLUMNS["DURATION"],
                ALGEBRA_COLUMNS["MIN_RATE"], ALGEBRA_COLUMNS["DISTANCE"]
            ]
            mask_repair_km_rate = self.df[ALGEBRA_COLUMNS["KM_RATE"]].isnull() & self.df[km_rate_components].notnull().all(axis=1)
            mask_repair_km_rate &= (self.df[ALGEBRA_COLUMNS["DISTANCE"]] != 0) # Avoid division by zero
            if mask_repair_km_rate.any():
                self.df.loc[mask_repair_km_rate, ALGEBRA_COLUMNS["KM_RATE"]] = (
                    (self.df.loc[mask_repair_km_rate, ALGEBRA_COLUMNS["PRICE"]] -
                    self.df.loc[mask_repair_km_rate, ALGEBRA_COLUMNS["BASE_FARE"]] -
                    (self.df.loc[mask_repair_km_rate, ALGEBRA_COLUMNS["DURATION"]] * self.df.loc[mask_repair_km_rate, ALGEBRA_COLUMNS["MIN_RATE"]])) /
                    self.df.loc[mask_repair_km_rate, ALGEBRA_COLUMNS["DISTANCE"]]
                )

            # --- 5. Attempt to repair Trip_Duration_Minutes ---
            duration_components = [
                ALGEBRA_COLUMNS["PRICE"], ALGEBRA_COLUMNS["BASE_FARE"], ALGEBRA_COLUMNS["DISTANCE"],
                ALGEBRA_COLUMNS["KM_RATE"], ALGEBRA_COLUMNS["MIN_RATE"]
            ]
            mask_repair_duration = self.df[ALGEBRA_COLUMNS["DURATION"]].isnull() & self.df[duration_components].notnull().all(axis=1)
            mask_repair_duration &= (self.df[ALGEBRA_COLUMNS["MIN_RATE"]] != 0) # Avoid division by zero
            if mask_repair_duration.any():
                self.df.loc[mask_repair_duration, ALGEBRA_COLUMNS["DURATION"]] = (
                    (self.df.loc[mask_repair_duration, ALGEBRA_COLUMNS["PRICE"]] -
                    self.df.loc[mask_repair_duration, ALGEBRA_COLUMNS["BASE_FARE"]] -
                    (self.df.loc[mask_repair_duration, ALGEBRA_COLUMNS["DISTANCE"]] * self.df.loc[mask_repair_duration, ALGEBRA_COLUMNS["KM_RATE"]])) /
                    self.df.loc[mask_repair_duration, ALGEBRA_COLUMNS["MIN_RATE"]]
                )

            # --- 6. Attempt to repair Per_Minute_Rate ---
            min_rate_components = [
                ALGEBRA_COLUMNS["PRICE"], ALGEBRA_COLUMNS["BASE_FARE"], ALGEBRA_COLUMNS["DISTANCE"],
                ALGEBRA_COLUMNS["KM_RATE"], ALGEBRA_COLUMNS["DURATION"]
            ]
            mask_repair_min_rate = self.df[ALGEBRA_COLUMNS["MIN_RATE"]].isnull() & self.df[min_rate_components].notnull().all(axis=1)
            mask_repair_min_rate &= (self.df[ALGEBRA_COLUMNS["DURATION"]] != 0) # Avoid division by zero
            if mask_repair_min_rate.any():
                self.df.loc[mask_repair_min_rate, ALGEBRA_COLUMNS["MIN_RATE"]] = (
                    (self.df.loc[mask_repair_min_rate, ALGEBRA_COLUMNS["PRICE"]] -
                    self.df.loc[mask_repair_min_rate, ALGEBRA_COLUMNS["BASE_FARE"]] -
                    (self.df.loc[mask_repair_min_rate, ALGEBRA_COLUMNS["DISTANCE"]] * self.df.loc[mask_repair_min_rate, ALGEBRA_COLUMNS["KM_RATE"]])) /
                    self.df.loc[mask_repair_min_rate, ALGEBRA_COLUMNS["DURATION"]]
                )

            # If no NaNs were filled in a full pass, break the loop
            if self.df.isnull().sum().sum() == nans_before_pass:
                break
                
        # Log the final state after all possible repairs
        self._log_repair_status('After Algebraic Repair', all_algebra_cols)

    def repair_using_imputation(self, max_cat_values: int = 5):
        """
        This method takes a dataframe and attempts to fill null values inside all columns
        by iteratively training models on the non-null data.
        """
        cat_columns = find_categorical_columns(self.df, max_cat_values)

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
        cat_cols = find_categorical_columns(features_df, max_cat_values=max_cat_values)
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

        cat_cols = find_categorical_columns(X, max_cat_values=max_cat_values)
        X_encoded = pd.get_dummies(X, columns=cat_cols)

        clean_indices = X_encoded.dropna().index
        X_train_clean = X_encoded.loc[clean_indices]
        y_train_clean = y_encoded.loc[clean_indices]

        if X_train_clean.empty:
            return pd.Series(dtype=object)

        best_model = self._find_best_classification_model(X_train_clean, y_train_clean)

        X_predict = predict_df_orig.drop(target_column, axis=1)

        X_predict_encoded = pd.get_dummies(X_predict, columns=find_categorical_columns(X_predict, max_cat_values=max_cat_values))
        X_predict_encoded = X_predict_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        predict_indices = X_predict_encoded.dropna().index
        X_predict_clean = X_predict_encoded.loc[predict_indices]

        if X_predict_clean.empty:
            return pd.Series(dtype=object)

        numeric_predictions = best_model.predict(X_predict_clean)
        text_predictions = pd.Series(numeric_predictions, index=predict_indices).map(inverse_map)
        return text_predictions


    @staticmethod
    def _find_best_regression_model(X, y):
        """Function tries both linear regression/random forest to predict column values and returns
        the model with lowest rsme"""

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42, train_size=0.8)
        lr_model = LinearRegression().fit(Xtrain, ytrain)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtrain, ytrain)

        lr_error_dict = TaxiData._calculate_errors(lr_model, Xtest, ytest)
        rf_error_dict = TaxiData._calculate_errors(rf_model, Xtest, ytest)
        print(f"{lr_error_dict=}")
        print(f"{rf_error_dict=}")

        return lr_model if lr_error_dict["rmse"] < rf_error_dict["rmse"] else rf_model

    @staticmethod
    def _calculate_errors(model, Xtest, ytest):
        y_pred = model.predict(Xtest)
        error_dict = {}
        error_dict["r2"] = r2_score(y_pred=y_pred, y_true=ytest)
        error_dict["mae"] = mean_absolute_error(y_pred=y_pred, y_true=ytest)
        error_dict["mse"] = mean_squared_error(y_pred=y_pred, y_true=ytest)
        error_dict["rmse"] = root_mean_squared_error(y_pred=y_pred, y_true=ytest)
        return error_dict

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
    
@staticmethod
def find_categorical_columns(df:pd.DataFrame, max_cat_values:int = 5) -> list[str]:
    """function figures out which columns should be treated as categorical"""
    category_columns = []
    for column_name in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column_name]) or df[column_name].nunique() < max_cat_values:
            category_columns.append(column_name)
    return category_columns
    
if __name__ == "__main__":
    sample_trip = Trip(
    Trip_Distance_km=5.0,
    Time_of_Day='Morning',
    Day_of_Week='Mon',
    Passenger_Count=1,
    Traffic_Conditions='Light',
    Weather='Sunny'
)
    data = TaxiData(ALTERED_CSV_PATH)
    data.train_model()
    
    y_pred = data.predict_trip_price(sample_trip)
    print(y_pred)

    