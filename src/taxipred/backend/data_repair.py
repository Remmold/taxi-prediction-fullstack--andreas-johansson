import pandas as pd

COLUMNS = {
    "DISTANCE": "Trip_Distance_km",
    "BASE_FARE": "Base_Fare",
    "KM_RATE": "Per_Km_Rate",
    "MIN_RATE": "Per_Minute_Rate", 
    "DURATION": "Trip_Duration_Minutes",
    "PRICE": "Trip_Price"
}

def _log_repair_status(df: pd.DataFrame, stage: str, repaired_cols: list):
    """Internal helper function to log the status of missing values."""
    print(f"--- {stage} ---")
    print("Missing values in key columns:")
    print(df[repaired_cols].isnull().sum())
    print("-" * 25)

def repair_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repairs price and distance columns using a known fare calculation formula
    and logs the changes.

    This function modifies the DataFrame in place.
    """
    # Log the initial state before any changes are made
    _log_repair_status(df, 'Before Repair', [COLUMNS["PRICE"], COLUMNS["DISTANCE"]])

    # --- Repair Missing Trip Price ---
    price_components = [
        COLUMNS["BASE_FARE"], COLUMNS["DISTANCE"], COLUMNS["KM_RATE"],
        COLUMNS["DURATION"], COLUMNS["MIN_RATE"]
    ]
    mask_repair_price = df[COLUMNS["PRICE"]].isnull() & df[price_components].notnull().all(axis=1)

    if mask_repair_price.any():
        df.loc[mask_repair_price, COLUMNS["PRICE"]] = (
            df.loc[mask_repair_price, COLUMNS["BASE_FARE"]] +
            (df.loc[mask_repair_price, COLUMNS["DISTANCE"]] * df.loc[mask_repair_price, COLUMNS["KM_RATE"]]) +
            (df.loc[mask_repair_price, COLUMNS["DURATION"]] * df.loc[mask_repair_price, COLUMNS["MIN_RATE"]])
        )

    # --- Repair Missing Trip Distance ---
    distance_components = [
        COLUMNS["BASE_FARE"], COLUMNS["KM_RATE"], COLUMNS["MIN_RATE"],
        COLUMNS["DURATION"], COLUMNS["PRICE"]
    ]
    mask_repair_distance = df[COLUMNS["DISTANCE"]].isnull() & df[distance_components].notnull().all(axis=1)
    mask_repair_distance &= (df[COLUMNS["KM_RATE"]] != 0)

    if mask_repair_distance.any():
        df.loc[mask_repair_distance, COLUMNS["DISTANCE"]] = (
            (df.loc[mask_repair_distance, COLUMNS["PRICE"]] -
             df.loc[mask_repair_distance, COLUMNS["BASE_FARE"]] -
             (df.loc[mask_repair_distance, COLUMNS["DURATION"]] * df.loc[mask_repair_distance, COLUMNS["MIN_RATE"]])) /
            df.loc[mask_repair_distance, COLUMNS["KM_RATE"]]
        )
    
    # Log the final state after all repairs are attempted
    _log_repair_status(df, 'After Repair', [COLUMNS["PRICE"], COLUMNS["DISTANCE"]])
    
    return df



##### machinelearning repair #####
"""
This script provides a machine learning-based approach to impute missing values
in a pandas DataFrame. It intelligently handles both numerical and categorical
columns by training a model on the non-missing data to predict the nulls.

The main function to use is `impute_dataframe`.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Optional

# --- Internal Helper Functions (prefixed with _) ---

def _classify_columns(dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Analyzes the DataFrame's dtypes to separate columns into numerical and
    categorical lists.
    """
    numerical_cols = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_cols, categorical_cols

def _prepare_features(dataframe: pd.DataFrame, target_col_name: str) -> pd.DataFrame:
    """
    Prepares the DataFrame for modeling by one-hot encoding categorical features.
    """
    df_encoded = dataframe.copy()
    _, categorical_cols = _classify_columns(df_encoded)

    for col in categorical_cols:
        if col != target_col_name:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, dummy_na=False)

    return df_encoded

def _fill_missing_values_in_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Trains a model to predict and fill missing values in a single target column.
    """
    print(f"--- Processing column: {target_col} ---")
    df_copy = df.copy()

    numerical_cols, _ = _classify_columns(df_copy)
    is_regression = target_col in numerical_cols
    model_type = "Regressor" if is_regression else "Classifier"
    print(f"Detected as {model_type.lower()} task. Using RandomForest{model_type}.")

    df_encoded = _prepare_features(df_copy, target_col)
    train_df = df_encoded.dropna(subset=[target_col])
    predict_df = df_encoded[df_encoded[target_col].isnull()]

    if predict_df.empty:
        print(f"No missing values to impute in '{target_col}'. Skipping.\n")
        return df

    features = [col for col in train_df.columns if col != target_col]
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_predict = predict_df[features]

    X_train, X_predict = X_train.align(X_predict, join='left', axis=1, fill_value=0)

    if is_regression:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_predict)
    else:
        target_encoder = LabelEncoder()
        y_train_encoded = target_encoder.fit_transform(y_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train_encoded)
        predicted_labels = model.predict(X_predict)
        predictions = target_encoder.inverse_transform(predicted_labels)

    df_copy.loc[df_copy[target_col].isnull(), target_col] = predictions
    print(f"Successfully imputed {len(predictions)} values in '{target_col}'.\n")

    return df_copy

def _run_iterative_imputation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Internal function to find and iteratively fill all columns with missing values.
    """
    df_imputed = dataframe.copy()
    cols_with_nulls = df_imputed.columns[df_imputed.isnull().any()].tolist()

    if not cols_with_nulls:
        print("No missing values found in the DataFrame. No action taken.")
        return df_imputed

    print(f"Found missing values in columns: {cols_with_nulls}")
    for col_name in cols_with_nulls:
        df_imputed = _fill_missing_values_in_column(df_imputed, col_name)

    print("--- Iterative imputation complete! ---")
    return df_imputed

# --- Main Public Function ---

def impute_dataframe(dataframe: pd.DataFrame, force_categorical: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Imputes all missing values in a DataFrame using machine learning. This is the
    main function to import and use.

    Args:
        dataframe (pd.DataFrame): The DataFrame with missing values.
        force_categorical (Optional[List[str]], optional): A list of column names
            that should be treated as categorical, even if they have a numeric
            dtype (e.g., 'Passenger_Count'). Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with all missing values imputed.
    """
    df_processed = dataframe.copy()

    # Step 1: Adjust dtypes for columns that should be treated as categorical.
    # This is crucial for columns like 'Passenger_Count' which are numeric but
    # should not have fractional values imputed.
    if force_categorical:
        print(f"\nForcing columns to 'category' dtype: {force_categorical}\n")
        for col in force_categorical:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype('category')
            else:
                # A simple warning if the user provides a column that doesn't exist.
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping dtype conversion.")

    # Step 2: Run the iterative imputation process using the internal helper.
    df_final = _run_iterative_imputation(df_processed)

    return df_final


if __name__ == '__main__':
    # --- How to use the `impute_dataframe` function ---

    # 1. Create a sample DataFrame
    data = {
        'Trip_Distance_km': [10.5, 12.1, np.nan, 5.3, 8.9, 15.2, 2.1, np.nan, 7.5, 9.0],
        'Time_of_Day': ['Morning', 'Afternoon', 'Morning', 'Evening', np.nan, 'Afternoon', 'Morning', 'Evening', 'Morning', 'Afternoon'],
        'Day_of_Week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', np.nan, 'Mon', 'Tue', 'Wed'],
        'Passenger_Count': [1, 2, 1, np.nan, 3, 1, 2, 1, 1, np.nan],
        'Traffic_Conditions': ['Heavy', 'Light', 'Medium', 'Heavy', 'Light', np.nan, 'Medium', 'Heavy', 'Light', 'Medium'],
        'Weather': ['Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy', 'Rainy', 'Sunny', np.nan, 'Cloudy', 'Rainy'],
        'Trip_Price': [25.5, 30.0, np.nan, 15.0, 22.0, 35.5, 10.0, 28.0, 18.5, np.nan]
    }
    df_original = pd.DataFrame(data)
    print("Original DataFrame info:")
    df_original.info()
    print("\nDataFrame before imputation:\n", df_original)

    # 2. Call the single imputation function
    # Provide the list of columns that should be handled as categorical, even if they are numbers.
    df_final = impute_dataframe(
        dataframe=df_original,
        force_categorical=['Passenger_Count']
    )

    # 3. Review the results
    print("\nImputed DataFrame info:")
    df_final.info()
    print("\nDataFrame after imputation:\n", df_final)

