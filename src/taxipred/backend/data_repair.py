import pandas as pd
import numpy as np

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


##### machinelearning to fill nulls #####
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
def fill_all_null(df: pd.DataFrame, max_cat_values: int = 5) -> pd.DataFrame:
    """
    This method takes a dataframe and attempts to fill null values inside all columns
    by iteratively training models on the non-null data.
    """
    cat_columns = find_categorical_columns(df, max_cat_values)
    
    # Loop as long as there are any null values in the DataFrame
    while df.isnull().sum().sum() > 0:
        total_nans_before = df.isnull().sum().sum()
        
        # Get a list of columns that currently have nulls
        cols_with_nulls = df.columns[df.isnull().any()].tolist()
        
        # Process one column at a time
        for col_name in cols_with_nulls:
            print(f"Attempting to fill column: {col_name}")
            
            if col_name in cat_columns:
                predictions = fill_one_categorical_column(df=df, target_column=col_name, max_cat_values=max_cat_values)
            else:
                predictions = fill_one_numeric_column(df=df, target_column=col_name, max_cat_values=max_cat_values)
            
            # If predictions were made, update the DataFrame
            if not predictions.empty:
                df.loc[predictions.index, col_name] = predictions
                print(f"Filled {len(predictions)} values in {col_name}.")

        # Safety break: if no NaNs were filled in a full pass, exit the loop
        if df.isnull().sum().sum() == total_nans_before:
            print("Could not fill any more NaNs. Exiting.")
            break
            
    return df




def fill_one_numeric_column(df: pd.DataFrame, target_column: str, max_cat_values: int = 5) -> pd.Series:
    """
    This function trains a model and returns a Series containing the
    predicted values for the nulls in the target column.
    """
    # Create a copy to avoid modifying the original df inside the function
    df_copy = df.copy()

    # Create a df with only the features and create encoded df
    features_df = df_copy.drop(target_column, axis=1)
    # find categorical columns to be one-hot encoded
    cat_cols = find_categorical_columns(features_df, max_cat_values=max_cat_values)
    encoded_df = pd.get_dummies(data=features_df, columns=cat_cols)

    # Add target column to encoded df
    encoded_df[target_column] = df_copy[target_column]

    # Splitting df on nulls
    train_df = encoded_df[encoded_df[target_column].notnull()]
    predict_df = encoded_df[encoded_df[target_column].isnull()]

    # If there's nothing to predict, return an empty Series
    if predict_df.empty:
        return pd.Series(dtype=float)

    # Separates features from target
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    
    # The features for the rows we want to predict on
    X_predict = predict_df.drop(target_column, axis=1)

    # Drop any rows with remaining NaNs in the training set
    clean_indices = X_train.dropna().index
    X_train_clean = X_train.loc[clean_indices]
    y_train_clean = y_train.loc[clean_indices]
    
    # If no clean training data is available, we can't predict
    if X_train_clean.empty:
        return pd.Series(dtype=float)

    # Find the best model
    best_model = find_best_regression_model(X_train_clean, y_train_clean)
    
    # Make sure prediction columns match training columns
    X_predict = X_predict.reindex(columns=X_train_clean.columns, fill_value=0)
    
    # Drop rows with NaNs in the prediction set as well
    predict_indices = X_predict.dropna().index
    X_predict_clean = X_predict.loc[predict_indices]
    
    if X_predict_clean.empty:
        return pd.Series(dtype=float)

    # Make predictions
    predictions = best_model.predict(X_predict_clean)

    # Return predictions as a Series with the correct index
    return pd.Series(predictions, index=predict_indices)
    

def find_categorical_columns(df:pd.DataFrame,max_cat_values:int = 5) -> list[str]: 
    """function figures out which columns should be treated as categorical"""
    category_columns = []
    for column_name in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column_name]) or df[column_name].nunique() < max_cat_values:
            category_columns.append(column_name)
    return category_columns
            
    

def find_best_regression_model(X,y):
    """Function tries both linear regression/random forest to predict column values and returns
       the model with lowest rsme"""
    from sklearn.model_selection import train_test_split

    Xtrain, Xtest,ytrain,ytest = train_test_split(X,y,random_state=42,train_size=0.7)

    # instantiates regression models
    lr_model = LinearRegression().fit(Xtrain,ytrain)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtrain,ytrain)

    # calculates rsme on models
    linear_regression_rsme = calculate_rsme(lr_model,Xtest,ytest)
    random_forest_rsme = calculate_rsme(rf_model,Xtest,ytest)

    # displays rsme in terminal
    print(f"{linear_regression_rsme=}")
    print(f"{random_forest_rsme=}")
    # picks and returns model with lowest rsme

    return lr_model if linear_regression_rsme<random_forest_rsme else rf_model

    
def calculate_rsme(model,Xtest,ytest):
    from sklearn.metrics import root_mean_squared_error
    y_pred = model.predict(Xtest)
    return root_mean_squared_error(y_pred=y_pred,y_true=ytest)

def fill_one_categorical_column(df: pd.DataFrame, target_column: str, max_cat_values: int = 5) -> pd.Series:
    """
    Fills null values of one categorical column using a classification model
    and returns the predictions as a Series.
    """
    df_copy = df.copy()

    # Split data based on nulls in the target column
    train_df_orig = df_copy[df_copy[target_column].notnull()]
    predict_df_orig = df_copy[df_copy[target_column].isnull()]

    if predict_df_orig.empty:
        return pd.Series(dtype=object)

    # Separate features from target
    X = train_df_orig.drop(target_column, axis=1)
    y = train_df_orig[target_column]

    # Create mapping dictionaries and encode y into numbers
    forward_map = map_category(y)
    inverse_map = inverse_dict(forward_map)
    y_encoded = y.map(forward_map)

    # Encode the feature set X
    cat_cols = find_categorical_columns(X, max_cat_values=max_cat_values)
    X_encoded = pd.get_dummies(X, columns=cat_cols)

    # Clean NaNs from the training data
    clean_indices = X_encoded.dropna().index
    X_train_clean = X_encoded.loc[clean_indices]
    y_train_clean = y_encoded.loc[clean_indices]
    
    if X_train_clean.empty:
        return pd.Series(dtype=object)

    best_model = find_best_classification_model(X_train_clean, y_train_clean)

    # Prepare the prediction features (X_predict)
    X_predict = predict_df_orig.drop(target_column, axis=1)
    
    # Ensure X_predict has the same columns as X_encoded
    X_predict_encoded = pd.get_dummies(X_predict, columns=find_categorical_columns(X_predict, max_cat_values=max_cat_values))
    X_predict_encoded = X_predict_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Clean NaNs from the prediction features
    predict_indices = X_predict_encoded.dropna().index
    X_predict_clean = X_predict_encoded.loc[predict_indices]

    if X_predict_clean.empty:
        return pd.Series(dtype=object)
        
    # Model predicts numbers (e.g., 0, 1, 2)
    numeric_predictions = best_model.predict(X_predict_clean)
    
    # Convert numbers back to text labels using the inverse_map
    text_predictions = pd.Series(numeric_predictions, index=predict_indices).map(inverse_map)
    
    return text_predictions

def find_best_classification_model(X,y):
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()

    rf_model.fit(X,y)
    return rf_model

def map_category(series:pd.Series)-> dict:
    mapping_dict = {category: i for i, category in enumerate(series.unique())}
    return mapping_dict
def inverse_dict(map:dict):
    inverse_map = {value: key for key, value in map.items()}
    return inverse_map


if __name__ == "__main__":
    data = {
    'Trip_Distance_km': [10.5, 12.1, 7.0, 5.3, 8.9, 15.2, 2.1, 3, 7.5, 9.0],
    'Time_of_Day': ['Morning', 'Afternoon', 'Morning', 'Evening', np.nan, 'Afternoon', 'Morning', 'Evening', 'Morning', 'Afternoon'],
    'Day_of_Week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', np.nan, 'Mon', 'Tue', 'Wed'],
    'Passenger_Count': [1, 2, 1, np.nan, 3, 1, 2, 1, 1, 2],
    'Traffic_Conditions': ['Heavy', 'Light', 'Medium', 'Heavy', 'Light', 'Light', 'Medium', 'Heavy', 'Light', 'Medium'],
    'Weather': ['Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy', 'Rainy', 'Sunny', np.nan, 'Cloudy', 'Rainy'],
    'Trip_Price': [25.5, 30.0, 18.0, 15.0, 22.0, 35.5, 10.0, 28.0, 18.5, 5]
}
    
    df = pd.DataFrame(data)
    print(fill_one_numeric_column(df,"Trip_Distance_km"))
    
