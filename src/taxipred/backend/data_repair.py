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
def fill_one_numeric_column(df: pd.DataFrame,target_column:str,max_cat_values:int = 5):
    """this function fills in nullvalues of one numeric column expecting maximum 1 null value per row"""
    # creates a df with only the features and creates encoded df
    features_df = df.drop(target_column,axis=1)
    encoded_df = pd.get_dummies(data=features_df,columns=find_categorical_columns(features_df,max_cat_values=max_cat_values))
    # adds target column to encoded df
    encoded_df[target_column] = df[target_column]
    # splitting df on nulls
    train_df = encoded_df[encoded_df[target_column].isnull() == False]
    predict_df = encoded_df[encoded_df[target_column].isnull()]

    # Separates features from target
    X = train_df.drop(target_column,axis=1)
    y = train_df[target_column]

    # asseses various regression models on rsme score and picks best one
    best_model = find_best_regression_model(X,y)

    # makes real assesment trying to fill in nulls
    X_predict = predict_df.drop(target_column,axis=1)
    y_pred = best_model.predict(X_predict)
    predict_df[target_column] = y_pred

    completed_df = pd.concat([train_df,predict_df])
    return completed_df
    

def find_categorical_columns(df:pd.DataFrame,max_cat_values:int) -> list[str]: 
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
    
