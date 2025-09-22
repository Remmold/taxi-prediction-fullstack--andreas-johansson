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
def fill_one_null_column(df: pd.DataFrame,target_column:str):

    train_df = df[df[target_column].isnull() == False]
    predict_df = df[df[target_column].isnull()]
    X = train_df.drop(target_column,axis=1)
    y = train_df[target_column]


def find_best_regression_model(X,y):
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest,ytrain,ytest = train_test_split(X,y,random_state=42,train_size=0.7)

    linear_regression_rsme = evaluate_linear_regression(Xtrain, Xtest,ytrain,ytest)
    random_forest_rsme = evaluate_random_forest(Xtrain, Xtest,ytrain,ytest)

    best_regression_model = "linear_regression" if linear_regression_rsme<random_forest_rsme else "random_forest"
    print(f"{linear_regression_rsme=}")
    print(f"{random_forest_rsme=}")

    return best_regression_model

def evaluate_linear_regression(Xtrain, Xtest,ytrain,ytest):

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error
    model = LinearRegression()
    model.fit(Xtrain,ytrain)
    
    y_pred = model.predict(Xtest)
    return root_mean_squared_error(y_pred=y_pred, y_true=ytest)

def evaluate_random_forest(Xtrain, Xtest,ytrain,ytest):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    
    return root_mean_squared_error(y_true=ytest, y_pred=y_pred)


if __name__ == "__main__":
    data = {
    'Trip_Distance_km': [10.5, 12.1, 7.0, 5.3, 8.9, 15.2, 2.1, np.nan, 7.5, 9.0],
    'Time_of_Day': ['Morning', 'Afternoon', 'Morning', 'Evening', np.nan, 'Afternoon', 'Morning', 'Evening', 'Morning', 'Afternoon'],
    'Day_of_Week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', np.nan, 'Mon', 'Tue', 'Wed'],
    'Passenger_Count': [1, 2, 1, np.nan, 3, 1, 2, 1, 1, np.nan],
    'Traffic_Conditions': ['Heavy', 'Light', 'Medium', 'Heavy', 'Light', np.nan, 'Medium', 'Heavy', 'Light', 'Medium'],
    'Weather': ['Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy', 'Rainy', 'Sunny', np.nan, 'Cloudy', 'Rainy'],
    'Trip_Price': [25.5, 30.0, 18.0, 15.0, 22.0, 35.5, 10.0, 28.0, 18.5, np.nan]
}
    
    df = pd.DataFrame(data)
    print(fill_one_null_column(df,"Weather"))
    
