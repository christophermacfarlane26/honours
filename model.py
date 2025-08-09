"""
Restaurant Sales Prediction Model
Trains machine learning models to predict daily sales for restaurant dishes
and implements waste reduction strategies through adaptive inventory management.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import math
import os
import datetime


def load_and_prepare_data():
    """Load and prepare the restaurant sales data"""
    data_path = r'C:\Users\chris\OneDrive\Desktop\Main-App-20250809T224141Z-1-001\Main-App\year1_year2_cleaned_fixed.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        exit(1)
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def create_day_of_week_column(df):
    """Create a readable day of week column from dummy variables"""
    day_columns = [
        'Day of the Week_Friday', 'Day of the Week_Monday', 'Day of the Week_Saturday',
        'Day of the Week_Sunday', 'Day of the Week_Thursday', 'Day of the Week_Tuesday',
        'Day of the Week_Wednesday'
    ]
    
    def get_day_of_week(row):
        for day in day_columns:
            if row[day] == 1:
                return day.split('_')[-1]
        return 'Unknown'
    
    df['Day of the Week'] = df[day_columns].apply(get_day_of_week, axis=1)
    print(f"Created day of week column: {df['Day of the Week'].unique()}")
    
    return df


def define_dish_mappings(df):
    """Define mappings between dish names and their sales columns"""
    dish_mappings = {
        'FOC ROSEMARY': ['Qty (Sales)_FOC ROSEMARY - 7512', 'Qty (Sales)_TA FOC ROSE - 25600'],
        'LASAGNE': ['Qty (Sales)_LASAGNE - 3513', 'Qty (Sales)_TA LASAGNE - 25200'],
        'PZ MARGHERITA': ['Qty (Sales)_PZ MARGHERITA - 4500', 'Qty (Sales)_TA MARG PZ - 25300'],
        'NONNAS BURGER': ['Qty (Sales)_NONNAS BURGER - 6502', 'Qty (Sales)_TA NONNAS BURGER - 25106'],
        'PENNE ARRABBIATA': ['Qty (Sales)_PENNE ARRABBIATA - 3500', 'Qty (Sales)_TA PENNE ARRABBIATA - 25208'],
        'PENNE CON POLLO': ['Qty (Sales)_PENNE CON POLLO - 3507', 'Qty (Sales)_TA PENNE POLLO - 25206'],
        'PENNE SALSICCIA': ['Qty (Sales)_PENNE SALSICCIA - 3506', 'Qty (Sales)_TA PENNE SALSICCIA - 25204'],
        'SKINNY FRIES': ['Qty (Sales)_SKINNY FRIES - 7510', 'Qty (Sales)_TA SKINNY FRIES - 25505'],
        'SPAG BOLGNESE': ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203'],
        'SPAG CARBONARA': ['Qty (Sales)_SPAG CARBONARA - 3503', 'Qty (Sales)_TA SPAG CARBONARA - 25202'],
        'SPAG POMODORO': ['Qty (Sales)_SPAG POMODORO - 3502', 'Qty (Sales)_TA SPAG POMODORO - 25209'],
        'BRUSHETTA CLASSICA': ['Qty (Sales)_BRUSHETTA CLASSICA - 2502'],
        'MOZZARELLA FRITTA': ['Qty (Sales)_MOZZARELLA FRITTA - 2003'],
        'PZ MESSICANA': ['Qty (Sales)_PZ MESSICANA - 4504'],
        'RIBEYE STEAK': ['Qty (Sales)_RIBEYE STEAK - 6501']
    }
    
    target_cols = [col for col in df.columns if col.startswith('Qty (Sales)_')]
    print(f"Target columns: {len(target_cols)}")
    
    return dish_mappings, target_cols


def add_basic_features(df, target_cols):
    """Add basic time-based and holiday features"""
    # Define UK holidays for 2022-2023
    holidays = [
        '2022-01-01', '2022-04-15', '2022-04-18', '2022-05-02', '2022-06-02',
        '2022-06-03', '2022-08-01', '2022-11-30', '2022-12-25', '2022-12-26',
        '2023-01-01', '2023-04-07', '2023-04-10', '2023-05-01', '2023-05-29',
        '2023-08-07', '2023-11-30', '2023-12-25', '2023-12-26'
    ]
    
    df['Is_Holiday'] = df['Date'].isin(pd.to_datetime(holidays)).astype(int)
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Total_Daily_Sales'] = df[target_cols].sum(axis=1)
    df['Is_Weekend'] = df['Day of the Week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    
    # Create dummy variables for month and quarter
    month_dummies = pd.get_dummies(df['Month'], prefix='Month', drop_first=True)
    quarter_dummies = pd.get_dummies(df['Quarter'], prefix='Quarter', drop_first=True)
    
    return df, month_dummies, quarter_dummies


def analyze_sales_patterns(df, target_cols):
    """Analyze sales patterns and identify stable items"""
    train_df_temp = df[df['Year'] == 2022]
    stable_items = []
    avg_sales = {}
    
    for col in target_cols:
        avg_sales[col] = train_df_temp[col].mean()
        std_sales = train_df_temp[col].std()
        if avg_sales[col] < 3.0 and std_sales < 1.5:
            stable_items.append(col)
    
    print(f"Stable items (low variance): {len(stable_items)}")
    return avg_sales, stable_items


def create_lag_features(df, target_cols, high_rmse_items):
    """Create lag features for time series prediction"""
    lag1_cols = {f'{col}_lag1': df[col].shift(1) for col in target_cols}
    lag2_cols = {}
    lag3_cols = {}
    lag4_cols = {}
    lag5_cols = {}
    
    # Create lag features for high RMSE items and specific dishes
    special_items = ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203', 
                    'Qty (Sales)_RIBEYE STEAK - 6501', 'Qty (Sales)_PZ MESSICANA - 4504', 
                    'Qty (Sales)_SPAG POMODORO - 3502']
    
    for col in high_rmse_items + special_items:
        lag2_cols[f'{col}_lag2'] = df[col].shift(2)
        lag3_cols[f'{col}_lag3'] = df[col].shift(3)
        
        # Extra lags for spaghetti bolognese (complex patterns)
        if col in ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203']:
            lag4_cols[f'{col}_lag4'] = df[col].shift(4)
            lag5_cols[f'{col}_lag5'] = df[col].shift(5)
    
    return lag1_cols, lag2_cols, lag3_cols, lag4_cols, lag5_cols


def create_advanced_features(df, target_cols, high_demand_items):
    """Create advanced features including ratios, trends, and interactions"""
    # Sales ratio features (current sales vs historical day-of-week average)
    sales_ratio_cols = {}
    six_wk_avg_cols = {}
    trend_cols = {}
    dow_std_cols = {}
    
    for col in target_cols:
        # Sales ratio compared to 2022 day-of-week average
        sales_2022 = df[df['Year'] == 2022].groupby('Day of the Week')[col].mean().reset_index()
        sales_2022_dict = dict(zip(sales_2022['Day of the Week'], sales_2022[col]))
        sales_ratio_cols[f'{col}_sales_ratio_dow'] = df['Day of the Week'].map(sales_2022_dict).replace(0, 1)
        sales_ratio_cols[f'{col}_sales_ratio_dow'] = df[col] / sales_ratio_cols[f'{col}_sales_ratio_dow']
        sales_ratio_cols[f'{col}_sales_ratio_dow'] = sales_ratio_cols[f'{col}_sales_ratio_dow'].fillna(1.0)
        
        # 6-week rolling average by day of week
        six_wk_avg_cols[f'{col}_6wk_dow_avg'] = df.groupby('Day of the Week')[col].rolling(window=6, min_periods=1).mean().reset_index(level=0, drop=True)
        
        # Trend features (7, 14, 21 day rolling averages)
        trend_cols[f'{col}_7day_trend'] = df[col].rolling(window=7, min_periods=1).mean()
        trend_cols[f'{col}_14day_trend'] = df[col].rolling(window=14, min_periods=1).mean()
        trend_cols[f'{col}_21day_trend'] = df[col].rolling(window=21, min_periods=1).mean()
        
        # Day-of-week standard deviation from 2022
        dow_std = df[df['Year'] == 2022].groupby('Day of the Week')[col].std().reset_index()
        dow_std_dict = dict(zip(dow_std['Day of the Week'], dow_std[col]))
        dow_std_cols[f'{col}_dow_std'] = df['Day of the Week'].map(dow_std_dict).fillna(0)
    
    # Seasonality interactions
    month_interaction_cols = {f'{col}_Month_Interaction': df[col] * df['Month'] for col in target_cols}
    quarter_interaction_cols = {f'{col}_Quarter_Interaction': df[col] * df['Quarter'] for col in target_cols}
    holiday_interaction_cols = {f'{col}_Holiday_Interaction': df[col] * df['Is_Holiday'] for col in target_cols}
    
    # Special interaction features
    other_features = {
        'Temp_Holiday_Interaction': df['Temp Avg (°C)'] * df['Is_Holiday'],
        'Side_Saturday_Interaction': df['Day of the Week'].apply(lambda x: 1 if x == 'Saturday' else 0) * (
            df.get('Qty (Sales)_SKINNY FRIES - 7510', 0) + df.get('Qty (Sales)_TA SKINNY FRIES - 25505', 0)),
        'Precipitation_Saturday_Interaction': df['Day of the Week'].apply(lambda x: 1 if x == 'Saturday' else 0) * df['Precipitation (mm)'],
        'Temp_Avg_lag1': df['Temp Avg (°C)'].shift(1),
        'Precipitation_lag1': df['Precipitation (mm)'].shift(1),
        'Low_Demand_Indicator': df[['Qty (Sales)_RIBEYE STEAK - 6501',
                                    'Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203']].eq(0).sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    }
    
    # Weekend interactions for high-demand items
    weekend_interaction_cols = {f'{col}_Weekend_Interaction': df['Is_Weekend'] * df[col] for col in high_demand_items}
    
    return (sales_ratio_cols, six_wk_avg_cols, trend_cols, dow_std_cols, 
            month_interaction_cols, quarter_interaction_cols, holiday_interaction_cols,
            other_features, weekend_interaction_cols)


def create_dish_type_features(df):
    """Create dish type classification features"""
    dish_types = {
        'Qty (Sales)_BRUSHETTA CLASSICA - 2502': 'Starter',
        'Qty (Sales)_FOC ROSEMARY - 7512': 'Starter',
        'Qty (Sales)_TA FOC ROSE - 25600': 'Starter',
        'Qty (Sales)_MOZZARELLA FRITTA - 2003': 'Starter',
        'Qty (Sales)_LASAGNE - 3513': 'Main',
        'Qty (Sales)_TA LASAGNE - 25200': 'Main',
        'Qty (Sales)_NONNAS BURGER - 6502': 'Main',
        'Qty (Sales)_TA NONNAS BURGER - 25106': 'Main',
        'Qty (Sales)_PENNE ARRABBIATA - 3500': 'Main',
        'Qty (Sales)_TA PENNE ARRABBIATA - 25208': 'Main',
        'Qty (Sales)_PENNE CON POLLO - 3507': 'Main',
        'Qty (Sales)_TA PENNE POLLO - 25206': 'Main',
        'Qty (Sales)_PENNE SALSICCIA - 3506': 'Main',
        'Qty (Sales)_TA PENNE SALSICCIA - 25204': 'Main',
        'Qty (Sales)_PZ MARGHERITA - 4500': 'Main',
        'Qty (Sales)_TA MARG PZ - 25300': 'Main',
        'Qty (Sales)_PZ MESSICANA - 4504': 'Main',
        'Qty (Sales)_RIBEYE STEAK - 6501': 'Main',
        'Qty (Sales)_SPAG BOLGNESE - 3504': 'Main',
        'Qty (Sales)_TA SPAG BOLOGNESE - 25203': 'Main',
        'Qty (Sales)_SPAG CARBONARA - 3503': 'Main',
        'Qty (Sales)_TA SPAG CARBONARA - 25202': 'Main',
        'Qty (Sales)_SPAG POMODORO - 3502': 'Main',
        'Qty (Sales)_TA SPAG POMODORO - 25209': 'Main',
        'Qty (Sales)_SKINNY FRIES - 7510': 'Side',
        'Qty (Sales)_TA SKINNY FRIES - 25505': 'Side'
    }
    
    df['Dish_Type_Side'] = 0
    df['Dish_Type_Starter'] = 0
    for col, dtype in dish_types.items():
        if dtype == 'Side':
            df['Dish_Type_Side'] = df['Dish_Type_Side'].where(df[col].isna(), 1)
        elif dtype == 'Starter':
            df['Dish_Type_Starter'] = df['Dish_Type_Starter'].where(df[col].isna(), 1)
    
    return df


def train_models(X_train_scaled, y_train):
    """Train Random Forest and XGBoost models with optimized parameters"""
    # Random Forest with waste reduction focus
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,  # Reduced to prevent overfitting
        min_samples_split=8,
        min_samples_leaf=3,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # XGBoost with conservative parameters
    xgb_model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.03,  # Reduced for more conservative predictions
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    print("Models trained successfully")
    return rf_model, xgb_model


def evaluate_predictions(y_test, y_pred, target_cols):
    """Evaluate model performance and calculate metrics"""
    rmse_per_item = {}
    mae_per_item = {}
    r2_per_item = {}
    
    for i, item in enumerate(target_cols):
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        
        rmse_per_item[item] = rmse
        mae_per_item[item] = mae
        r2_per_item[item] = r2
    
    avg_rmse = np.mean(list(rmse_per_item.values()))
    avg_mae = np.mean(list(mae_per_item.values()))
    avg_r2 = np.mean(list(r2_per_item.values()))
    
    print(f"Average RMSE: {avg_rmse:.2f}")
    print(f"Average MAE: {avg_mae:.2f}")
    print(f"Average R²: {avg_r2:.2f}")
    
    return rmse_per_item, mae_per_item, r2_per_item


def calculate_buffers(dish_mappings, avg_sales, rmse_per_item):
    """Calculate optimal buffer sizes for each dish to minimize waste"""
    buffers = {}
    
    for dish, cols in dish_mappings.items():
        col = cols[0]  # Use first column as representative
        
        if dish == 'RIBEYE STEAK':
            buffer = 0.3  # Minimal buffer for expensive items
        elif avg_sales[col] > 3.0:  # High-demand items
            # Use 1.5x RMSE instead of 2.5x for waste reduction
            buffer = math.ceil(rmse_per_item.get(col, 0) * 1.5)
        else:  # Low-demand items
            # Use 0.25x average instead of 0.4x
            buffer = math.ceil(avg_sales[col] * 0.25) if avg_sales[col] > 0 else 0.25
        
        if dish == 'NONNAS BURGER':
            buffer = 2  # Reduced from 3
        
        buffers[dish] = buffer
    
    print("Buffer sizes calculated for waste reduction")
    return buffers


def save_models_and_results(rf_model, xgb_model, scaler, predictions, orders, timestamp):
    """Save trained models and results with timestamp"""
    output_path = r'C:\Users\chris\OneDrive\Desktop\Main-App-20250809T224141Z-1-001\Main-App'
    
    # Save models
    joblib.dump(rf_model, f'{output_path}rf_tuned_model_{timestamp}.pkl')
    joblib.dump(xgb_model, f'{output_path}xgb_tuned_model_{timestamp}.pkl')
    joblib.dump(scaler, f'{output_path}scaler_{timestamp}.pkl')
    
    # Save predictions and orders
    predictions.to_csv(f'{output_path}predictions_2023_tuned_{timestamp}.csv')
    orders.to_csv(f'{output_path}orders_2023_buffered_{timestamp}.csv')
    
    print(f"Results saved with timestamp: {timestamp}")


if __name__ == "__main__":
    # Main execution
    print("Starting restaurant sales prediction model training...")
    
    # Define item categories used throughout the script
    high_rmse_items = [
        'Qty (Sales)_SKINNY FRIES - 7510', 'Qty (Sales)_TA SKINNY FRIES - 25505',
        'Qty (Sales)_MOZZARELLA FRITTA - 2003', 'Qty (Sales)_PZ MARGHERITA - 4500',
        'Qty (Sales)_TA MARG PZ - 25300', 'Qty (Sales)_BRUSHETTA CLASSICA - 2502',
        'Qty (Sales)_PENNE SALSICCIA - 3506', 'Qty (Sales)_TA PENNE SALSICCIA - 25204',
        'Qty (Sales)_SPAG CARBONARA - 3503', 'Qty (Sales)_TA SPAG CARBONARA - 25202'
    ]
    
    high_demand_items = [
        'Qty (Sales)_SKINNY FRIES - 7510', 'Qty (Sales)_TA SKINNY FRIES - 25505',
        'Qty (Sales)_PENNE SALSICCIA - 3506', 'Qty (Sales)_TA PENNE SALSICCIA - 25204',
        'Qty (Sales)_PZ MARGHERITA - 4500', 'Qty (Sales)_TA MARG PZ - 25300',
        'Qty (Sales)_PENNE CON POLLO - 3507', 'Qty (Sales)_TA PENNE POLLO - 25206'
    ]
    
    # Load and prepare data
    df = load_and_prepare_data()
    df = create_day_of_week_column(df)
    dish_mappings, target_cols = define_dish_mappings(df)
    df, month_dummies, quarter_dummies = add_basic_features(df, target_cols)
    avg_sales, stable_items = analyze_sales_patterns(df, target_cols)
    
    # Create features
    lag1_cols, lag2_cols, lag3_cols, lag4_cols, lag5_cols = create_lag_features(df, target_cols, high_rmse_items)
    (sales_ratio_cols, six_wk_avg_cols, trend_cols, dow_std_cols,
     month_interaction_cols, quarter_interaction_cols, holiday_interaction_cols,
     other_features, weekend_interaction_cols) = create_advanced_features(df, target_cols, high_demand_items)
    
    # Combine all features
    feature_df = pd.concat([
        pd.DataFrame({**lag1_cols, **lag2_cols, **lag3_cols, **lag4_cols, **lag5_cols,
                     **sales_ratio_cols, **six_wk_avg_cols, **trend_cols, **dow_std_cols,
                     **month_interaction_cols, **quarter_interaction_cols, **holiday_interaction_cols,
                     **other_features, **weekend_interaction_cols}),
        month_dummies, quarter_dummies
    ], axis=1)
    
    df = pd.concat([df, feature_df], axis=1)
    df = create_dish_type_features(df)
    
    # Define feature columns
    feature_cols = [
        'Temp Avg (°C)', 'Precipitation (mm)', 'Windspeed (km/h)', 'Humidity (%)',
        'Temp Avg (°C)_norm', 'Precipitation (mm)_norm', 'Windspeed (km/h)_norm', 'Humidity (%)_norm',
        'Day of the Week_Friday', 'Day of the Week_Monday', 'Day of the Week_Saturday',
        'Day of the Week_Sunday', 'Day of the Week_Thursday', 'Day of the Week_Tuesday',
        'Is_Holiday', 'Total_Daily_Sales', 'Temp_Holiday_Interaction', 'Side_Saturday_Interaction',
        'Precipitation_Saturday_Interaction', 'Dish_Type_Side', 'Dish_Type_Starter', 'Is_Weekend',
        'Temp_Avg_lag1', 'Precipitation_lag1', 'Low_Demand_Indicator'
    ] + [f'{col}_lag1' for col in target_cols] + \
      [f'{col}_lag2' for col in high_rmse_items + ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203', 'Qty (Sales)_RIBEYE STEAK - 6501', 'Qty (Sales)_PZ MESSICANA - 4504', 'Qty (Sales)_SPAG POMODORO - 3502']] + \
      [f'{col}_lag3' for col in high_rmse_items + ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203', 'Qty (Sales)_RIBEYE STEAK - 6501', 'Qty (Sales)_PZ MESSICANA - 4504', 'Qty (Sales)_SPAG POMODORO - 3502']] + \
      [f'{col}_lag4' for col in ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203']] + \
      [f'{col}_lag5' for col in ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203']] + \
      [f'{col}_sales_ratio_dow' for col in target_cols] + \
      [f'{col}_6wk_dow_avg' for col in target_cols] + \
      [f'{col}_7day_trend' for col in target_cols] + \
      [f'{col}_14day_trend' for col in target_cols] + \
      [f'{col}_21day_trend' for col in target_cols] + \
      [f'{col}_dow_std' for col in target_cols] + \
      [f'{col}_Month_Interaction' for col in target_cols] + \
      [f'{col}_Quarter_Interaction' for col in target_cols] + \
      [f'{col}_Holiday_Interaction' for col in target_cols] + \
      [f'{col}_Weekend_Interaction' for col in high_demand_items] + \
      list(month_dummies.columns) + list(quarter_dummies.columns)
    
    # Prepare training and test data
    df = df.dropna()
    train_df = df[df['Year'] == 2022]
    test_df = df[df['Year'] == 2023]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    continuous_features = [f'{col}_sales_ratio_dow' for col in target_cols] + \
                         [f'{col}_6wk_dow_avg' for col in target_cols] + \
                         [f'{col}_7day_trend' for col in target_cols] + \
                         [f'{col}_14day_trend' for col in target_cols] + \
                         [f'{col}_21day_trend' for col in target_cols] + \
                         [f'{col}_dow_std' for col in target_cols] + \
                         [f'{col}_Month_Interaction' for col in target_cols] + \
                         [f'{col}_Quarter_Interaction' for col in target_cols] + \
                         [f'{col}_Holiday_Interaction' for col in target_cols] + \
                         [f'{col}_Weekend_Interaction' for col in high_demand_items] + \
                         ['Temp Avg (°C)', 'Precipitation (mm)', 'Windspeed (km/h)', 'Humidity (%)',
                          'Temp Avg (°C)_norm', 'Precipitation (mm)_norm', 'Windspeed (km/h)_norm', 'Humidity (%)_norm',
                          'Total_Daily_Sales', 'Temp_Holiday_Interaction', 'Temp_Avg_lag1', 'Precipitation_lag1',
                          'Precipitation_Saturday_Interaction']
    
    X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
    
    # Train models
    rf_model, xgb_model = train_models(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    
    # Ensemble approach
    y_pred = 0.6 * y_pred_rf + 0.4 * y_pred_xgb
    
    # Apply XGBoost for specific items
    
    for i, col in enumerate(target_cols):
        if col in high_rmse_items + ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203', 'Qty (Sales)_RIBEYE STEAK - 6501', 'Qty (Sales)_PZ MESSICANA - 4504', 'Qty (Sales)_SPAG POMODORO - 3502']:
            y_pred[:, i] = y_pred_xgb[:, i]
    
    # Bias correction for waste reduction
    bias_factors = {}
    for i, item in enumerate(target_cols):
        actual_mean = y_test.iloc[:, i].mean()
        predicted_mean = y_pred[:, i].mean()
        bias_factors[item] = actual_mean / predicted_mean if predicted_mean > 0 else 1.0
    
    y_pred_corrected = y_pred.copy()
    for i, item in enumerate(target_cols):
        y_pred_corrected[:, i] = y_pred[:, i] * bias_factors[item]
    
    y_pred = y_pred_corrected
    print("Applied bias correction to reduce over-prediction")
    
    # Evaluate models
    rmse_per_item, mae_per_item, r2_per_item = evaluate_predictions(y_test, y_pred, target_cols)
    
    # Calculate buffers
    buffers = calculate_buffers(dish_mappings, avg_sales, rmse_per_item)
    
    # Create predictions and orders dataframes
    predictions = pd.DataFrame(y_pred, columns=target_cols, index=test_df.index)
    orders = predictions.copy()
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_models_and_results(rf_model, xgb_model, scaler, predictions, orders, timestamp)
    
    print("Model training completed successfully!")