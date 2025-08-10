import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import math
from streamlit_option_menu import option_menu
import os

# Load models/scaler (your .pkl in folder)
try:
    model_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
    rf_model = joblib.load(model_path)
    xgb_model_path = os.path.join(os.path.dirname(__file__), 'xgb_model.pkl')
    xgb_model = joblib.load(xgb_model_path)
    scaler_model_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    scaler_model = joblib.load(scaler_model_path)
    print("✅ Models loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

def define_constants():
    """Define application constants and configurations"""
    # High-demand items for special handling
    high_demand_items = [
        'Qty (Sales)_SKINNY FRIES - 7510', 'Qty (Sales)_TA SKINNY FRIES - 25505',
        'Qty (Sales)_PENNE SALSICCIA - 3506', 'Qty (Sales)_TA PENNE SALSICCIA - 25204',
        'Qty (Sales)_PZ MARGHERITA - 4500', 'Qty (Sales)_TA MARG PZ - 25300',
        'Qty (Sales)_PENNE CON POLLO - 3507', 'Qty (Sales)_TA PENNE POLLO - 25206'
    ]
    
    # UK holidays for 2022-2023
    holidays = [
        '2022-01-01', '2022-04-15', '2022-04-18', '2022-05-02', '2022-06-02',
        '2022-06-03', '2022-08-01', '2022-11-30', '2022-12-25', '2022-12-26',
        '2023-01-01', '2023-04-07', '2023-04-10', '2023-05-01', '2023-05-29',
        '2023-08-07', '2023-11-30', '2023-12-25', '2023-12-26'
    ]
    
    # Dish type classifications
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
    
    return high_demand_items, holidays, dish_types


@st.cache_data
def load_restaurant_data():
    """Load and preprocess restaurant sales data with improved error handling"""
    try:
        df_path = os.path.join(os.path.dirname(__file__), 'year1_year2_cleaned_fixed.csv')
        
        if not os.path.exists(df_path):
            st.error(f"Data file not found at: {df_path}")
            return None
        
        df = pd.read_csv(df_path)
        
        # Ensure Date column exists and convert to datetime
        if 'Date' not in df.columns:
            st.error("Date column not found in the data file")
            return None
        
        # Convert Date column to datetime with error handling
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Remove rows where date conversion failed
            df = df.dropna(subset=['Date'])
            if df.empty:
                st.error("No valid dates found in the data file")
                return None
        except Exception as e:
            st.error(f"Error converting dates: {e}")
            return None
        
        # Create day of week column from dummy variables
        day_columns = [
            'Day of the Week_Friday', 'Day of the Week_Monday', 'Day of the Week_Saturday',
            'Day of the Week_Sunday', 'Day of the Week_Thursday', 'Day of the Week_Tuesday',
            'Day of the Week_Wednesday'
        ]
        
        def get_day_of_week(row):
            for day in day_columns:
                if day in row and row[day] == 1:
                    return day.split('_')[-1]
            return 'Unknown'
        
        # Only apply if day columns exist
        existing_day_cols = [col for col in day_columns if col in df.columns]
        if existing_day_cols:
            df['Day of the Week'] = df.apply(get_day_of_week, axis=1)
        else:
            # Fallback: create day of week from datetime
            df['Day of the Week'] = df['Date'].dt.day_name()
        
        # Add time-based features
        _, holidays, dish_types = define_constants()
        df['Is_Holiday'] = df['Date'].isin(pd.to_datetime(holidays)).astype(int)
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        df['Is_Weekend'] = df['Day of the Week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
        
        # Add dish type features
        df['Dish_Type_Side'] = 0
        df['Dish_Type_Starter'] = 0
        
        for col, dtype in dish_types.items():
            if col in df.columns:
                if dtype == 'Side':
                    df['Dish_Type_Side'] = df['Dish_Type_Side'].where(df[col].isna(), 1)
                elif dtype == 'Starter':
                    df['Dish_Type_Starter'] = df['Dish_Type_Starter'].where(df[col].isna(), 1)
        
        # Calculate total daily sales
        sales_cols = [col for col in df.columns if col.startswith('Qty (Sales)_')]
        if sales_cols:
            df['Total_Daily_Sales'] = df[sales_cols].sum(axis=1)
        else:
            df['Total_Daily_Sales'] = 0
        
        print(f"✅ Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Define constants early
high_demand_items, holidays, dish_types = define_constants()

# Load CSV data with improved error handling
@st.cache_data
def load_data():
    """Load and preprocess data with comprehensive error handling"""
    df = load_restaurant_data()
    
    if df is None:
        st.error("Failed to load restaurant data")
        return None
    
    try:
        # Ensure Date column is properly formatted
        if 'Date' not in df.columns:
            st.error("Date column not found in data")
            return None
        
        # Verify date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            st.error("Date column is not in datetime format")
            return None
        
        # Create day of week column if not exists
        if 'Day of the Week' not in df.columns:
            df['Day of the Week'] = df['Date'].dt.day_name()
        
        # Add time-based features
        df['Is_Holiday'] = df['Date'].isin(pd.to_datetime(holidays)).astype(int)
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        df['Is_Weekend'] = df['Day of the Week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
        
        # Add dish type features
        df['Dish_Type_Side'] = 0
        df['Dish_Type_Starter'] = 0
        
        for col, dtype in dish_types.items():
            if col in df.columns:
                if dtype == 'Side':
                    df['Dish_Type_Side'] = df['Dish_Type_Side'].where(df[col].isna(), 1)
                elif dtype == 'Starter':
                    df['Dish_Type_Starter'] = df['Dish_Type_Starter'].where(df[col].isna(), 1)
        
        # Calculate total daily sales
        sales_cols = [col for col in df.columns if col.startswith('Qty (Sales)_')]
        if sales_cols:
            df['Total_Daily_Sales'] = df[sales_cols].sum(axis=1)
        else:
            df['Total_Daily_Sales'] = 0
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing dates: {str(e)}")
        return None

# Load data
df = load_data()
if df is None:
    st.error("Failed to load data. Please check the data file.")
    st.stop()

# Target columns for ML models
target_cols = [col for col in df.columns if col.startswith('Qty (Sales)_')]

# High RMSE items for XGBoost
high_rmse_items = [
    'Qty (Sales)_SKINNY FRIES - 7510', 'Qty (Sales)_TA SKINNY FRIES - 25505',
    'Qty (Sales)_MOZZARELLA FRITTA - 2003', 'Qty (Sales)_PZ MARGHERITA - 4500',
    'Qty (Sales)_TA MARG PZ - 25300', 'Qty (Sales)_BRUSHETTA CLASSICA - 2502',
    'Qty (Sales)_PENNE SALSICCIA - 3506', 'Qty (Sales)_TA PENNE SALSICCIA - 25204',
    'Qty (Sales)_SPAG CARBONARA - 3503', 'Qty (Sales)_TA SPAG CARBONARA - 25202'
]

# XGBoost specific items
xgb_specific_items = high_rmse_items + [
    'Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203',
    'Qty (Sales)_RIBEYE STEAK - 6501', 'Qty (Sales)_PZ MESSICANA - 4504', 
    'Qty (Sales)_SPAG POMODORO - 3502'
]

# Continuous features for scaling
continuous_features = [f'{col}_sales_ratio_dow' for col in target_cols] + \
                     [f'{col}_6wk_dow_avg' for col in target_cols] + \
                     [f'{col}_7day_trend' for col in target_cols] + \
                     [f'{col}_14day_trend' for col in target_cols] + \
                     [f'{col}_21day_trend' for col in target_cols] + \
                     [f'{col}_dow_std' for col in target_cols] + \
                     [f'{col}_Month_Interaction' for col in target_cols] + \
                     [f'{col}_Quarter_Interaction' for col in target_cols] + \
                     [f'{col}_Holiday_Interaction' for col in target_cols]

# Simple dish mappings
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

# Product images mapping with high-quality food photography
product_images = {
    'FOC ROSEMARY': 'https://images.unsplash.com/photo-1506280754576-f6fa8a873550?w=400&h=300&fit=crop&auto=format',  # Focaccia with rosemary
    'LASAGNE': 'https://images.unsplash.com/photo-1574894709920-11b28e7367e3?w=400&h=300&fit=crop&auto=format',  # Classic lasagne
    'PZ MARGHERITA': 'https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=400&h=300&fit=crop&auto=format',  # Margherita pizza
    'NONNAS BURGER': 'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&h=300&fit=crop&auto=format',  # Gourmet burger
    'PENNE ARRABBIATA': 'https://images.immediate.co.uk/production/volatile/sites/2/2023/09/Penne-arrabbiata-72de043.jpg?w=400&h=300&fit=crop&auto=format',  # Penne arrabbiata
    'PENNE CON POLLO': 'https://images.unsplash.com/photo-1598866594230-a7c12756260f?w=400&h=300&fit=crop&auto=format',  # Penne with chicken
    'PENNE SALSICCIA': 'https://redwoodkitchen.com/wp-content/uploads/2023/06/penne-con-Salsiccia-03.jpg?w=400&h=300&fit=crop&auto=format',  # Penne with sausage
    'SKINNY FRIES': 'https://images.unsplash.com/photo-1541592106381-b31e9677c0e5?w=400&h=300&fit=crop&auto=format',  # Crispy fries
    'SPAG BOLGNESE': 'https://img.taste.com.au/5qlr1PkR/taste/2016/11/spaghetti-bolognese-106560-1.jpeg?w=400&h=300&fit=crop&auto=format',  # Spaghetti bolognese
    'SPAG CARBONARA': 'https://static01.nyt.com/images/2021/02/14/dining/carbonara-horizontal/carbonara-horizontal-jumbo-v2.jpg?w=400&h=300&fit=crop&auto=format',  # Spaghetti carbonara
    'SPAG POMODORO': 'https://www.aline-made.com/wp-content/uploads/2024/06/Spaghetti-Pomodoro-2.jpg?w=400&h=300&fit=crop&auto=format',  # Spaghetti pomodoro
    'BRUSHETTA CLASSICA': 'https://images.unsplash.com/photo-1572695157366-5e585ab2b69f?w=400&h=300&fit=crop&auto=format',  # Classic bruschetta
    'MOZZARELLA FRITTA': 'https://images.unsplash.com/photo-1571091718767-18b5b1457add?w=400&h=300&fit=crop&auto=format',  # Fried mozzarella
    'PZ MESSICANA': 'https://foodandchips.com/wp-content/uploads/2019/02/MEXICANA.jpg?w=400&h=300&fit=crop&auto=format',  # Mexican pizza
    'RIBEYE STEAK': 'https://thesageapron.com/wp-content/uploads/2022/07/Grilled-Ribeye-7.jpg?w=400&h=300&fit=crop'  # Ribeye steak
}

# Fallback emoji-based images for when external URLs are blocked
fallback_images = {
    'FOC ROSEMARY': '🍞',
    'LASAGNE': '🍝',
    'PZ MARGHERITA': '🍕',
    'NONNAS BURGER': '🍔',
    'PENNE ARRABBIATA': '🍝',
    'PENNE CON POLLO': '🍝',
    'PENNE SALSICCIA': '🍝',
    'SKINNY FRIES': '🍟',
    'SPAG BOLGNESE': '🍝',
    'SPAG CARBONARA': '🍝',
    'SPAG POMODORO': '🍝',
    'BRUSHETTA CLASSICA': '🥖',
    'MOZZARELLA FRITTA': '🧀',
    'PZ MESSICANA': '🍕',
    'RIBEYE STEAK': '🥩'
}

# Function to safely display images with fallback
def safe_display_image(dish, image_url, fallback_emoji='🍽️'):
    """
    Safely display an image with fallback to emoji if loading fails.
    This handles cases where external images are blocked by ad blockers or firewalls.
    All images are displayed with consistent sizing (300px width, 200px height).
    """
    try:
        # Try to display the image with consistent sizing using HTML for exact control
        st.markdown(f'<div class="image-container"><img src="{image_url}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;" alt="{dish}"></div>', unsafe_allow_html=True)
        return True
    except Exception as e:
        # If image fails, display emoji fallback with consistent sizing
        st.markdown(f'<div class="image-container" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-size: 4rem; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">{fallback_emoji}</div>', unsafe_allow_html=True)
        return False

# Feature generation for ML models
def get_features_for_date(date, df):
    try:
        row_idx = df[df['Date'] == pd.to_datetime(date).strftime('%Y-%m-%d')].index[0]
        row = df.iloc[row_idx]
        prev_row = df.iloc[max(0, row_idx - 1)]
        
        # Get the expected feature names from the model
        expected_features = rf_model.feature_names_in_
        
        # Initialize input_data with all expected features set to 0
        input_data = {feature: 0 for feature in expected_features}
        
        # Fill in the actual values for features we can generate
        # Weather features
        weather_features = ['Temp Avg (°C)', 'Precipitation (mm)', 'Windspeed (km/h)', 'Humidity (%)']
        for feature in weather_features:
            if feature in expected_features:
                input_data[feature] = row.get(feature, 0)
                norm_feature = f'{feature}_norm'
                if norm_feature in expected_features:
                    input_data[norm_feature] = row.get(norm_feature, 0)
        
        # Day of week features
        day_features = ['Day of the Week_Friday', 'Day of the Week_Monday', 'Day of the Week_Saturday',
                       'Day of the Week_Sunday', 'Day of the Week_Thursday', 'Day of the Week_Tuesday']
        for feature in day_features:
            if feature in expected_features:
                input_data[feature] = row.get(feature, 0)
        
        # Basic features
        basic_features = ['Is_Holiday', 'Total_Daily_Sales', 'Dish_Type_Side', 'Dish_Type_Starter', 'Is_Weekend']
        for feature in basic_features:
            if feature in expected_features:
                input_data[feature] = row.get(feature, 0)
        
        # Interaction features
        if 'Temp_Holiday_Interaction' in expected_features:
            input_data['Temp_Holiday_Interaction'] = input_data['Temp Avg (°C)'] * input_data['Is_Holiday']
        
        if 'Precipitation_Saturday_Interaction' in expected_features:
            is_saturday = 1 if row.get('Day of the Week') == 'Saturday' else 0
            input_data['Precipitation_Saturday_Interaction'] = is_saturday * input_data['Precipitation (mm)']
        
        if 'Side_Saturday_Interaction' in expected_features:
            side_cols = ['Qty (Sales)_SKINNY FRIES - 7510', 'Qty (Sales)_TA SKINNY FRIES - 25505']
            side_sales = sum(row.get(col, 0) for col in side_cols)
            is_saturday = 1 if row.get('Day of the Week') == 'Saturday' else 0
            input_data['Side_Saturday_Interaction'] = is_saturday * side_sales
        
        # Lag features
        if 'Temp_Avg_lag1' in expected_features:
            input_data['Temp_Avg_lag1'] = prev_row.get('Temp Avg (°C)', 0)
        
        if 'Precipitation_lag1' in expected_features:
            input_data['Precipitation_lag1'] = prev_row.get('Precipitation (mm)', 0)
        
        if 'Low_Demand_Indicator' in expected_features:
            low_demand_cols = ['Qty (Sales)_RIBEYE STEAK - 6501', 
                              'Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203']
            input_data['Low_Demand_Indicator'] = 1 if all(prev_row.get(col, 0) == 0 for col in low_demand_cols) else 0
        
        # Add lag1 for all target columns
        for col in target_cols:
            lag1_feature = f'{col}_lag1'
            if lag1_feature in expected_features:
                input_data[lag1_feature] = prev_row.get(col, 0)
        
        # Add sales ratio features
        train_df = df[df['Year'] == 2022]
        for col in target_cols:
            ratio_feature = f'{col}_sales_ratio_dow'
            if ratio_feature in expected_features:
                sales_2022 = train_df.groupby('Day of the Week')[col].mean()
                sales_ratio = sales_2022.get(row.get('Day of the Week', 'Monday'), 1)
                current_sales = row.get(col, 0)
                input_data[ratio_feature] = current_sales / sales_ratio if sales_ratio != 0 else 1.0
        
        # Add 6-week rolling averages
        for col in target_cols:
            avg_feature = f'{col}_6wk_dow_avg'
            if avg_feature in expected_features:
                try:
                    group = df.groupby('Day of the Week')[col]
                    if row.get('Day of the Week') in group.indices and row_idx in group.indices[row.get('Day of the Week')]:
                        input_data[avg_feature] = group.rolling(window=6, min_periods=1).mean().loc[row.get('Day of the Week'), row_idx]
                    else:
                        input_data[avg_feature] = 0
                except:
                    input_data[avg_feature] = 0
        
        # Add trend features
        for col in target_cols:
            for trend_type in ['7day_trend', '14day_trend', '21day_trend']:
                trend_feature = f'{col}_{trend_type}'
                if trend_feature in expected_features:
                    if trend_type == '7day_trend':
                        start_idx = max(0, row_idx-6)
                    elif trend_type == '14day_trend':
                        start_idx = max(0, row_idx-13)
                    else:  # 21day_trend
                        start_idx = max(0, row_idx-20)
                    input_data[trend_feature] = df[col].iloc[start_idx:row_idx+1].mean()
        
        # Add day-of-week standard deviation
        for col in target_cols:
            std_feature = f'{col}_dow_std'
            if std_feature in expected_features:
                try:
                    dow_std = train_df.groupby('Day of the Week')[col].std().get(row.get('Day of the Week', 'Monday'), 0)
                    input_data[std_feature] = dow_std
                except:
                    input_data[std_feature] = 0
        
        # Add seasonality interactions
        for col in target_cols:
            for interaction_type in ['Month_Interaction', 'Quarter_Interaction', 'Holiday_Interaction']:
                interaction_feature = f'{col}_{interaction_type}'
                if interaction_feature in expected_features:
                    if interaction_type == 'Month_Interaction':
                        input_data[interaction_feature] = row.get(col, 0) * row.get('Month', 1)
                    elif interaction_type == 'Quarter_Interaction':
                        input_data[interaction_feature] = row.get(col, 0) * row.get('Quarter', 1)
                    else:  # Holiday_Interaction
                        input_data[interaction_feature] = row.get(col, 0) * input_data['Is_Holiday']
        
        # Add weekend interactions for high-demand items
        for col in high_demand_items:
            weekend_feature = f'{col}_Weekend_Interaction'
            if weekend_feature in expected_features:
                input_data[weekend_feature] = input_data['Is_Weekend'] * row.get(col, 0)
        
        # Add month/quarter dummies
        for m in range(2, 13):
            month_feature = f'Month_{m}'
            if month_feature in expected_features:
                input_data[month_feature] = 1 if row.get('Month', 1) == m else 0
        
        for q in range(2, 5):
            quarter_feature = f'Quarter_{q}'
            if quarter_feature in expected_features:
                input_data[quarter_feature] = 1 if row.get('Quarter', 1) == q else 0
        
        # Add lag2-5 for specific items
        prev2_row = df.iloc[max(0, row_idx - 2)]
        prev3_row = df.iloc[max(0, row_idx - 3)]
        prev4_row = df.iloc[max(0, row_idx - 4)]
        prev5_row = df.iloc[max(0, row_idx - 5)]
        
        lag2_cols = high_rmse_items + ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203', 
                                       'Qty (Sales)_RIBEYE STEAK - 6501', 'Qty (Sales)_PZ MESSICANA - 4504', 
                                       'Qty (Sales)_SPAG POMODORO - 3502']
        
        for col in lag2_cols:
            for lag_type in ['lag2', 'lag3']:
                lag_feature = f'{col}_{lag_type}'
                if lag_feature in expected_features:
                    if lag_type == 'lag2':
                        input_data[lag_feature] = prev2_row.get(col, 0)
                    else:  # lag3
                        input_data[lag_feature] = prev3_row.get(col, 0)
        
        lag4_cols = ['Qty (Sales)_SPAG BOLGNESE - 3504', 'Qty (Sales)_TA SPAG BOLOGNESE - 25203']
        for col in lag4_cols:
            for lag_type in ['lag4', 'lag5']:
                lag_feature = f'{col}_{lag_type}'
                if lag_feature in expected_features:
                    if lag_type == 'lag4':
                        input_data[lag_feature] = prev4_row.get(col, 0)
                    else:  # lag5
                        input_data[lag_feature] = prev5_row.get(col, 0)
        
        # Create DataFrame with features in the exact order expected by the model
        X = pd.DataFrame([input_data])
        X = X[expected_features]  # Ensure correct order
        
        # Scale continuous features if available
        X_scaled = X.copy()
        available_continuous = [f for f in continuous_features if f in X.columns]
        if available_continuous:
            try:
                X_scaled[available_continuous] = scaler_model.transform(X[available_continuous])
            except:
                # If scaling fails, use original values
                pass
        
        return X_scaled
        
    except Exception as e:
        st.error(f"Error generating features for date {date}: {str(e)}")
        return None

# ML-based prediction function with improved accuracy and bias correction
def simple_predict_sales(date, df):
    X_scaled = get_features_for_date(date, df)
    if X_scaled is None:
        return None
    
    # Get predictions from both models
    y_pred_rf = rf_model.predict(X_scaled)
    y_pred_xgb = xgb_model.predict(X_scaled)
    
    # Use ensemble weighting for better stability
    rf_weight = 0.6
    xgb_weight = 0.4
    y_pred = rf_weight * y_pred_rf + xgb_weight * y_pred_xgb
    
    # Use XGBoost for specific items that perform better with it
    ta_nonnas_idx = target_cols.index('Qty (Sales)_TA NONNAS BURGER - 25106')
    y_pred[:, ta_nonnas_idx] = y_pred_xgb[:, ta_nonnas_idx]
    
    # Use XGBoost for high RMSE items
    for i, col in enumerate(target_cols):
        if col in xgb_specific_items:
            y_pred[:, i] = y_pred_xgb[:, i]
    
    # Apply conservative adjustments to reduce over-prediction
    predictions = pd.DataFrame(y_pred, columns=target_cols)
    
    # Apply bias correction factors (calculated from model training)
    bias_factors = {
        'Qty (Sales)_FOC ROSEMARY - 7512': 0.85,
        'Qty (Sales)_TA FOC ROSE - 25600': 0.82,
        'Qty (Sales)_LASAGNE - 3513': 0.88,
        'Qty (Sales)_TA LASAGNE - 25200': 0.85,
        'Qty (Sales)_PZ MARGHERITA - 4500': 0.83,
        'Qty (Sales)_TA MARG PZ - 25300': 0.80,
        'Qty (Sales)_NONNAS BURGER - 6502': 0.87,
        'Qty (Sales)_TA NONNAS BURGER - 25106': 0.84,
        'Qty (Sales)_PENNE ARRABBIATA - 3500': 0.86,
        'Qty (Sales)_TA PENNE ARRABBIATA - 25208': 0.83,
        'Qty (Sales)_PENNE CON POLLO - 3507': 0.85,
        'Qty (Sales)_TA PENNE POLLO - 25206': 0.82,
        'Qty (Sales)_PENNE SALSICCIA - 3506': 0.84,
        'Qty (Sales)_TA PENNE SALSICCIA - 25204': 0.81,
        'Qty (Sales)_SKINNY FRIES - 7510': 0.83,
        'Qty (Sales)_TA SKINNY FRIES - 25505': 0.80,
        'Qty (Sales)_SPAG BOLGNESE - 3504': 0.86,
        'Qty (Sales)_TA SPAG BOLOGNESE - 25203': 0.83,
        'Qty (Sales)_SPAG CARBONARA - 3503': 0.85,
        'Qty (Sales)_TA SPAG CARBONARA - 25202': 0.82,
        'Qty (Sales)_SPAG POMODORO - 3502': 0.87,
        'Qty (Sales)_TA SPAG POMODORO - 25209': 0.84,
        'Qty (Sales)_BRUSHETTA CLASSICA - 2502': 0.88,
        'Qty (Sales)_MOZZARELLA FRITTA - 2003': 0.86,
        'Qty (Sales)_PZ MESSICANA - 4504': 0.84,
        'Qty (Sales)_RIBEYE STEAK - 6501': 0.89
    }
    
    # Apply bias correction to each column
    for col in predictions.columns:
        if col in bias_factors:
            predictions[col] = predictions[col] * bias_factors[col]
        else:
            predictions[col] = predictions[col] * 0.85  # Default 15% reduction
    
    # Ensure predictions are non-negative
    predictions = predictions.clip(lower=0)
    
    # Aggregate by dish
    dish_preds = {}
    for dish, cols in dish_mappings.items():
        dish_preds[dish] = sum(predictions.loc[0, col] for col in cols if col in predictions.columns)
    
    return dish_preds

# Adaptive ordering system that learns from waste patterns with enhanced waste reduction
def adaptive_predict_sales(date, df, waste_history=None):
    """Enhanced prediction with adaptive learning from waste patterns"""
    base_preds = simple_predict_sales(date, df)
    if base_preds is None:
        return None
    
    if waste_history is None:
        return base_preds
    
    # Apply waste-based adjustments with more aggressive waste reduction
    adjusted_preds = {}
    for dish, predicted in base_preds.items():
        if dish in waste_history:
            waste_ratio = waste_history[dish].get('waste_ratio', 0)
            
            # More aggressive waste reduction
            if waste_ratio > 0.25:  # Very high waste - reduce by 40%
                adjustment = 0.6
            elif waste_ratio > 0.2:  # High waste - reduce by 30%
                adjustment = 0.7
            elif waste_ratio > 0.15:  # Medium waste - reduce by 20%
                adjustment = 0.8
            elif waste_ratio > 0.1:  # Some waste - reduce by 10%
                adjustment = 0.9
            elif waste_ratio < 0.05:  # Very low waste - slight increase
                adjustment = 1.05
            else:
                adjustment = 1.0  # No adjustment
            
            adjusted_preds[dish] = predicted * adjustment
        else:
            adjusted_preds[dish] = predicted
    
    return adjusted_preds

# Enhanced simulation with waste learning
def run_adaptive_simulation():
    """Run simulation with adaptive learning"""
    # Initialize tracking
    waste_history = {dish: {'waste': 0, 'orders': 0, 'waste_ratio': 0} for dish in dish_mappings.keys()}
    inventory_sim = {dish: [] for dish in dish_mappings.keys()}
    
    # Simulation dates
    current_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    
    total_waste = 0
    total_stockouts = 0
    
    for day in range((end_date - current_date).days + 1):
        # Get adaptive predictions
        dish_preds = adaptive_predict_sales(current_date, df, waste_history)
        
        if dish_preds:
            # Process each dish
            for dish, predicted in dish_preds.items():
                # Calculate order with current buffer
                usable = sum(qty for qty, exp in inventory_sim[dish] if exp >= current_date)
                buffer = buffers.get(dish, 0)
                order = max(0, math.ceil(predicted + buffer - usable))
                
                if order > 0:
                    inventory_sim[dish].append((order, current_date + datetime.timedelta(days=3)))
                    waste_history[dish]['orders'] += order
                
                # Get actual sales
                actual_row = df[df['Date'] == pd.to_datetime(current_date).strftime('%Y-%m-%d')]
                if not actual_row.empty:
                    actual_row = actual_row.iloc[0]
                    actual_qty = sum(actual_row.get(col, 0) for col in dish_mappings[dish])
                    
                    # Process sales (FIFO)
                    remaining_demand = actual_qty
                    inventory_sim[dish].sort(key=lambda x: x[1])
                    
                    i = 0
                    while i < len(inventory_sim[dish]) and remaining_demand > 0:
                        qty, exp = inventory_sim[dish][i]
                        if exp >= current_date:
                            used = min(qty, remaining_demand)
                            inventory_sim[dish][i] = (qty - used, exp)
                            remaining_demand -= used
                        i += 1
                    
                    # Calculate waste and stockouts
                    waste_qty = sum(qty for qty, exp in inventory_sim[dish] if exp < current_date)
                    stockout_qty = max(0, remaining_demand)
                    
                    waste_history[dish]['waste'] += waste_qty
                    total_waste += waste_qty
                    total_stockouts += stockout_qty
                    
                    # Update waste ratio
                    if waste_history[dish]['orders'] > 0:
                        waste_history[dish]['waste_ratio'] = waste_history[dish]['waste'] / waste_history[dish]['orders']
                    
                    # Clean up inventory
                    inventory_sim[dish] = [(qty, exp) for qty, exp in inventory_sim[dish] 
                                         if exp >= current_date and qty > 0]
        
        current_date += datetime.timedelta(days=1)
    
    return total_waste, total_stockouts, waste_history

# Buffer calculations with improved waste reduction
def calculate_buffers():
    buffers = {}
    for dish, cols in dish_mappings.items():
        col = cols[0]  # First column as representative
        
        # Get historical average sales for this dish
        avg_sales = df[col].mean() if col in df.columns else 0
        
        # More conservative buffer strategy
        if dish == 'RIBEYE STEAK':
            buffer = 0.3  # Reduced from 0.5 to 0.3
        elif dish == 'NONNAS BURGER':
            buffer = 1.5  # Reduced from 3 to 1.5
        elif avg_sales > 5.0:  # High-demand items
            # Use 1.5x RMSE instead of 2.5x
            buffer = math.ceil(avg_sales * 0.3)  # 30% of average instead of RMSE multiplier
        elif avg_sales > 2.0:  # Medium-demand items
            buffer = math.ceil(avg_sales * 0.25)  # 25% of average
        else:  # Low-demand items
            buffer = math.ceil(avg_sales * 0.2)  # 20% of average
        
        buffers[dish] = buffer
    
    return buffers

buffers = calculate_buffers()

# Calculate dish-specific pricing from historical data
def calculate_dish_prices():
    """Calculate average price per portion for each dish from historical revenue data"""
    dish_prices = {}
    
    for dish, qty_cols in dish_mappings.items():
        total_revenue = 0
        total_quantity = 0
        
        # For each quantity column, find the corresponding revenue column
        for qty_col in qty_cols:
            if qty_col in df.columns:
                # Convert quantity column name to revenue column name
                revenue_col = qty_col.replace('Qty (Sales)_', 'Gross Rev._')
                
                if revenue_col in df.columns:
                    # Sum revenue and quantity across all days
                    revenue_sum = df[revenue_col].sum()
                    qty_sum = df[qty_col].sum()
                    
                    if qty_sum > 0:
                        total_revenue += revenue_sum
                        total_quantity += qty_sum
        
        if total_quantity > 0:
            dish_prices[dish] = total_revenue / total_quantity
        else:
            dish_prices[dish] = 10.0  # Default fallback
    
    return dish_prices

# Calculate dish-specific prices
dish_prices = calculate_dish_prices()

# Debug: Display dish prices (commented out for cleaner UI)
# st.sidebar.write("**Dish-Specific Pricing:**")
# for dish, price in dish_prices.items():
#     st.sidebar.write(f"{dish}: £{price:.2f}")

# Assumptions (now using dish-specific pricing)
# UNIT_SALES_PRICE and UNIT_COST will be calculated per dish

# Session state initialization
if 'current_date' not in st.session_state:
    st.session_state.current_date = datetime.date(2023, 1, 2)  # Start from Jan 2
if 'inventory' not in st.session_state:
    st.session_state.inventory = {dish: [] for dish in dish_mappings.keys()}
if 'waste' not in st.session_state:
    st.session_state.waste = {dish: 0 for dish in dish_mappings.keys()}
if 'stockouts' not in st.session_state:
    st.session_state.stockouts = {dish: 0 for dish in dish_mappings.keys()}
if 'pending_orders' not in st.session_state:
    st.session_state.pending_orders = {}
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = None
if 'tab_components' not in st.session_state:
    st.session_state.tab_components = {}
if 'tab_rendered' not in st.session_state:
    st.session_state.tab_rendered = {}

# Login
def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("Restaurant Management System")
        st.subheader("Login")
        st.write(f"Current Date: {st.session_state.current_date}")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            st.write("Login button clicked")
            if username == "manager" and password == "restaurant123":
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Incorrect credentials")
        st.stop()
    else:
        st.write("Already logged in, proceeding to main app")

check_login()

# Simulate day function
def simulate_day():
    date = st.session_state.current_date
    st.write(f"Simulate Day clicked: Processing {date}")
    for dish, cols in dish_mappings.items():
        try:
            actual_sales = sum(df[df['Date'] == pd.to_datetime(date).strftime('%Y-%m-%d')][col].values[0] for col in cols if col in df.columns)
        except IndexError:
            actual_sales = 0
        remaining_sales = actual_sales
        st.session_state.inventory[dish].sort(key=lambda x: x[1])
        i = 0
        while i < len(st.session_state.inventory[dish]) and remaining_sales > 0:
            qty, expiry = st.session_state.inventory[dish][i]
            if expiry >= date:
                used = min(qty, remaining_sales)
                st.session_state.inventory[dish][i] = (qty - used, expiry)
                remaining_sales -= used
            i += 1
        st.session_state.stockouts[dish] += max(0, remaining_sales)
        expired_qty = sum(qty for qty, expiry in st.session_state.inventory[dish] if expiry < date)
        st.session_state.waste[dish] += expired_qty
        st.session_state.inventory[dish] = [(qty, expiry) for qty, expiry in st.session_state.inventory[dish] if expiry >= date and qty > 0]

# Sidebar controls
with st.sidebar:
    st.title("Simulation Controls")
    new_date = st.date_input("Set Current Date", value=st.session_state.current_date, key=f"date_input_{st.session_state.current_date}")
    if st.button("Update Date"):
        st.write(f"Update Date clicked: Setting to {new_date}")
        st.session_state.current_date = new_date
        st.success(f"Date updated to {new_date.strftime('%Y-%m-%d')}")
    if st.button("Next Day"):
        st.write(f"Next Day clicked: Advancing to {st.session_state.current_date + datetime.timedelta(days=1)}")
        st.session_state.current_date += datetime.timedelta(days=1)
        simulate_day()
        st.success(f"Advanced to {st.session_state.current_date.strftime('%Y-%m-%d')}")
    if st.button("Clear Stock & Reset Waste"):
        st.write("Clear Stock & Reset Waste clicked")
        st.session_state.inventory = {dish: [] for dish in dish_mappings.keys()}
        st.session_state.waste = {dish: 0 for dish in dish_mappings.keys()}
        st.session_state.stockouts = {dish: 0 for dish in dish_mappings.keys()}
        st.session_state.pending_orders = {}
        st.success("Stock cleared and waste reset!")

# Initialize tab state management for HF deployment
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = None

# Menu with balanced state management for both local and cloud deployment
selected = option_menu(
    "Menu", 
    ["Dashboard", "Orders", "Products", "Accept Delivery", "Current Stock", "Full Year Sim", "Near Expiry"], 
    orientation="horizontal",
    key="main_menu"
)

# Enhanced tab switching logic for HF Spaces deployment
if selected != st.session_state.current_tab:
    # More aggressive clearing for HF Spaces
    ui_keys_to_clear = [k for k in st.session_state.keys() if 
                       k.startswith('predictions') or 
                       k.startswith('suggested_orders') or 
                       k.startswith('adaptive_results') or 
                       k.startswith('standard_results') or
                       k.startswith('monthly_') or
                       k.startswith('tab_') or
                       k.endswith('_rendered') or
                       k.endswith('_results') or
                       k.endswith('_data') or
                       'chart' in k.lower() or
                       'plot' in k.lower()]
    
    for key in ui_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Update current tab
    st.session_state.current_tab = selected
    
    # Set a flag to indicate we need a clean render
    st.session_state.needs_clean_render = True

# Check if we need a clean render and force rerun only once
if st.session_state.get('needs_clean_render', False):
    st.session_state.needs_clean_render = False
    st.rerun()

# Create main container for content with forced clearing
main_container = st.container()

with main_container:
    # Clear any existing content with multiple empty placeholders
    empty1 = st.empty()
    empty2 = st.empty()
    empty3 = st.empty()
    
    # Clear them immediately to ensure clean state
    empty1.empty()
    empty2.empty()
    empty3.empty()
    
    if selected == "Dashboard":
        # Dashboard content with isolated container
        with st.container():
            # Custom CSS for better styling
            st.markdown("""
            <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 2rem;
                padding: 1.5rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .metric-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                margin: 0.5rem 0;
                border-left: 4px solid #1f77b4;
                border: 2px solid #e9ecef;
            }
            .weather-card {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                border: 2px solid #fff;
            }
            .metric-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: #2c3e50;
                text-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            .metric-label {
                font-size: 0.9rem;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Main header
            st.markdown('<h1 class="main-header">📊 Restaurant Management Dashboard</h1>', unsafe_allow_html=True)
            
            # Get current date
            date = st.session_state.current_date
            
            # Current date display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border: 2px solid #fff;">
                    <h3 style="margin: 0; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3); font-weight: bold;">📅 Current Date: {date.strftime('%A, %B %d, %Y')}</h3>
                </div>
                """, unsafe_allow_html=True)
            row = df[df['Date'] == pd.to_datetime(date).strftime('%Y-%m-%d')]
            
            if row.empty:
                st.error("❌ No data available for this date.")
            else:
                row = row.iloc[0]
                
                # Weather section with improved layout
                st.markdown("### 🌤️ Weather Information")
                
                # Today's weather
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="weather-card">
                        <h4 style="margin: 0 0 1rem 0;">🌡️ Today's Weather</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <div class="metric-label">Temperature</div>
                                <div class="metric-value">{round(row.get('Temp Avg (°C)', 0), 1)}°C</div>
                            </div>
                            <div>
                                <div class="metric-label">Precipitation</div>
                                <div class="metric-value">{round(row.get('Precipitation (mm)', 0), 2)}mm</div>
                            </div>
                            <div>
                                <div class="metric-label">Wind Speed</div>
                                <div class="metric-value">{round(row.get('Windspeed (km/h)', 0), 1)} km/h</div>
                            </div>
                            <div>
                                <div class="metric-label">Humidity</div>
                                <div class="metric-value">{round(row.get('Humidity (%)', 0), 1)}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tomorrow's weather
                tomorrow_date = date + datetime.timedelta(days=1)
                tomorrow_row = df[df['Date'] == pd.to_datetime(tomorrow_date).strftime('%Y-%m-%d')]
                with col2:
                    if not tomorrow_row.empty:
                        tomorrow_row = tomorrow_row.iloc[0]
                        st.markdown(f"""
                        <div class="weather-card">
                            <h4 style="margin: 0 0 1rem 0;">🌅 Tomorrow's Weather</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                    <div class="metric-label">Temperature</div>
                                    <div class="metric-value">{round(tomorrow_row.get('Temp Avg (°C)', 0), 1)}°C</div>
                                </div>
                                <div>
                                    <div class="metric-label">Precipitation</div>
                                    <div class="metric-value">{round(tomorrow_row.get('Precipitation (mm)', 0), 2)}mm</div>
                                </div>
                                <div>
                                    <div class="metric-label">Wind Speed</div>
                                    <div class="metric-value">{round(tomorrow_row.get('Windspeed (km/h)', 0), 1)} km/h</div>
                                </div>
                                <div>
                                    <div class="metric-label">Humidity</div>
                                    <div class="metric-value">{round(tomorrow_row.get('Humidity (%)', 0), 1)}%</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="weather-card">
                            <h4 style="margin: 0 0 1rem 0;">🌅 Tomorrow's Weather</h4>
                            <div style="text-align: center; padding: 2rem;">
                                <p>No weather data available for tomorrow</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Sales comparison section with improved styling
            st.markdown("### 📈 Sales Performance Analysis")
            
            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            rev_cols = [col for col in df.columns if col.startswith('Gross Rev._')]
            
            if rev_cols:  # Only show if revenue columns exist
                # Get the day of week for the current date
                current_dow = pd.to_datetime(date).day_name()
                
                # Get 2023 data for the current date range
                start_date_2023 = datetime.date(date.year, 1, 2)
                end_date_2023 = date
                date_range_2023 = df[(df['Date'] >= pd.to_datetime(start_date_2023).strftime('%Y-%m-%d')) & 
                                     (df['Date'] <= pd.to_datetime(end_date_2023).strftime('%Y-%m-%d'))]
                
                if not date_range_2023.empty:
                    # Get all 2023 dates in the range
                    date_range_2023_dates = date_range_2023['Date'].tolist()
                    
                    # Group 2023 data by day of week and calculate averages
                    date_range_2023['Day_of_Week'] = pd.to_datetime(date_range_2023['Date']).dt.day_name()
                    # Calculate total sales for each row first, then group by day of week
                    date_range_2023['Total_Sales'] = date_range_2023[rev_cols].sum(axis=1)
                    avg_2023_by_dow = date_range_2023.groupby('Day_of_Week')['Total_Sales'].mean()
                    
                    # Create aligned 2022 data for the same period
                    aligned_2022_data = []
                    aligned_2022_dates = []
                    
                    for date_2023 in date_range_2023_dates:
                        date_2023_dt = pd.to_datetime(date_2023)
                        day_of_week = date_2023_dt.day_name()
                        
                        # Find the corresponding date in 2022 with the same day of week
                        start_2022 = pd.to_datetime('2022-01-02')
                        while start_2022.day_name() != day_of_week:
                            start_2022 += datetime.timedelta(days=1)
                        
                        # Calculate how many weeks from the start this date should be
                        weeks_from_start = (date_2023_dt - pd.to_datetime(start_date_2023)).days // 7
                        
                        # Calculate the corresponding 2022 date
                        corresponding_2022_date = start_2022 + datetime.timedelta(weeks=weeks_from_start)
                        
                        # Get the 2022 data for this date
                        date_2022_data = df[df['Date'] == corresponding_2022_date.strftime('%Y-%m-%d')]
                        
                        if not date_2022_data.empty:
                            aligned_2022_data.append(date_2022_data[rev_cols].sum(axis=1).iloc[0])
                            aligned_2022_dates.append(corresponding_2022_date.strftime('%Y-%m-%d'))
                        else:
                            aligned_2022_data.append(0)
                            aligned_2022_dates.append(corresponding_2022_date.strftime('%Y-%m-%d'))
                    
                    # Group 2022 data by day of week and calculate averages
                    aligned_2022_df = pd.DataFrame({
                        'Date': aligned_2022_dates,
                        'Sales': aligned_2022_data
                    })
                    aligned_2022_df['Day_of_Week'] = pd.to_datetime(aligned_2022_df['Date']).dt.day_name()
                    avg_2022_by_dow = aligned_2022_df.groupby('Day_of_Week')['Sales'].mean()
                    
                    # Create the comparison data with averages by day of week
                    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    comparison_data = pd.DataFrame({
                        '2023': [avg_2023_by_dow.get(day, 0) for day in days_of_week],
                        '2022': [avg_2022_by_dow.get(day, 0) for day in days_of_week]
                    }, index=days_of_week)
                    
                    # Create info cards for quick insights
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Data Points</div>
                            <div class="metric-value">{len(date_range_2023)}</div>
                            <div style="font-size: 0.8rem; color: #888;">Days analyzed</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        avg_2023 = comparison_data['2023'].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">2023 Average</div>
                            <div class="metric-value">£{avg_2023:.0f}</div>
                            <div style="font-size: 0.8rem; color: #888;">Daily sales</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        avg_2022 = comparison_data['2022'].mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">2022 Average</div>
                            <div class="metric-value">£{avg_2022:.0f}</div>
                            <div style="font-size: 0.8rem; color: #888;">Daily sales</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if comparison_data.empty or comparison_data[['2022', '2023']].eq(0).all().all():
                        st.warning("No sales data available for the selected date range.")
                    else:
                        # Chart section with improved styling
                        st.markdown("### 📊 Sales Comparison Chart")
                        st.markdown("*Hover over the lines to see both 2023 and 2022 dates for each point*")
                        
                        import plotly.graph_objects as go
                        
                        # Create a simple day-of-week comparison chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=days_of_week,
                            y=comparison_data['2023'],
                            mode='lines+markers',
                            name='2023',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=8),
                            hovertemplate='<b>%{x}</b><br>2023 Sales: £%{y:.0f}<extra></extra>'
                        ))
                        fig.add_trace(go.Scatter(
                            x=days_of_week,
                            y=comparison_data['2022'],
                            mode='lines+markers',
                            name='2022',
                            line=dict(color='#ff7f0e', width=3),
                            marker=dict(size=8),
                            hovertemplate='<b>%{x}</b><br>2022 Sales: £%{y:.0f}<extra></extra>'
                        ))
                        fig.update_layout(
                            title={
                                'text': "Sales Comparison by Day of Week",
                                'x': 0.5,
                                'xanchor': 'center',
                                'font': {'size': 18, 'color': '#1f77b4'}
                            },
                            xaxis_title="Day of Week",
                            yaxis_title="Sales (£)",
                            hovermode='x unified',
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=12),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
                        
                        # Show the data table with improved styling
                        st.markdown("### 📋 Detailed Data Table")
                        st.markdown("*Average sales values for each day of the week*")
                        
                        display_data = comparison_data.copy()
                        display_data.index.name = 'Day of Week'
                        display_data.columns = ['2022 Sales (£)', '2023 Sales (£)']
                        
                        # Format the data for better display
                        formatted_data = display_data.round(0).astype(int)
                        
                        # Add percentage change column
                        formatted_data['Change (%)'] = ((formatted_data['2023 Sales (£)'] - formatted_data['2022 Sales (£)']) / formatted_data['2022 Sales (£)'] * 100).round(2)
                        
                        # Apply styling with manual color coding
                        st.dataframe(formatted_data, use_container_width=True)
                else:
                    st.info("No sales data available for the current year.")
            else:
                st.info("No revenue data columns found in the dataset.")

    elif selected == "Orders":
        # Orders content with isolated container
        with st.container():
            st.title("Orders")
            
            # Create two columns for the buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Predictions", key="gen_pred_orders"):
                    st.write("Generate Predictions clicked")
                    dish_preds = simple_predict_sales(st.session_state.current_date, df)
                    if dish_preds:
                        st.session_state.predictions = dish_preds
                        # Calculate suggested orders
                        suggested_orders = {}
                        for dish, predicted in dish_preds.items():
                            usable_stock = sum(qty for qty, expiry in st.session_state.inventory.get(dish, []) if expiry >= st.session_state.current_date)
                            suggested_orders[dish] = max(0, math.ceil(predicted + buffers.get(dish, 0) - usable_stock))
                        st.session_state.suggested_orders = suggested_orders
                        st.success("Predictions generated.")
                    else:
                        st.error("Failed to generate predictions.")
            
            with col2:
                if 'predictions' in st.session_state:
                    if st.button("Submit Orders", key="submit_orders"):
                        st.write("Submit Orders clicked")
                        st.session_state.pending_orders = {}
                        order_table = []
                        for dish, predicted in st.session_state.predictions.items():
                            usable_stock = sum(qty for qty, expiry in st.session_state.inventory.get(dish, []) if expiry >= st.session_state.current_date)
                            suggested_order = st.session_state.suggested_orders.get(dish, 0)
                            order_table.append({"Dish": dish, "Predicted Sales": predicted, "Buffer": buffers.get(dish, 0), "Usable Stock": usable_stock, "Suggested Order": suggested_order})
                        
                        for row in order_table:
                            if row["Suggested Order"] > 0:
                                st.session_state.pending_orders[row["Dish"]] = row["Suggested Order"]
                        st.success("Orders submitted.")
            
            if 'predictions' in st.session_state:
                order_table = []
                for dish, predicted in st.session_state.predictions.items():
                    usable_stock = sum(qty for qty, expiry in st.session_state.inventory.get(dish, []) if expiry >= st.session_state.current_date)
                    suggested_order = st.session_state.suggested_orders.get(dish, 0)
                    order_table.append({"Dish": dish, "Predicted Sales": predicted, "Buffer": buffers.get(dish, 0), "Usable Stock": usable_stock, "Suggested Order": suggested_order})
                st.table(order_table)

    elif selected == "Products":
        # Products content with isolated container
        with st.container():
            # Enhanced Products header with styling
            st.markdown("""
            <style>
            .product-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                padding: 1.5rem;
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin: 1rem 0;
                border: 2px solid #e9ecef;
                text-align: center;
                transition: transform 0.2s ease-in-out;
                min-height: 500px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .product-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            .product-title {
                font-size: 1.2rem;
                font-weight: bold;
                color: #2c3e50;
                margin: 1rem 0 0.5rem 0;
                text-transform: capitalize;
            }
            .product-price {
                font-size: 1.1rem;
                color: #27ae60;
                font-weight: bold;
                margin: 0.5rem 0;
            }
            .suggested-order {
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                padding: 0.5rem;
                border-radius: 8px;
                font-size: 0.9rem;
                margin: 0.5rem 0;
            }
            .product-image {
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            .image-container {
                width: 100%;
                height: 200px;
                overflow: hidden;
                border-radius: 10px;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<h1 style="text-align: center; color: #2c3e50; margin-bottom: 2rem;">🍽️ Restaurant Products Menu</h1>', unsafe_allow_html=True)
            
            # Enhanced product grid with images
            cols = st.columns(3)
            order_inputs = {}
            suggested_orders = st.session_state.get('suggested_orders', {})
            
            for i, dish in enumerate(dish_mappings.keys()):
                with cols[i % 3]:
                    # Create product card
                    # st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    
                    # Display product image with error handling and fallback emojis
                    if dish in product_images:
                        fallback_emoji = fallback_images.get(dish, '🍽️')
                        safe_display_image(dish, product_images[dish], fallback_emoji)
                    else:
                        # Use emoji fallback if no image URL found
                        emoji = fallback_images.get(dish, '🍽️')
                        st.markdown(f'<div class="image-container" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-size: 4rem; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">{emoji}</div>', unsafe_allow_html=True)
                    
                    # Product title
                    st.markdown(f'<div class="product-title">{dish.replace("_", " ").title()}</div>', unsafe_allow_html=True)
                    
                    # Product price
                    dish_price = dish_prices.get(dish, 10.0)
                    st.markdown(f'<div class="product-price">£{dish_price:.2f}</div>', unsafe_allow_html=True)
                    
                    # Suggested order
                    suggested = suggested_orders.get(dish, 0)
                    if suggested > 0:
                        st.markdown(f'<div class="suggested-order">💡 Suggested Order: {suggested}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
                    
                    # Order input
                    qty = st.number_input(
                        "Amount to Order", 
                        value=0.0, 
                        step=1.0, 
                        key=f"prod_qty_{dish}",
                        help=f"Enter quantity for {dish.replace('_', ' ').title()}"
                    )
                    order_inputs[dish] = qty
                    
                    # Add to order button with styling
                    if st.button("🛒 Add to Order", key=f"add_{dish}", help=f"Add {dish} to your order"):
                        if qty > 0:
                            st.session_state.pending_orders[dish] = qty
                            st.success(f"✅ Added {qty} of {dish.replace('_', ' ').title()}")
                        else:
                            st.warning("⚠️ Please enter a quantity greater than 0")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced submit all button
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Submit All Orders", key="submit_all_products", help="Submit all product orders"):
                    st.session_state.pending_orders = {}
                    total_items = 0
                    for dish, qty in order_inputs.items():
                        if qty > 0:
                            st.session_state.pending_orders[dish] = qty
                            total_items += qty
                    
                    if total_items > 0:
                        st.success(f"🎉 Successfully added {total_items} items to your order!")
                        st.balloons()
                    else:
                        st.info("ℹ️ No items selected. Please add quantities to submit an order.")

    elif selected == "Accept Delivery":
        # Accept Delivery content with isolated container
        with st.container():
            st.title("Accept Delivery")
            
            # Accept All button at the top
            if st.button("Accept All", key="accept_all_delivery"):
                st.write("Accept All clicked")
                for dish, qty in list(st.session_state.pending_orders.items()):
                    if qty > 0:
                        st.session_state.inventory[dish].append((qty, st.session_state.current_date + datetime.timedelta(days=3)))
                        st.success(f"Accepted {qty} of {dish} into stock")
                st.session_state.pending_orders = {}
            
            # Individual item acceptance
            for dish, qty in list(st.session_state.pending_orders.items()):
                accepted_qty = st.number_input(dish, value=float(qty), step=1.0, key=f"acc_qty_{dish}")
                if st.button("Accept", key=f"acc_{dish}"):
                    st.write(f"Accept clicked for {dish}: {accepted_qty}")
                    if accepted_qty > 0:
                        st.session_state.inventory[dish].append((accepted_qty, st.session_state.current_date + datetime.timedelta(days=3)))
                    del st.session_state.pending_orders[dish]
                    st.success(f"Accepted {accepted_qty} of {dish} into stock")

    elif selected == "Current Stock":
        # Current Stock content with isolated container
        with st.container():
            st.title("Current Stock")
            stock = []
            for dish, inv in st.session_state.inventory.items():
                for qty, exp in inv:
                    if qty > 0:
                        days_left = (exp - st.session_state.current_date).days
                        stock.append({"Dish": dish, "Quantity": qty, "Expiry Date": exp, "Days Left": days_left})
            if stock:
                st.table(stock)
            else:
                st.info("No stock available.")

    elif selected == "Full Year Sim":
        # Full Year Sim content
        full_year_sim_container = st.container()
        with full_year_sim_container:
            st.title("Full Year Simulation")
            
            if st.button("Run Adaptive 2023 Simulation", key="run_adaptive_sim"):
                # st.write("Run Adaptive 2023 Simulation clicked")
                with st.spinner("Running adaptive simulation..."):
                    # Initialize tracking variables for adaptive simulation
                    monthly_waste_money = np.zeros(12)
                    monthly_stockouts_money = np.zeros(12)
                    total_revenue = 0
                    
                    # Initialize simulation inventory
                    inventory_sim = {dish: [] for dish in dish_mappings.keys()}
                    waste_history = {dish: {'waste': 0, 'orders': 0, 'waste_ratio': 0} for dish in dish_mappings.keys()}
                    
                    # Simulation dates
                    current_date = datetime.date(2023, 1, 1)
                    end_date = datetime.date(2023, 12, 31)
                    days = (end_date - current_date).days + 1
                    
                    progress = st.progress(0)
                    chart_placeholder = st.empty()
                
                for d in range(days):
                    progress.progress(d / days)
                    
                    # Get adaptive predictions
                    dish_preds = adaptive_predict_sales(current_date, df, waste_history)
                    if not dish_preds:
                        st.error(f"Adaptive simulation failed at {current_date}: No predictions.")
                        break
                    
                    # Get actual sales for this day
                    actual_row = df[df['Date'] == pd.to_datetime(current_date).strftime('%Y-%m-%d')]
                    if actual_row.empty:
                        current_date += datetime.timedelta(days=1)
                        continue
                    
                    actual_row = actual_row.iloc[0]
                    
                    # Initialize daily tracking
                    daily_waste_cost = 0
                    daily_stockout_revenue_loss = 0
                    daily_revenue = 0
                    
                    for dish, predicted in dish_preds.items():
                        # Get dish-specific pricing
                        dish_price = dish_prices.get(dish, 10.0)
                        dish_cost = dish_price * 0.25  # 25% cost assumption
                        
                        # Calculate actual sales for this dish
                        actual_qty = sum(actual_row.get(col, 0) for col in dish_mappings[dish])
                        daily_revenue += actual_qty * dish_price
                        
                        # Calculate order with adaptive buffer
                        usable = sum(qty for qty, exp in inventory_sim[dish] if exp >= current_date)
                        buffer = buffers.get(dish, 0)
                        
                        # Adjust buffer based on waste history - more aggressive waste reduction
                        if waste_history[dish]['orders'] > 0:
                            waste_ratio = waste_history[dish]['waste_ratio']
                            if waste_ratio > 0.2:  # High waste - reduce buffer more aggressively
                                buffer = max(0, buffer * 0.6)  # 40% reduction instead of 30%
                            elif waste_ratio > 0.15:  # Medium waste - reduce buffer
                                buffer = max(0, buffer * 0.75)  # 25% reduction
                            elif waste_ratio > 0.1:  # Some waste - reduce buffer slightly
                                buffer = max(0, buffer * 0.9)  # 10% reduction
                            elif waste_ratio < 0.05:  # Very low waste - increase buffer slightly
                                buffer = min(buffer * 1.1, buffers.get(dish, 0) * 1.3)
                        
                        order = max(0, math.ceil(predicted + buffer - usable))
                        
                        if order > 0:
                            inventory_sim[dish].append((order, current_date + datetime.timedelta(days=3)))
                            waste_history[dish]['orders'] += order
                        
                        # Process sales (FIFO)
                        remaining_demand = actual_qty
                        inventory_sim[dish].sort(key=lambda x: x[1])
                        
                        i = 0
                        while i < len(inventory_sim[dish]) and remaining_demand > 0:
                            qty, exp = inventory_sim[dish][i]
                            if exp >= current_date:
                                used = min(qty, remaining_demand)
                                inventory_sim[dish][i] = (qty - used, exp)
                                remaining_demand -= used
                            i += 1
                        
                        # Calculate waste and stockouts
                        waste_qty = sum(qty for qty, exp in inventory_sim[dish] if exp < current_date)
                        stockout_qty = max(0, remaining_demand)
                        
                        daily_waste_cost += waste_qty * dish_cost
                        daily_stockout_revenue_loss += stockout_qty * dish_price
                        
                        # Update waste history
                        waste_history[dish]['waste'] += waste_qty
                        if waste_history[dish]['orders'] > 0:
                            waste_history[dish]['waste_ratio'] = waste_history[dish]['waste'] / waste_history[dish]['orders']
                        
                        # Clean up inventory
                        inventory_sim[dish] = [(qty, exp) for qty, exp in inventory_sim[dish] 
                                             if exp >= current_date and qty > 0]
                    
                    # Add to monthly totals
                    month_idx = current_date.month - 1
                    monthly_waste_money[month_idx] += daily_waste_cost
                    monthly_stockouts_money[month_idx] += daily_stockout_revenue_loss
                    total_revenue += daily_revenue
                    
                    # Update chart with same detail as standard simulation
                    chart_data = pd.DataFrame({
                        'Waste £': monthly_waste_money,
                        'Stockouts £': monthly_stockouts_money
                    }, index=range(1, 13))
                    chart_placeholder.bar_chart(chart_data)
                    
                    current_date += datetime.timedelta(days=1)
                
                st.write("Adaptive simulation completed.")
                
                # Display results with same format as standard simulation
                year_waste = sum(monthly_waste_money)
                year_stockouts = sum(monthly_stockouts_money)
                
                st.write(f"Total Year Revenue £: {total_revenue:.2f}")
                st.write(f"Total Yearly Cost of Sales:  £{total_revenue * 0.25:.2f}")
                st.write(f"Total Year Waste: £{year_waste:.2f}")
                st.write(f"Total yearly waste Percentage [percentage waste vs total revenue] {year_waste/(total_revenue)*100:.2f}%")
                st.write(f"Total yearly waste Percentage [percentage waste vs total cost of sales] {year_waste/(total_revenue*0.25)*100:.2f}%")
                st.write(f"Total Year Stockouts: £{year_stockouts:.2f}")
                

                
                # Show detailed waste analysis
                st.write("**Adaptive Simulation - Waste Analysis:**")
                
                # Create pie chart for waste ratios
                waste_ratios_data = []
                waste_labels = []
                waste_values = []
                
                for dish, history in waste_history.items():
                    if history['orders'] > 0:
                        ratio = history['waste_ratio']
                        waste_labels.append(dish)
                        waste_values.append(ratio)
                        waste_ratios_data.append({"Dish": dish, "Waste Ratio": ratio})
                
                if waste_ratios_data:
                    # Create pie chart using plotly
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # Create DataFrame for plotting
                    waste_df = pd.DataFrame(waste_ratios_data)
                    
                    # Create pie chart
                    fig = px.pie(
                        waste_df, 
                        values='Waste Ratio', 
                        names='Dish',
                        title='Waste Ratios by Dish',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    # Update layout for better appearance
                    fig.update_layout(
                        title_x=0.5,
                        title_font_size=16,
                        showlegend=True,
                        height=500
                    )
                    
                    # Display the pie chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Also show the data in a table below the chart
                    st.write("**Waste Ratios Data:**")
                    waste_df['Waste Ratio %'] = waste_df['Waste Ratio'] * 100
                    st.dataframe(waste_df[['Dish', 'Waste Ratio %']].round(2))
                else:
                    st.write("No waste data available for visualization.")
                
                # Show improvement potential
                high_waste_dishes = [(dish, history['waste_ratio']) 
                                   for dish, history in waste_history.items() 
                                   if history['waste_ratio'] > 0.15]  # Lowered threshold to 15%
                
                if high_waste_dishes:
                    st.write("**Dishes with High Waste (>15%):**")
                    for dish, ratio in high_waste_dishes:
                        st.write(f"- {dish}: {ratio:.2%}")
                
                # Calculate and show improvement metrics
                if 'standard_results' in st.session_state:
                    standard_waste = st.session_state.standard_results.get('waste', 0)
                    standard_stockouts = st.session_state.standard_results.get('stockouts', 0)
                    
                    waste_improvement = ((standard_waste - year_waste) / standard_waste * 100) if standard_waste > 0 else 0
                    stockout_change = ((year_stockouts - standard_stockouts) / standard_stockouts * 100) if standard_stockouts > 0 else 0
                    
                    st.write("**Improvement vs Standard Simulation:**")
                    st.write(f"Waste Reduction: {waste_improvement:.1f}%")
                    st.write(f"Stockout Change: {stockout_change:.1f}%")
                
                # Store results for comparison
                st.session_state.adaptive_results = {
                    'waste': year_waste,
                    'stockouts': year_stockouts,
                    'revenue': total_revenue,
                    'waste_history': waste_history,
                    'monthly_waste': monthly_waste_money.tolist(),
                    'monthly_stockouts': monthly_stockouts_money.tolist()
                }
                
                # Add comparison section if both simulations have been run
                if 'standard_results' in st.session_state and 'adaptive_results' in st.session_state:
                    st.markdown("### 📊 Simulation Comparison")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Standard Simulation**")
                        st.write(f"Revenue: £{st.session_state.standard_results['revenue']:.2f}")
                        st.write(f"Waste: £{st.session_state.standard_results['waste']:.2f}")
                        st.write(f"Stockouts: £{st.session_state.standard_results['stockouts']:.2f}")
                    
                    with col2:
                        st.markdown("**Adaptive Simulation**")
                        st.write(f"Revenue: £{st.session_state.adaptive_results['revenue']:.2f}")
                        st.write(f"Waste: £{st.session_state.adaptive_results['waste']:.2f}")
                        st.write(f"Stockouts: £{st.session_state.adaptive_results['stockouts']:.2f}")
                    
                    with col3:
                        st.markdown("**Improvement**")
                        revenue_change = ((st.session_state.adaptive_results['revenue'] - st.session_state.standard_results['revenue']) / st.session_state.standard_results['revenue'] * 100) if st.session_state.standard_results['revenue'] > 0 else 0
                        waste_improvement = ((st.session_state.standard_results['waste'] - st.session_state.adaptive_results['waste']) / st.session_state.standard_results['waste'] * 100) if st.session_state.standard_results['waste'] > 0 else 0
                        stockout_change = ((st.session_state.adaptive_results['stockouts'] - st.session_state.standard_results['stockouts']) / st.session_state.standard_results['stockouts'] * 100) if st.session_state.standard_results['stockouts'] > 0 else 0
                        
                        st.write(f"Revenue: {revenue_change:+.1f}%")
                        st.write(f"Waste: {waste_improvement:+.1f}%")
                        st.write(f"Stockouts: {stockout_change:+.1f}%")

    elif selected == "Near Expiry":
        # Near Expiry content
        near_expiry_container = st.container()
        with near_expiry_container:
            st.title("Near Expiry Stock")
            near = []
            for dish, inv in st.session_state.inventory.items():
                for qty, exp in inv:
                    days_left = (exp - st.session_state.current_date).days
                    if 0 <= days_left <= 2:
                        near.append({"Dish": dish, "Qty": qty, "Expiry": exp, "Days Left": days_left})
            if near:
                st.table(near)
            else:
                st.info("No stock near expiry.")
