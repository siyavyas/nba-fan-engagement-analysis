"""
NBA Fan Engagement Analysis - Feature Engineering
Prepares data for machine learning models
"""


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

DATA_PATH = 'data/raw/nets_home_games_raw.csv'
OUTPUT_DIR = 'data/processed'
RANDOM_SEED = 42

def load_data():
    """Load the collected data"""
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("  Please run data collection first: python src/data_collection.py")
        return None
    
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])

    print()
    print("FEATURE ENGINEERING - DATA LOADED")
    print("-" * 70)
    print()
    print(f"Total games: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def create_interaction_features(df):
    """
    Create interaction features that capture combined effects

    Based on EDA insights:
    - Weekend x Star Opponent (both drive attendance)
    - Weekend x Rival (weekend games with rival teams)
    - Holiday x Star Opponent (holiday games with star opponents)
    """

    # Creating interaction features
    df = df.copy()

    df['weekend_star'] = df['is_weekend'] * df['is_star_opponent']
    df['weekend_rival'] = df['is_weekend'] * df['is_rival']
    df['holiday_star'] = df['is_holiday_week'] * df['is_star_opponent']
    # Considering star team thats also a rival - star x rival
    df['star_rival'] = df['is_star_opponent'] * df['is_rival']
    # Large market x Star Opponent (large market games with star opponents)
    df['market_star'] = df['is_large_market'] * df['is_star_opponent']

    return df


def create_temporal_features(df):
    """Create additional temporal features beyond basic ones"""

    df = df.copy()

    df['is_fr_sat'] = df['day_of_week'].isin([4, 5]).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)

    # Season phase (early, mid, late)
    #  NBA season: Oct-Dec = Early, Jan-Feb = Mid, Mar-Apr = Late
    def get_season_phase(month):
        if month in [10, 11, 12]:
            return 0 # Early
        elif month in [1, 2]:
            return 1 # Mid
        else: # 3, 4, 5
            return 2 # Late
    
    df['season_phase'] = df['month'].apply(get_season_phase)

    # dummy variables for season phase
    df['is_early_season'] = (df['season_phase'] == 0).astype(int)
    df['is_mid_season'] = (df['season_phase'] == 1).astype(int)
    df['is_late_season'] = (df['season_phase'] == 2).astype(int)

    return df


def handle_performance_features(df):
    """
    Based on EDA, performance features have weak/negative correlations
    We'll keep them but won't rely heavily on them
    """

    df = df.copy()

    # is team above .500?
    df['is_above_500'] = (df['nets_win_pct'] > 0.5).astype(int)
    # won majority of last 5 games?
    df['is_last_5_above_500'] = (df['nets_last_5_wins'] > 3).astype(int)
    # on any win streak?
    df['is_on_win_streak'] = (df['current_win_streak'] > 0).astype(int)

    return df


def select_features_for_modeling(df):
    """
    Select final feature set based on EDA insights
    """
    
    target = 'attendance'

    core_features = [
        'day_of_week',
        'is_weekend',
        'month',
        'is_holiday_week',
        'is_star_opponent',
        'is_rival',
        'is_large_market',
    ]

    interaction_features = [
        'weekend_star',
        'weekend_rival',
        'holiday_star',
        'star_rival',
        'market_star',
    ]

    temporal_features = [
        'is_fr_sat',
        'is_monday',
        'is_early_season',
        'is_mid_season',
        'is_late_season',
    ]

    performance_features = [
        'is_above_500',
        'is_last_5_above_500',
        'is_on_win_streak',
    ]

    feature_columns = (
        core_features +
        interaction_features +
        temporal_features +
        performance_features
    )

    print(f"\nFeature categories:")
    print(f"  Core features:        {len(core_features)}")
    print(f"  Interaction features: {len(interaction_features)}")
    print(f"  Temporal features:    {len(temporal_features)}")
    print(f"  Performance features: {len(performance_features)}")
    print(f"  ---")
    print(f"  TOTAL FEATURES:       {len(feature_columns)}")
    
    metadata_columns = ['Date', 'season', 'Opponent']
    
    # final dataset
    all_columns = metadata_columns + feature_columns + [target]
    df_final = df[all_columns].copy()
    
    print(f"\nFinal dataset shape: {df_final.shape}")
    
    return df_final, feature_columns, target


def create_temporal_train_test_split(df, feature_columns, target, test_size=0.25):
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate split point
    n_total = len(df)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    
    split_date = df.iloc[n_train]['Date']
    
    print(f"Split strategy: Temporal (chronological)")
    print(f"  Total games: {n_total}")
    print(f"  Training games: {n_train} ({(n_train/n_total)*100:.1f}%)")
    print(f"  Test games: {n_test} ({(n_test/n_total)*100:.1f}%)")
    print(f"  Split date: {split_date.date()}")
    
    # Split data
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    
    print(f"\nTraining period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"Test period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    
    # Extract features and target
    X_train = train_df[feature_columns].copy()
    y_train = train_df[target].copy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target].copy()
    
    # Keep metadata for analysis later
    train_metadata = train_df[['Date', 'season', 'Opponent']].copy()
    test_metadata = test_df[['Date', 'season', 'Opponent']].copy()
    
    print(f"\nSplit complete")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, train_metadata, test_metadata


def create_attendance_tiers(y_train, y_test):
    """
    Attendance tier labels for classification based on training data percentiles
    """

    # thresholds from training data
    low_threshold = y_train.quantile(0.33)
    high_threshold = y_train.quantile(0.67)

    print(f"Tier thresholds (from training data):")
    print(f"  Low:    < {low_threshold:,.0f}")
    print(f"  Medium: {low_threshold:,.0f} - {high_threshold:,.0f}")
    print(f"  High:   > {high_threshold:,.0f}")

    def assign_tier(attendance):
        if attendance < low_threshold:
            return 0 # Low
        elif attendance < high_threshold:
            return 1 # Medium
        else:
            return 2 # High

    y_train_class = y_train.apply(assign_tier)
    y_test_class = y_test.apply(assign_tier)

    print(f"\nTraining set distribution:")
    train_dist = y_train_class.value_counts().sort_index()
    for tier, count in train_dist.items():
        tier_name = ['Low', 'Medium', 'High'][tier]
        pct = (count / len(y_train_class)) * 100
        print(f"  {tier_name:>6}: {count:>3} ({pct:>5.1f}%)")
    
    print(f"\nTest set distribution:")
    test_dist = y_test_class.value_counts().sort_index()
    for tier, count in test_dist.items():
        tier_name = ['Low', 'Medium', 'High'][tier]
        pct = (count / len(y_test_class)) * 100
        print(f"  {tier_name:>6}: {count:>3} ({pct:>5.1f}%)")
    
    return y_train_class, y_test_class, low_threshold, high_threshold


def scale_features(X_train, X_test):
    """
    Standardize features
    Fitting scalar on training data only
    """

    # Scaling strategy: StandardScaler (zero mean, unit variance

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled, 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, 
        columns=X_test.columns, 
        index=X_test.index
    )

    print(f"\nFeature scaling complete")
    print(f"  Training mean: {X_train_scaled.mean().mean():.6f} (should be ~0)")
    print(f"  Training std: {X_train_scaled.std().mean():.6f} (should be ~1)")

    return X_train_scaled, X_test_scaled, scaler


def save_processed_data(X_train, X_test, X_train_scaled, X_test_scaled,
                        y_train, y_test, y_train_class, y_test_class,
                        train_metadata, test_metadata, scaler,
                        feature_columns, low_threshold, high_threshold):

    """
    Save processed data and artifacts
    """

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save datasets
    # Unscaled (for tree-based models)
    X_train.to_csv(f'{OUTPUT_DIR}/X_train.csv', index=False)
    X_test.to_csv(f'{OUTPUT_DIR}/X_test.csv', index=False)
    print("Saved: X_train.csv, X_test.csv (unscaled)")
    
    # Scaled (for linear models)
    X_train_scaled.to_csv(f'{OUTPUT_DIR}/X_train_scaled.csv', index=False)
    X_test_scaled.to_csv(f'{OUTPUT_DIR}/X_test_scaled.csv', index=False)
    print("Saved: X_train_scaled.csv, X_test_scaled.csv (scaled)")
    
    # Target variables
    y_train.to_csv(f'{OUTPUT_DIR}/y_train_regression.csv', index=False, header=['attendance'])
    y_test.to_csv(f'{OUTPUT_DIR}/y_test_regression.csv', index=False, header=['attendance'])
    print("Saved: y_train_regression.csv, y_test_regression.csv")
    
    y_train_class.to_csv(f'{OUTPUT_DIR}/y_train_classification.csv', index=False, header=['tier'])
    y_test_class.to_csv(f'{OUTPUT_DIR}/y_test_classification.csv', index=False, header=['tier'])
    print("Saved: y_train_classification.csv, y_test_classification.csv")
    
    # Metadata
    train_metadata.to_csv(f'{OUTPUT_DIR}/train_metadata.csv', index=False)
    test_metadata.to_csv(f'{OUTPUT_DIR}/test_metadata.csv', index=False)
    print("Saved: train_metadata.csv, test_metadata.csv")
    
    # Save scaler
    with open(f'{OUTPUT_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved: scaler.pkl")
    
    # Save feature list and configuration
    config = {
        'feature_columns': feature_columns,
        'n_features': len(feature_columns),
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
    
    with open(f'{OUTPUT_DIR}/config.pkl', 'wb') as f:
        pickle.dump(config, f)
    print("Saved: config.pkl")


def main():
    """Main execution function"""
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Step 1: Create interaction features
    df = create_interaction_features(df)
    
    # Step 2: Enhance temporal features
    df = create_temporal_features(df)
    
    # Step 3: Handle performance features
    df = handle_performance_features(df)
    
    # Step 4: Select features for modeling
    df_model, feature_columns, target = select_features_for_modeling(df)
    
    # Step 5: Train/test split (temporal)
    X_train, X_test, y_train, y_test, train_metadata, test_metadata = \
        create_temporal_train_test_split(df_model, feature_columns, target)
    
    # Step 6: Create classification target
    y_train_class, y_test_class, low_threshold, high_threshold = \
        create_attendance_tiers(y_train, y_test)
    
    # Step 7: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 8: Save everything
    save_processed_data(
        X_train, X_test, X_train_scaled, X_test_scaled,
        y_train, y_test, y_train_class, y_test_class,
        train_metadata, test_metadata, scaler,
        feature_columns, low_threshold, high_threshold
    )
    
    print(f"\nDataset ready for modeling:")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Target variable: attendance (continuous) + tier (classification)")


if __name__ == "__main__":
    main()