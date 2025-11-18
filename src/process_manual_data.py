"""
Process manually downloaded CSV files from Basketball Reference
Run this after manually downloading the CSV files
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import from data_collection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_manual_csv(df, season_year):
    """
    Clean manually downloaded CSV from Basketball Reference
    
    Args:
        df: Raw DataFrame from CSV
        season_year: Season end year (e.g., 2025 for 2024-25)
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Add season column
    df['season'] = f"{season_year-1}-{season_year}"
    
    # The @ symbol appears in the 5th column (index 5) for away games
    # If column 5 has @, it's away; if empty/NaN, it's home
    if len(df.columns) > 5:
        df['is_home'] = df.iloc[:, 5].isna() | (df.iloc[:, 5] == '')
    else:
        # Fallback: check if Opponent column has @
        df['is_home'] = ~df['Opponent'].str.contains('@', na=False)
    
    # Clean opponent names (remove @ if present)
    df['Opponent'] = df['Opponent'].str.replace('@', '').str.strip()
    
    # Extract just the 3-letter team code (some have full names)
    # e.g., "Atlanta Hawks" -> "ATL", "New York Knicks" -> "NYK"
    team_mapping = {
        'Atlanta Hawks': 'ATL',
        'Boston Celtics': 'BOS',
        'Brooklyn Nets': 'BRK',
        'Charlotte Hornets': 'CHA',
        'Chicago Bulls': 'CHI',
        'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL',
        'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW',
        'Houston Rockets': 'HOU',
        'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK',
        'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI',
        'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC',
        'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA',
        'Washington Wizards': 'WAS'
    }
    
    # Map full names to codes
    df['Opponent'] = df['Opponent'].replace(team_mapping)
    
    # Handle W/L column - determine if Nets won
    # W/L is in column index 11 (or look for 'W' and 'L' columns)
    if 'W' in df.columns and 'L' in df.columns:
        # Basketball Reference format: there are cumulative W and L columns
        # We need to check if this specific game was a win
        # The simplest way: compare Tm vs Opp scores
        df['nets_won'] = (pd.to_numeric(df['Tm'], errors='coerce') > 
                          pd.to_numeric(df['Opp'], errors='coerce')).astype(int)
    else:
        # Fallback
        df['nets_won'] = 0
    
    # Convert points to numeric
    df['PTS'] = pd.to_numeric(df['Tm'], errors='coerce')
    df['OPP_PTS'] = pd.to_numeric(df['Opp'], errors='coerce')
    
    # Attendance data
    if 'Attend.' in df.columns:
        df['attendance'] = pd.to_numeric(
            df['Attend.'].astype(str).str.replace(',', '').str.replace('N/A', ''), 
            errors='coerce'
        )
    else:
        df['attendance'] = np.nan
    
    # Keep only relevant columns
    columns_to_keep = ['Date', 'season', 'Opponent', 'is_home', 'nets_won', 
                       'PTS', 'OPP_PTS', 'attendance']
    
    # Only keep columns that exist
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    df = df[columns_to_keep].copy()
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df[df['Date'].notna()].copy()
    
    return df


def add_temporal_features(df):
    """Add time-based features from game date"""
    df = df.copy()
    
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['Date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype(int)  # Fri, Sat, Sun
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['is_holiday_week'] = df['month'].isin([11, 12, 1]).astype(int)
    
    return df


def add_opponent_features(df):
    """Add opponent-related features"""
    df = df.copy()
    
    # Star teams - consistently have star players that draw crowds
    star_teams = [
        'LAL', 'GSW', 'LAC', 'MIA', 'BOS', 'PHI', 'MIL', 'PHX', 'DAL', 'DEN'
    ]
    
    # Geographic rivals
    rival_teams = ['NYK', 'BOS', 'PHI', 'TOR']
    
    # Large market teams
    large_markets = [
        'LAL', 'LAC', 'NYK', 'CHI', 'PHI', 'DAL', 'GSW', 'HOU', 
        'MIA', 'BOS', 'ATL', 'PHX', 'WAS'
    ]
    
    df['is_star_opponent'] = df['Opponent'].isin(star_teams).astype(int)
    df['is_rival'] = df['Opponent'].isin(rival_teams).astype(int)
    df['is_large_market'] = df['Opponent'].isin(large_markets).astype(int)
    
    return df


def add_team_performance_features(df):
    """Calculate Brooklyn Nets performance metrics"""
    df = df.copy()
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate cumulative stats within each season
    df['games_played'] = df.groupby('season').cumcount() + 1
    df['season_wins'] = df.groupby('season')['nets_won'].cumsum()
    df['nets_win_pct'] = df['season_wins'] / df['games_played']
    
    # Last 5 games performance
    df['nets_last_5_wins'] = df.groupby('season')['nets_won'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    
    # Current win streak
    def calculate_streak(group):
        streaks = []
        streak = 0
        for won in group['nets_won']:
            if won == 1:
                streak = max(streak, 0) + 1
            else:
                streak = 0
            streaks.append(streak)
        return pd.Series(streaks, index=group.index)
    
    df['current_win_streak'] = df.groupby('season').apply(calculate_streak).droplevel(0)
    
    return df


def process_manual_csvs():
    """Process manually downloaded CSV files"""
    
    print("=" * 70)
    print("PROCESSING MANUAL DATA")
    print("=" * 70)
    
    seasons = [2023, 2024, 2025]
    all_games = []
    
    for season in seasons:
        filename = f"data/raw/nets_{season}_schedule.csv"
        
        if os.path.exists(filename):
            print(f"\nProcessing {filename}...")
            try:
                # Read CSV - Basketball Reference CSVs can have varying formats
                df = pd.read_csv(filename)
                
                # Clean the data
                df = clean_manual_csv(df, season)
                
                all_games.append(df)
                print(f"  ✓ Loaded {len(df)} games")
                
            except Exception as e:
                print(f"  ✗ Error reading {filename}: {e}")
                print(f"  Please check the file format.")
        else:
            print(f"\n⚠ File not found: {filename}")
            print(f"  Please download from:")
            print(f"  https://www.basketball-reference.com/teams/BRK/{season}_games.html")
    
    if not all_games:
        print("\n✗ No data files found!")
        print("\nTo manually download:")
        print("1. Visit: https://www.basketball-reference.com/teams/BRK/2025_games.html")
        print("2. Scroll to 'Schedule and Results' table")
        print("3. Click 'Share & Export' → 'Get table as CSV (for Excel)'")
        print("4. Copy all the text")
        print("5. Paste into a text editor")
        print("6. Save as: data/raw/nets_2025_schedule.csv")
        print("7. Repeat for 2024 and 2023")
        return None
    
    # Combine all seasons
    combined_df = pd.concat(all_games, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"✓ Total games collected: {len(combined_df)}")
    print(f"{'='*70}")
    
    # Filter for home games only
    home_games = combined_df[combined_df['is_home'] == True].copy()
    print(f"\n✓ Home games only: {len(home_games)} games")
    
    # Add all features
    print("\nEngineering features...")
    print("  → Adding temporal features...")
    home_games = add_temporal_features(home_games)
    
    print("  → Adding opponent features...")
    home_games = add_opponent_features(home_games)
    
    print("  → Adding team performance features...")
    home_games = add_team_performance_features(home_games)
    
    print("✓ Feature engineering complete")
    
    # Select final columns
    columns_to_keep = [
        'Date', 'season', 'Opponent', 'nets_won', 'PTS', 'OPP_PTS', 'attendance',
        'day_of_week', 'day_name', 'is_weekend', 'month', 'is_holiday_week',
        'is_star_opponent', 'is_rival', 'is_large_market',
        'games_played', 'nets_win_pct', 'nets_last_5_wins', 'current_win_streak'
    ]
    
    final_df = home_games[columns_to_keep].copy()
    
    # Save processed data
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/nets_home_games_raw.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"DATA PROCESSING COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n✓ Data saved to: {output_path}")
    
    # Summary statistics
    print(f"\nDataset Summary:")
    print(f"  Total home games: {len(final_df)}")
    print(f"  Seasons: {', '.join(final_df['season'].unique())}")
    print(f"  Date range: {final_df['Date'].min().date()} to {final_df['Date'].max().date()}")
    print(f"  Games with attendance: {final_df['attendance'].notna().sum()}")
    print(f"  Missing attendance: {final_df['attendance'].isna().sum()}")
    
    if final_df['attendance'].notna().sum() > 0:
        print(f"\nAttendance Stats:")
        print(f"  Mean: {final_df['attendance'].mean():,.0f}")
        print(f"  Median: {final_df['attendance'].median():,.0f}")
        print(f"  Min: {final_df['attendance'].min():,.0f}")
        print(f"  Max: {final_df['attendance'].max():,.0f}")
    
    print(f"\n{'='*70}")
    print(f"NEXT STEP")
    print(f"{'='*70}")
    print(f"\nRun EDA analysis:")
    print(f"  python src/eda_analysis.py")
    print(f"\n{'='*70}")
    
    return final_df


if __name__ == "__main__":
    df = process_manual_csvs()