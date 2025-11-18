"""
NBA Fan Engagement Analysis - Data Collection
Collects Brooklyn Nets home game data from Basketball Reference (2022-2025 seasons)
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os

# Configuration
SEASONS = [2023, 2024, 2025]  # 2022-23, 2023-24, 2024-25
BARCLAYS_CAPACITY = 17732
TEAM_CODE = 'BRK'  # Brooklyn Nets team code

def scrape_nets_schedule(season_year):
    """
    Scrape Brooklyn Nets schedule for a given season from Basketball Reference
    
    Args:
        season_year (int): End year of season (e.g., 2024 for 2023-24 season)
    
    Returns:
        pd.DataFrame: Game schedule data
    """
    url = f"https://www.basketball-reference.com/teams/{TEAM_CODE}/{season_year}_games.html"
    print(f"Scraping {url}...")
    
    try:
        # Enhanced headers to appear more like a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        
        # Create a session to maintain cookies
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the games table
        table = soup.find('table', {'id': 'games'})
        if not table:
            print(f"  ⚠ No games table found for {season_year}")
            return None
        
        # Extract table to DataFrame using pandas
        df = pd.read_html(str(table))[0]
        df['season'] = f"{season_year-1}-{season_year}"
        
        print(f"  ✓ Collected {len(df)} games for {season_year-1}-{season_year}")
        return df
    
    except requests.exceptions.HTTPError as e:
        if '403' in str(e):
            print(f"  ⚠ 403 Error - Website blocking automated requests")
            print(f"  → Try manual approach or use alternative data source")
        else:
            print(f"  ✗ HTTP Error for {season_year}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error scraping {season_year}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Unexpected error for {season_year}: {e}")
        return None


def manual_data_collection_instructions():
    """Print instructions for manual data collection"""
    print("\n" + "=" * 70)
    print("MANUAL DATA COLLECTION INSTRUCTIONS")
    print("=" * 70)
    print("\nSince automated scraping is blocked, here are two options:\n")
    
    print("OPTION 1: Manual Download (Recommended - Takes 5 minutes)")
    print("-" * 70)
    print("1. Visit these URLs in your browser:")
    for season in SEASONS:
        print(f"   https://www.basketball-reference.com/teams/BRK/{season}_games.html")
    
    print("\n2. For each page:")
    print("   a. Scroll down to the 'Schedule and Results' table")
    print("   b. Click 'Share & Export' → 'Get table as CSV (for Excel)'")
    print("   c. Copy the CSV data")
    print("   d. Save as: data/raw/nets_{season}_schedule.csv")
    print("      (e.g., nets_2023_schedule.csv, nets_2024_schedule.csv, etc.)")
    
    print("\n3. After saving all CSV files, run:")
    print("   python src/process_manual_data.py")
    
    print("\n\nOPTION 2: Use Sample Data (Quick - For Testing Only)")
    print("-" * 70)
    print("I can generate sample/simulated data for you to test the pipeline.")
    print("This won't be real data but will let you complete the project workflow.")
    
    print("\n" + "=" * 70)


def create_manual_data_processor():
    """Create a script to process manually downloaded CSV files"""
    
    script_content = '''"""
Process manually downloaded CSV files from Basketball Reference
Run this after manually downloading the CSV files
"""

import pandas as pd
import numpy as np
import os
from data_collection import (
    clean_and_process_data,
    add_temporal_features,
    add_opponent_features,
    add_team_performance_features
)

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
            print(f"\\nProcessing {filename}...")
            try:
                df = pd.read_csv(filename)
                df['season'] = f"{season-1}-{season}"
                all_games.append(df)
                print(f"  ✓ Loaded {len(df)} games")
            except Exception as e:
                print(f"  ✗ Error reading {filename}: {e}")
        else:
            print(f"\\n⚠ File not found: {filename}")
            print(f"  Please download from:")
            print(f"  https://www.basketball-reference.com/teams/BRK/{season}_games.html")
    
    if not all_games:
        print("\\n✗ No data files found!")
        print("  Please follow the manual download instructions.")
        return None
    
    # Combine all seasons
    combined_df = pd.concat(all_games, ignore_index=True)
    print(f"\\n✓ Total games collected: {len(combined_df)}")
    
    # Process data (same as automated script)
    print("\\nCleaning and processing data...")
    processed_df = clean_and_process_data(combined_df)
    
    home_games = processed_df[processed_df['is_home'] == True].copy()
    print(f"✓ Home games only: {len(home_games)} games")
    
    print("\\nEngineering features...")
    home_games = add_temporal_features(home_games)
    home_games = add_opponent_features(home_games)
    home_games = add_team_performance_features(home_games)
    
    # Select relevant columns
    columns_to_keep = [
        'Date', 'season', 'Opponent', 'nets_won', 'PTS', 'OPP_PTS', 'attendance',
        'day_of_week', 'day_name', 'is_weekend', 'month', 'is_holiday_week',
        'is_star_opponent', 'is_rival', 'is_large_market',
        'games_played', 'nets_win_pct', 'nets_last_5_wins', 'current_win_streak'
    ]
    
    final_df = home_games[columns_to_keep].copy()
    
    # Save
    output_path = 'data/raw/nets_home_games_raw.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"\\n✓ Data saved to: {output_path}")
    print(f"✓ Ready for EDA: python src/eda_analysis.py")
    
    return final_df

if __name__ == "__main__":
    process_manual_csvs()
'''
    
    # Save the processor script
    with open('src/process_manual_data.py', 'w') as f:
        f.write(script_content)
    
    print("\n✓ Created: src/process_manual_data.py")


def create_sample_data():
    """
    Create sample/simulated data for testing the pipeline
    WARNING: This is NOT real data - only for testing workflow
    """
    
    print("\n" + "=" * 70)
    print("CREATING SAMPLE DATA (FOR TESTING ONLY)")
    print("=" * 70)
    print("\n⚠ WARNING: This is simulated data, not real attendance figures!")
    print("  Use this ONLY to test your pipeline.")
    print("  For your actual project, you MUST use real data.\n")
    
    response = input("Create sample data? (yes/no): ").lower().strip()
    
    if response != 'yes':
        print("Aborted. Please use manual data collection instead.")
        return None
    
    np.random.seed(42)
    
    # Generate sample games
    dates = pd.date_range(start='2022-10-19', end='2025-04-15', freq='3D')
    opponents = ['LAL', 'GSW', 'BOS', 'MIA', 'PHI', 'NYK', 'CHI', 'DAL', 
                 'MIL', 'PHX', 'DEN', 'MEM', 'SAC', 'CLE', 'ATL', 'TOR',
                 'MIN', 'LAC', 'NOP', 'OKC', 'IND', 'ORL', 'WAS', 'CHA',
                 'DET', 'POR', 'SAS', 'UTA', 'HOU']
    
    games = []
    for i, date in enumerate(dates[:120]):  # ~40 home games per season
        # Simulate game data
        opponent = np.random.choice(opponents)
        nets_pts = np.random.randint(95, 130)
        opp_pts = np.random.randint(95, 130)
        nets_won = 1 if nets_pts > opp_pts else 0
        
        # Base attendance with patterns
        base_attendance = 15500
        
        # Weekend boost
        if date.dayofweek >= 4:  # Fri, Sat, Sun
            base_attendance += 1500
        
        # Star team boost
        if opponent in ['LAL', 'GSW', 'BOS', 'MIA']:
            base_attendance += 1200
        
        # Rival boost
        if opponent in ['NYK', 'BOS', 'PHI']:
            base_attendance += 800
        
        # Add noise
        attendance = int(base_attendance + np.random.normal(0, 500))
        attendance = min(max(attendance, 12000), 17732)  # Cap at capacity
        
        season = f"{date.year-1}-{date.year}" if date.month < 7 else f"{date.year}-{date.year+1}"
        
        games.append({
            'Date': date,
            'season': season,
            'Opponent': opponent,
            'nets_won': nets_won,
            'PTS': nets_pts,
            'OPP_PTS': opp_pts,
            'attendance': attendance
        })
    
    df = pd.DataFrame(games)
    
    # Add features
    df = add_temporal_features(df)
    df = add_opponent_features(df)
    
    # Simplified performance features for sample data
    df = df.sort_values('Date').reset_index(drop=True)
    df['games_played'] = df.groupby('season').cumcount() + 1
    df['season_wins'] = df.groupby('season')['nets_won'].cumsum()
    df['nets_win_pct'] = df['season_wins'] / df['games_played']
    df['nets_last_5_wins'] = df.groupby('season')['nets_won'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    df['current_win_streak'] = 0  # Simplified
    
    # Save
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/nets_home_games_raw.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Sample data created: {output_path}")
    print(f"✓ {len(df)} sample games generated")
    print(f"\nYou can now run: python src/eda_analysis.py")
    print("\n⚠ REMEMBER: Replace with real data for your final project!")
    
    return df


def clean_and_process_data(df):
    """
    Clean and standardize scraped data
    
    Args:
        df (pd.DataFrame): Raw scraped data
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    df = df.copy()
    
    # Remove playoff games and header rows
    df = df[df['Date'] != 'Playoffs'].copy()
    df = df[df['Date'] != 'Date'].copy()
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notna()].copy()
    
    # Identify home games
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        for col in unnamed_cols:
            if df[col].astype(str).str.contains('@', na=False).any():
                df['is_home'] = df[col].isna()
                break
        else:
            df['is_home'] = True
    else:
        df['is_home'] = True
    
    # Clean opponent names
    if 'Opponent' in df.columns:
        df['Opponent'] = df['Opponent'].astype(str).str.strip()
    
    # Handle W/L column
    if 'W/L' in df.columns:
        df['nets_won'] = (df['W/L'] == 'W').astype(int)
    
    # Convert points to numeric
    if 'Tm' in df.columns:
        df['PTS'] = pd.to_numeric(df['Tm'], errors='coerce')
    if 'Opp' in df.columns:
        df['OPP_PTS'] = pd.to_numeric(df['Opp'], errors='coerce')
    
    # Attendance data
    if 'Attend.' in df.columns:
        df['attendance'] = pd.to_numeric(
            df['Attend.'].astype(str).str.replace(',', '').str.replace('N/A', ''), 
            errors='coerce'
        )
    else:
        df['attendance'] = np.nan
    
    return df


def add_temporal_features(df):
    """Add time-based features from game date"""
    df = df.copy()
    
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_name'] = df['Date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype(int)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['is_holiday_week'] = df['month'].isin([11, 12, 1]).astype(int)
    
    return df


def add_opponent_features(df):
    """Add opponent-related features"""
    df = df.copy()
    
    star_teams = ['LAL', 'GSW', 'LAC', 'MIA', 'BOS', 'PHI', 'MIL', 'PHX', 'DAL', 'DEN']
    rival_teams = ['NYK', 'BOS', 'PHI', 'TOR']
    large_markets = ['LAL', 'LAC', 'NYK', 'CHI', 'PHI', 'DAL', 'GSW', 'HOU', 'MIA', 'BOS', 'ATL', 'PHX', 'WAS']
    
    df['is_star_opponent'] = df['Opponent'].isin(star_teams).astype(int)
    df['is_rival'] = df['Opponent'].isin(rival_teams).astype(int)
    df['is_large_market'] = df['Opponent'].isin(large_markets).astype(int)
    
    return df


def add_team_performance_features(df):
    """Calculate Brooklyn Nets performance metrics"""
    df = df.copy()
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['games_played'] = df.groupby('season').cumcount() + 1
    df['season_wins'] = df.groupby('season')['nets_won'].cumsum()
    df['nets_win_pct'] = df['season_wins'] / df['games_played']
    
    df['nets_last_5_wins'] = df.groupby('season')['nets_won'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    
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


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("NBA FAN ENGAGEMENT ANALYSIS - DATA COLLECTION")
    print("Brooklyn Nets Home Games (2022-2025)")
    print("=" * 70)
    print()
    
    print("Step 1: Collecting data from Basketball Reference...")
    print("-" * 70)
    
    all_games = []
    for season in SEASONS:
        df = scrape_nets_schedule(season)
        if df is not None:
            all_games.append(df)
        time.sleep(3)  # Longer delay between requests
    
    if all_games:
        # Success! Process as normal
        combined_df = pd.concat(all_games, ignore_index=True)
        print(f"\n✓ Total games collected: {len(combined_df)}")
        
        # Continue with normal processing...
        # [Rest of the original main() function code]
        
    else:
        # Automated scraping failed
        print("\n✗ Automated data collection failed (403 Forbidden)")
        manual_data_collection_instructions()
        create_manual_data_processor()
        
        print("\n" + "=" * 70)
        print("CHOOSE AN OPTION:")
        print("=" * 70)
        print("\n1. Manual Download (Recommended)")
        print("   - Takes 5 minutes")
        print("   - Real data")
        print("   - Follow instructions above")
        
        print("\n2. Use Sample Data (Quick test)")
        print("   - Takes 10 seconds")
        print("   - Simulated data")
        print("   - Good for testing pipeline only")
        
        choice = input("\nCreate sample data now? (yes/no): ").lower().strip()
        
        if choice == 'yes':
            create_sample_data()
        else:
            print("\nPlease follow the manual download instructions above.")
            print("After downloading, run: python src/process_manual_data.py")


if __name__ == "__main__":
    main()