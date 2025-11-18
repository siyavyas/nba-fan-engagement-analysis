"""
NBA Fan Engagement Analysis - Exploratory Data Analysis
Analyzes patterns in Brooklyn Nets home game attendance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Load data
DATA_PATH = 'data/raw/nets_home_games_raw.csv'
OUTPUT_DIR = 'results/eda'

def load_data():
    """Load the collected data"""
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("  Please run data collection first: python src/data_collection.py")
        return None
    
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def print_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70 + "\n")


def basic_overview(df):
    """Display basic dataset information"""
    print_header("1. DATASET OVERVIEW")
    
    print(f"Total home games: {len(df)}")
    print(f"Seasons covered: {', '.join(df['season'].unique())}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  • {col} ({df[col].dtype})")
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            print(f"  • {col}: {count} ({pct:.1f}%)")
    else:
        print("  No missing values!")


def attendance_analysis(df):
    """Analyze attendance distribution and create tiers"""
    print_header("2. ATTENDANCE DISTRIBUTION")
    
    if df['attendance'].notna().sum() == 0:
        print("WARNING: No attendance data available!")
        print("  You'll need to collect this data for modeling.")
        return df
    
    attendance_data = df[df['attendance'].notna()]['attendance']
    
    print(f"Games with attendance data: {len(attendance_data)} / {len(df)}")
    
    print(f"\nAttendance Statistics:")
    print(f"  Mean:     {attendance_data.mean():>10,.0f}")
    print(f"  Median:   {attendance_data.median():>10,.0f}")
    print(f"  Std Dev:  {attendance_data.std():>10,.0f}")
    print(f"  Min:      {attendance_data.min():>10,.0f}")
    print(f"  Max:      {attendance_data.max():>10,.0f}")
    print(f"  25th %:   {attendance_data.quantile(0.25):>10,.0f}")
    print(f"  75th %:   {attendance_data.quantile(0.75):>10,.0f}")
    
    # Define attendance tiers
    low_threshold = attendance_data.quantile(0.33)
    high_threshold = attendance_data.quantile(0.67)
    
    print(f"\nProposed Attendance Tiers:")
    print(f"  Low:    < {low_threshold:,.0f}")
    print(f"  Medium: {low_threshold:,.0f} - {high_threshold:,.0f}")
    print(f"  High:   > {high_threshold:,.0f}")
    
    # Create tier column
    df['attendance_tier'] = pd.cut(
        df['attendance'],
        bins=[0, low_threshold, high_threshold, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    
    print(f"\nTier Distribution:")
    tier_counts = df['attendance_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = (count / tier_counts.sum()) * 100
        print(f"  {tier:>6}: {count:>3} games ({pct:>5.1f}%)")
    
    return df


def temporal_patterns(df):
    """Analyze temporal patterns in attendance"""
    print_header("3. TEMPORAL PATTERNS")
    
    if df['attendance'].notna().sum() == 0:
        print("No attendance data for temporal analysis")
        return
    
    # Day of week analysis
    print("Average Attendance by Day of Week:")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stats = df.groupby('day_name')['attendance'].agg(['mean', 'count'])
    day_stats = day_stats.reindex(day_order)
    
    for day in day_order:
        if day in day_stats.index and not pd.isna(day_stats.loc[day, 'mean']):
            mean_att = day_stats.loc[day, 'mean']
            count = day_stats.loc[day, 'count']
            print(f"  {day:>9}: {mean_att:>10,.0f}  ({count:>2} games)")
    
    # Weekend vs weekday
    weekend_avg = df[df['is_weekend'] == 1]['attendance'].mean()
    weekday_avg = df[df['is_weekend'] == 0]['attendance'].mean()
    
    if not pd.isna(weekend_avg) and not pd.isna(weekday_avg):
        weekend_lift = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        
        print(f"\nWeekend vs Weekday:")
        print(f"  Weekend average: {weekend_avg:>10,.0f}")
        print(f"  Weekday average: {weekday_avg:>10,.0f}")
        print(f"  Weekend lift:    {weekend_lift:>10.1f}%")
    
    # Monthly trends
    print("\nAverage Attendance by Month:")
    month_names = {10: 'Oct', 11: 'Nov', 12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May'}
    month_stats = df.groupby('month')['attendance'].agg(['mean', 'count'])
    
    for month in sorted(month_stats.index):
        month_name = month_names.get(month, str(month))
        mean_att = month_stats.loc[month, 'mean']
        count = month_stats.loc[month, 'count']
        if not pd.isna(mean_att):
            print(f"  {month_name:>3}: {mean_att:>10,.0f}  ({count:>2} games)")


def opponent_analysis(df):
    """Analyze opponent impact on attendance"""
    print_header("4. OPPONENT IMPACT")
    
    if df['attendance'].notna().sum() == 0:
        print("No attendance data for opponent analysis")
        return
    
    # Star opponents
    star_avg = df[df['is_star_opponent'] == 1]['attendance'].mean()
    non_star_avg = df[df['is_star_opponent'] == 0]['attendance'].mean()
    
    if not pd.isna(star_avg) and not pd.isna(non_star_avg):
        star_lift = ((star_avg - non_star_avg) / non_star_avg) * 100
        
        print("Star Opponents Impact:")
        print(f"  Star team games:     {star_avg:>10,.0f}")
        print(f"  Non-star games:      {non_star_avg:>10,.0f}")
        print(f"  Star team lift:      {star_lift:>10.1f}%")
    
    # Rival opponents
    rival_avg = df[df['is_rival'] == 1]['attendance'].mean()
    non_rival_avg = df[df['is_rival'] == 0]['attendance'].mean()
    
    if not pd.isna(rival_avg) and not pd.isna(non_rival_avg):
        rival_lift = ((rival_avg - non_rival_avg) / non_rival_avg) * 100
        
        print("\nRival Opponents Impact:")
        print(f"  Rival games:         {rival_avg:>10,.0f}")
        print(f"  Non-rival games:     {non_rival_avg:>10,.0f}")
        print(f"  Rival lift:          {rival_lift:>10.1f}%")
    
    # Top drawing opponents
    print("\nTop 10 Opponents by Average Attendance:")
    top_opponents = df.groupby('Opponent')['attendance'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
    
    for i, (opp, row) in enumerate(top_opponents.iterrows(), 1):
        print(f"  {i:>2}. {opp:>3}: {row['mean']:>10,.0f}  ({int(row['count']):>2} games)")


def performance_analysis(df):
    """Analyze team performance impact"""
    print_header("5. TEAM PERFORMANCE IMPACT")
    
    if df['attendance'].notna().sum() == 0:
        print("No attendance data for performance analysis")
        return
    
    # Correlations
    perf_features = ['nets_win_pct', 'nets_last_5_wins', 'current_win_streak']
    
    print("Correlation with Attendance:")
    for feature in perf_features:
        if feature in df.columns:
            corr = df[['attendance', feature]].corr().iloc[0, 1]
            print(f"  {feature:<20}: {corr:>6.3f}")
    
    # Win/loss comparison (note: this is outcome, not predictor)
    win_avg = df[df['nets_won'] == 1]['attendance'].mean()
    loss_avg = df[df['nets_won'] == 0]['attendance'].mean()
    
    if not pd.isna(win_avg) and not pd.isna(loss_avg):
        print(f"\nAttendance by Game Outcome:")
        print(f"  Games Nets won:  {win_avg:>10,.0f}")
        print(f"  Games Nets lost: {loss_avg:>10,.0f}")
        print(f"  Note: This is the outcome, not known before game")


def correlation_matrix(df):
    """Display correlation matrix"""
    print_header("6. FEATURE CORRELATIONS")
    
    if df['attendance'].notna().sum() == 0:
        print("No attendance data for correlation analysis")
        return
    
    numeric_features = [
        'attendance', 'day_of_week', 'is_weekend', 'month', 'is_holiday_week',
        'is_star_opponent', 'is_rival', 'is_large_market',
        'nets_win_pct', 'nets_last_5_wins', 'current_win_streak'
    ]
    
    # Filter to only features that exist
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    corr_matrix = df[numeric_features].corr()
    attendance_corr = corr_matrix['attendance'].sort_values(ascending=False)
    
    print("Correlations with Attendance (sorted):")
    for feature, corr_value in attendance_corr.items():
        if feature != 'attendance':
            bar = '█' * int(abs(corr_value) * 20)
            sign = '+' if corr_value > 0 else '-'
            print(f"  {feature:<25} {sign}{abs(corr_value):.3f}  {bar}")


def data_quality_checks(df):
    """Check data quality"""
    print_header("7. DATA QUALITY CHECKS")
    
    if df['attendance'].notna().sum() == 0:
        print("No attendance data for quality checks")
        return
    
    # Outlier detection
    Q1 = df['attendance'].quantile(0.25)
    Q3 = df['attendance'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['attendance'] < lower_bound) | (df['attendance'] > upper_bound)]
    
    print(f"Outliers (IQR method): {len(outliers)}")
    if len(outliers) > 0:
        print("\nOutlier games:")
        for _, game in outliers[['Date', 'Opponent', 'attendance', 'day_name']].head(5).iterrows():
            print(f"  {game['Date'].date()} vs {game['Opponent']:>3} - {game['attendance']:>6,.0f} ({game['day_name']})")
    
    # Feature distributions
    print("\nFeature Value Distributions:")
    print(f"  Weekend games:       {df['is_weekend'].sum():>3} ({df['is_weekend'].mean()*100:>5.1f}%)")
    print(f"  Star opponent games: {df['is_star_opponent'].sum():>3} ({df['is_star_opponent'].mean()*100:>5.1f}%)")
    print(f"  Rival games:         {df['is_rival'].sum():>3} ({df['is_rival'].mean()*100:>5.1f}%)")
    print(f"  Holiday week games:  {df['is_holiday_week'].sum():>3} ({df['is_holiday_week'].mean()*100:>5.1f}%)")


def create_visualizations(df):
    """Generate all EDA visualizations"""
    print_header("8. GENERATING VISUALIZATIONS")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if df['attendance'].notna().sum() == 0:
        print("No attendance data - skipping visualizations")
        return
    
    # Figure 1: Attendance Distribution
    print("  Creating: attendance_distribution.png")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(df['attendance'].dropna(), bins=25, edgecolor='black', alpha=0.7, color='#00A693')
    ax.axvline(df['attendance'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df["attendance"].mean():,.0f}')
    ax.axvline(df['attendance'].median(), color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {df["attendance"].median():,.0f}')
    ax.set_xlabel('Attendance', fontsize=12)
    ax.set_ylabel('Number of Games', fontsize=12)
    ax.set_title('Brooklyn Nets Home Game Attendance Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/attendance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Attendance by Day of Week
    print("  Creating: attendance_by_day.png")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_data = df.groupby('day_name')['attendance'].mean().reindex(day_order)
    
    colors = ['#00A693' if day in ['Friday', 'Saturday', 'Sunday'] else '#444444' for day in day_order]
    bars = ax.bar(range(len(day_data)), day_data.values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xticks(range(len(day_order)))
    ax.set_xticklabels(day_order, rotation=45, ha='right')
    ax.set_ylabel('Average Attendance', fontsize=12)
    ax.set_title('Average Attendance by Day of Week', fontsize=14, fontweight='bold')
    ax.axhline(df['attendance'].mean(), color='red', linestyle='--', alpha=0.5, label='Overall Average')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/attendance_by_day.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Attendance Over Time
    print("  Creating: attendance_time_series.png")
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors_season = ['#00A693', '#FFA500', '#DC143C']
    
    for i, season in enumerate(df['season'].unique()):
        season_data = df[df['season'] == season].sort_values('Date')
        ax.plot(season_data['Date'], season_data['attendance'], 
                marker='o', label=season, alpha=0.7, linewidth=2, 
                markersize=5, color=colors_season[i % len(colors_season)])
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Attendance', fontsize=12)
    ax.set_title('Brooklyn Nets Attendance Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/attendance_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Weekend vs Weekday
    print("  Creating: weekend_vs_weekday.png")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    weekend_data = [
        df[df['is_weekend'] == 0]['attendance'].dropna(),
        df[df['is_weekend'] == 1]['attendance'].dropna()
    ]
    
    bp = ax.boxplot(weekend_data, labels=['Weekday', 'Weekend'], patch_artist=True,
                    widths=0.6)
    
    for patch, color in zip(bp['boxes'], ['#444444', '#00A693']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Attendance', fontsize=12)
    ax.set_title('Attendance: Weekday vs Weekend', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    for i, data in enumerate(weekend_data, 1):
        mean_val = data.mean()
        ax.plot(i, mean_val, marker='D', color='red', markersize=10, 
                label='Mean' if i == 1 else '', zorder=5)
        ax.text(i + 0.2, mean_val, f'{mean_val:,.0f}', fontsize=10, va='center', fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/weekend_vs_weekday.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Opponent Category Impact
    print("  Creating: opponent_impact.png")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Star opponents
    star_data = [
        df[df['is_star_opponent'] == 0]['attendance'].dropna(),
        df[df['is_star_opponent'] == 1]['attendance'].dropna()
    ]
    bp1 = ax[0].boxplot(star_data, labels=['Regular', 'Star Team'], patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['#444444', '#FFA500']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax[0].set_title('Star Opponent Impact', fontweight='bold', fontsize=12)
    ax[0].set_ylabel('Attendance', fontsize=11)
    ax[0].grid(axis='y', alpha=0.3)
    
    # Rival opponents
    rival_data = [
        df[df['is_rival'] == 0]['attendance'].dropna(),
        df[df['is_rival'] == 1]['attendance'].dropna()
    ]
    bp2 = ax[1].boxplot(rival_data, labels=['Regular', 'Rival'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['#444444', '#DC143C']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax[1].set_title('Rival Opponent Impact', fontweight='bold', fontsize=12)
    ax[1].set_ylabel('Attendance', fontsize=11)
    ax[1].grid(axis='y', alpha=0.3)
    
    # Large market opponents
    market_data = [
        df[df['is_large_market'] == 0]['attendance'].dropna(),
        df[df['is_large_market'] == 1]['attendance'].dropna()
    ]
    bp3 = ax[2].boxplot(market_data, labels=['Small/Med', 'Large Market'], patch_artist=True)
    for patch, color in zip(bp3['boxes'], ['#444444', '#4169E1']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax[2].set_title('Market Size Impact', fontweight='bold', fontsize=12)
    ax[2].set_ylabel('Attendance', fontsize=11)
    ax[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/opponent_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 6: Correlation Heatmap
    print("  Creating: correlation_heatmap.png")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    heatmap_features = [
        'attendance', 'is_weekend', 'is_star_opponent', 'is_rival', 
        'is_large_market', 'nets_win_pct', 'nets_last_5_wins', 'current_win_streak'
    ]
    
    # Filter to existing features
    heatmap_features = [f for f in heatmap_features if f in df.columns]
    
    corr_matrix = df[heatmap_features].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 7: Team Performance vs Attendance
    print("  Creating: performance_vs_attendance.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Win percentage scatter
    mask1 = df['nets_win_pct'].notna() & df['attendance'].notna()
    axes[0].scatter(df[mask1]['nets_win_pct'], df[mask1]['attendance'], 
                    alpha=0.6, color='#00A693', s=50)
    axes[0].set_xlabel('Nets Win Percentage', fontsize=11)
    axes[0].set_ylabel('Attendance', fontsize=11)
    axes[0].set_title('Win % vs Attendance', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    if mask1.sum() > 1:
        z = np.polyfit(df[mask1]['nets_win_pct'], df[mask1]['attendance'], 1)
        p = np.poly1d(z)
        x_sorted = np.sort(df[mask1]['nets_win_pct'])
        axes[0].plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)
    
    # Last 5 wins scatter
    mask2 = df['nets_last_5_wins'].notna() & df['attendance'].notna()
    axes[1].scatter(df[mask2]['nets_last_5_wins'], df[mask2]['attendance'], 
                    alpha=0.6, color='#FFA500', s=50)
    axes[1].set_xlabel('Wins in Last 5 Games', fontsize=11)
    axes[1].set_ylabel('Attendance', fontsize=11)
    axes[1].set_title('Recent Form vs Attendance', fontweight='bold', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Win streak scatter
    mask3 = df['current_win_streak'].notna() & df['attendance'].notna()
    axes[2].scatter(df[mask3]['current_win_streak'], df[mask3]['attendance'], 
                    alpha=0.6, color='#DC143C', s=50)
    axes[2].set_xlabel('Current Win Streak', fontsize=11)
    axes[2].set_ylabel('Attendance', fontsize=11)
    axes[2].set_title('Win Streak vs Attendance', fontweight='bold', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/performance_vs_attendance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 8: Monthly Attendance Trends
    print("  Creating: monthly_attendance.png")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    month_names = {10: 'Oct', 11: 'Nov', 12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May'}
    monthly_avg = df.groupby('month')['attendance'].mean().sort_index()
    months = [month_names.get(m, str(m)) for m in monthly_avg.index]
    
    bars = ax.bar(months, monthly_avg.values, color='#00A693', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Attendance', fontsize=12)
    ax.set_title('Average Attendance by Month', fontsize=14, fontweight='bold')
    ax.axhline(df['attendance'].mean(), color='red', linestyle='--', alpha=0.5, 
               label='Season Average')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/monthly_attendance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 9: Attendance Tier Distribution
    if 'attendance_tier' in df.columns and df['attendance_tier'].notna().sum() > 0:
        print("  Creating: attendance_tiers.png")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        tier_counts = df['attendance_tier'].value_counts().sort_index()
        colors_tier = ['#DC143C', '#FFA500', '#00A693']  # Red, Orange, Green
        bars = ax.bar(tier_counts.index, tier_counts.values, color=colors_tier, 
                      alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Attendance Tier', fontsize=12)
        ax.set_ylabel('Number of Games', fontsize=12)
        ax.set_title('Distribution of Attendance Tiers', fontsize=14, fontweight='bold')
        
        # Add count and percentage labels
        total = tier_counts.sum()
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/attendance_tiers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nAll visualizations saved to results/eda/")


def generate_insights(df):
    """Generate key insights summary"""
    print_header("9. KEY INSIGHTS SUMMARY")
    
    if df['attendance'].notna().sum() == 0:
        print("No attendance data for insights generation")
        return
    
    insights = []
    
    # Weekend effect
    weekend_avg = df[df['is_weekend'] == 1]['attendance'].mean()
    weekday_avg = df[df['is_weekend'] == 0]['attendance'].mean()
    if not pd.isna(weekend_avg) and not pd.isna(weekday_avg):
        weekend_lift = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        insights.append(f"1. Weekend games draw {weekend_lift:.1f}% higher attendance than weekdays")
    
    # Star opponent effect
    if df['is_star_opponent'].sum() > 0:
        star_avg = df[df['is_star_opponent'] == 1]['attendance'].mean()
        non_star_avg = df[df['is_star_opponent'] == 0]['attendance'].mean()
        if not pd.isna(star_avg) and not pd.isna(non_star_avg):
            star_lift = ((star_avg - non_star_avg) / non_star_avg) * 100
            insights.append(f"2. Star opponent games boost attendance by {star_lift:.1f}%")
    
    # Rival effect
    if df['is_rival'].sum() > 0:
        rival_avg = df[df['is_rival'] == 1]['attendance'].mean()
        non_rival_avg = df[df['is_rival'] == 0]['attendance'].mean()
        if not pd.isna(rival_avg) and not pd.isna(non_rival_avg):
            rival_lift = ((rival_avg - non_rival_avg) / non_rival_avg) * 100
            insights.append(f"3. Rival games increase attendance by {rival_lift:.1f}%")
    
    # Performance correlation
    corr_perf = df[['attendance', 'nets_win_pct']].corr().iloc[0, 1]
    if not pd.isna(corr_perf):
        strength = 'strong' if abs(corr_perf) > 0.5 else ('moderate' if abs(corr_perf) > 0.3 else 'weak')
        insights.append(f"4. Team performance shows {strength} correlation (r={corr_perf:.2f}) with attendance")
    
    # Best drawing day
    day_avg = df.groupby('day_name')['attendance'].mean()
    if len(day_avg) > 0:
        best_day = day_avg.idxmax()
        best_day_avg = day_avg.max()
        insights.append(f"5. {best_day} games have highest average attendance ({best_day_avg:,.0f})")
    
    # Top opponent
    opp_avg = df.groupby('Opponent')['attendance'].mean()
    if len(opp_avg) > 0:
        top_opp = opp_avg.idxmax()
        top_opp_avg = opp_avg.max()
        insights.append(f"6. {top_opp} draws the highest average attendance ({top_opp_avg:,.0f})")
    
    # Print insights
    for insight in insights:
        print(f"  {insight}")
    
    return insights


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("NBA FAN ENGAGEMENT ANALYSIS - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Run all analyses
    basic_overview(df)
    df = attendance_analysis(df)
    temporal_patterns(df)
    opponent_analysis(df)
    performance_analysis(df)
    correlation_matrix(df)
    data_quality_checks(df)
    create_visualizations(df)
    insights = generate_insights(df)
      
    if df['attendance'].isna().sum() > 0:
        print("\n⚠ IMPORTANT: Missing attendance data detected!")
        print(f"  Games with attendance: {df['attendance'].notna().sum()}")
        print(f"  Games missing attendance: {df['attendance'].isna().sum()}")
        print("\n  Options:")
        print("  1. Collect missing attendance from ESPN box scores")
        print("  2. Proceed with available data (may limit model accuracy)")
        print("  3. Use partial dataset for proof of concept")


if __name__ == "__main__":
    main()
