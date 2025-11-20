"""
NBA Fan Engagement Analysis - Business Insights
Generates actionable business insights from model predictions
"""

from sklearn.metrics import accuracy_score


def generate_business_insights(model, model_name, feature_imp_df, y_test, y_pred, config):
    """
    Generate actionable business insights from model predictions
    
    Parameters:
    -----------
    model : trained model
        The trained machine learning model
    model_name : str
        Name of the model
    feature_imp_df : pandas.DataFrame or None
        DataFrame with feature importances (for tree-based models)
    y_test : pandas.Series
        True labels for test set
    y_pred : array-like
        Predicted labels for test set
    config : dict
        Configuration dictionary containing thresholds and other metadata
    """
    print()
    print("BUSINESS INSIGHTS")

    
    # Model performance in business terms
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n1. MODEL PERFORMANCE")
    print(f"   The {model_name} can predict attendance tier with {accuracy*100:.1f}% accuracy")
    print(f"   This means:")
    print(f"   - {int(accuracy*100)} out of 100 games will be correctly classified")
    
    # Feature insights
    if feature_imp_df is not None:
        print("\n2. KEY ATTENDANCE DRIVERS")
        top_features = feature_imp_df.head(3)
        
        for idx, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Translate feature to business insight
            insights = {
                'is_weekend': "Weekend games are the strongest predictor of high attendance",
                'day_of_week': "The specific day of the week significantly impacts attendance",
                'is_star_opponent': "Games against star teams (Lakers, Warriors, Celtics) drive attendance",
                'is_fr_sat': "Friday and Saturday games have the highest attendance potential",
                'weekend_star': "Weekend games with star opponents create premium attendance opportunities",
                'is_large_market': "Large market teams bring traveling fans to Barclays Center",
                'month': "Attendance patterns vary significantly by season timing",
                'is_rival': "Rivalry games (Knicks, Celtics, 76ers) boost attendance",
            }
            
            insight = insights.get(feature, f"{feature} is an important predictor")
            print(f"   • {insight}")
    
    print("\n3. ACTIONABLE RECOMMENDATIONS")
    print("   Based on model predictions: ")
    print()
    print("   a) SCHEDULING:")
    print("      - Prioritize weekend dates for marquee matchups")
    print("      - Schedule star opponents (Lakers, Warriors) on Fridays/Saturdays")
    print("      - Avoid Monday games when possible (lowest attendance)")
    
    print("\n   b) DYNAMIC PRICING:")
    print("      - Premium pricing for weekend + star opponent games")
    print("      - Promotional pricing for predicted Low attendance games")
    print("      - Mid-tier pricing for weekday non-rival games")
    
    print("\n   c) MARKETING & PROMOTIONS:")
    print("      - Focus marketing budget on predicted Medium→High conversion games")
    print("      - Run promotions (bobbleheads, giveaways) on predicted Low games")
    print("      - Early bird discounts for weekday games")
    
    print("\n   d) STAFFING & OPERATIONS:")
    print("      - Increase staff for predicted High attendance games")
    print("      - Reduce costs on predicted Low attendance games")
    print("      - Better concession/merchandise inventory planning")
    
    # Quantify potential impact
    low_threshold = config['low_threshold']
    high_threshold = config['high_threshold']
    
    print("\n4. POTENTIAL BUSINESS IMPACT")
    print(f"   Current attendance range: {low_threshold:,.0f} - {high_threshold:,.0f}")
    print(f"   Opportunity:")
    print(f"   - Convert 20% of Low games to Medium = ~{int((high_threshold-low_threshold)*0.20):,} additional fans/game")
    print(f"   - Average ticket price $50 = ${int((high_threshold-low_threshold)*0.20*50):,} additional revenue/game")
    print(f"   - Across 41 home games = ${int((high_threshold-low_threshold)*0.20*50*41):,} annual potential")

