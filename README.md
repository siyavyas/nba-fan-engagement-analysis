# NBA Fan Engagement Analysis

Predicting Brooklyn Nets home game attendance using machine learning to generate actionable business insights for Brooklyn Sports & Entertainment.

## Project Overview

This project analyzes Brooklyn Nets home game data (2022-2025 seasons) to predict attendance levels and identify key drivers of fan engagement. The analysis provides data-driven recommendations for pricing, marketing, and scheduling strategies.

## Problem Statement

This project builds a classification model to predict attendance tiers (Low/Medium/High) and identifies the most impactful factors driving fan engagement.

## Dataset

- **Source**: Basketball Reference, NBA Stats API
- **Time Period**: 2022-23, 2023-24, 2024-25 seasons
- **Games**: ~80-120 Brooklyn Nets home games at Barclays Center
- **Features**: 18+ engineered features including temporal patterns, opponent characteristics, and team performance metrics

## Methodology

1. **Data Collection**: Web scraping from Basketball Reference
2. **Exploratory Data Analysis**: Statistical analysis and visualization of attendance patterns
3. **Feature Engineering**: Created temporal, opponent, and performance-based features
4. **Model Development**: Trained and compared Logistic Regression, Random Forest, and XGBoost
5. **Model Evaluation**: Used temporal train/test split and weighted F1 score
6. **Business Insights**: Translated model results into actionable recommendations

## Key Findings

### Model Performance
- **Best Model**: Logistic Regression achieved **56.7% accuracy** with a weighted F1-score of **0.539**
- **Improvement**: 173.6% improvement over baseline (baseline F1: 0.197)
- **Model Comparison**:
  - Logistic Regression: 56.7% accuracy, 0.539 F1-score
  - Random Forest: 46.7% accuracy, 0.431 F1-score  
  - XGBoost: 43.3% accuracy, 0.414 F1-score

### Key Attendance Drivers
- **Weekend games** show significantly higher attendance than weekday games
- **Star opponent games** (Lakers, Warriors, Celtics) boost attendance substantially
- **Rivalry games** (Knicks, Celtics, 76ers) drive increased fan engagement
- **Day of week** and **season timing** are strong predictors of attendance levels
- **Interaction effects** between weekend games and star opponents create premium attendance opportunities

### Business Impact
- Model enables accurate attendance tier prediction (Low/Medium/High) for 57% of games
- Provides actionable insights for dynamic pricing, marketing, and staffing decisions
- Potential revenue optimization through strategic scheduling and targeted promotions

## Project Structure
```
nba-fan-engagement-analysis/
├── data/
│   ├── raw/              # Original scraped data
│   └── processed/        # Cleaned, feature-engineered data
├── notebooks/            # Jupyter notebooks
│   └── nba_fan_engagement_analysis.ipynb  # Complete analysis notebook
├── src/                  # Source code modules
│   ├── data_collection.py
│   ├── eda_analysis.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── business_insights.py
│   └── process_manual_data.py
├── results/
│   ├── eda/             # EDA visualizations
│   └── models/          # Model outputs and metrics
├── tests/               # Unit tests
├── requirements.txt
└── README.md
```

## Installation
```bash
# Clone the repository
git clone https://github.com/siyavyas/nba-fan-engagement-analysis.git
cd nba-fan-engagement-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Step 1: Collect data
python src/data_collection.py

# Step 2: Run EDA
python src/eda_analysis.py

# Step 3: Feature engineering
python src/feature_engineering.py

# Step 4: Train models
python src/train.py

# Step 5: View results
# Check results/eda/ for visualizations
# Check results/models/ for model performance metrics

# Alternative: Run complete analysis in Jupyter notebook
jupyter notebook notebooks/nba_fan_engagement_analysis.ipynb
```

## Results

### Model Performance

**Best Model: Logistic Regression**
- **Test Accuracy**: 56.7%
- **Weighted F1-Score**: 0.539
- **Baseline Comparison**: 173.6% improvement over baseline (baseline F1: 0.197)

**All Models Tested:**
| Model | Accuracy | F1-Score | Improvement over Baseline |
|-------|----------|----------|-------------------------|
| Baseline | 36.7% | 0.197 | - |
| Logistic Regression | **56.7%** | **0.539** | **173.6%** |
| Random Forest | 46.7% | 0.431 | 118.8% |
| XGBoost | 43.3% | 0.414 | 110.2% |

**Key Features (Top Predictors):**
- Weekend indicators (`is_weekend`, `is_fr_sat`)
- Opponent characteristics (`is_star_opponent`, `is_rival`, `is_large_market`)
- Interaction features (`weekend_star`, `weekend_rival`, `holiday_star`)
- Temporal features (`day_of_week`, `month`, `season_phase`)

### Business Recommendations

**1. Scheduling Strategy:**
- Prioritize weekend dates for marquee matchups (Lakers, Warriors, Celtics)
- Schedule star opponents on Fridays/Saturdays to maximize attendance
- Avoid Monday games when possible (lowest attendance)

**2. Dynamic Pricing:**
- Premium pricing for weekend + star opponent games
- Promotional pricing for predicted Low attendance games
- Mid-tier pricing for weekday non-rival games

**3. Marketing & Promotions:**
- Focus marketing budget on predicted Medium→High conversion games
- Run promotions (bobbleheads, giveaways) on predicted Low games
- Early bird discounts for weekday games

**4. Staffing & Operations:**
- Increase staff for predicted High attendance games
- Reduce costs on predicted Low attendance games
- Better concession/merchandise inventory planning

## Future Improvements

- Incorporate ticket pricing data
- Add social media sentiment analysis
- Build real-time prediction API
- Expand to other NBA teams for comparison

## Technologies Used

- **Python 3.9+**
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Web Scraping**: BeautifulSoup, Requests

## Author

Siya Vyas

## License

MIT License

## Contact

- Email: siyavyas02@rutgers.edu
- LinkedIn: https://www.linkedin.com/in/siya-vyas
- GitHub: https://github.com/siyavyas
