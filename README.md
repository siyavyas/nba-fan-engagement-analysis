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

*[To be filled after analysis]*

- Weekend games show X% higher attendance
- Star opponent games (Lakers, Warriors, Celtics) boost attendance by X%
- Model achieves X% accuracy (X% improvement over baseline)

## Project Structure
```
nba-fan-engagement-analysis/
├── data/
│   ├── raw/              # Original scraped data
│   └── processed/        # Cleaned, feature-engineered data
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code modules
│   ├── data_collection.py
│   ├── preprocessing.py
│   └── train.py
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

# Step 3: Train models
python src/train.py

# Step 4: View results
# Check results/eda/ for visualizations
# Check results/models/ for model performance metrics
```

## Results

*[To be filled after modeling]*

**Model Performance:**
- Best Model: XGBoost
- Accuracy: X%
- Weighted F1 Score: X
- Key Features: is_weekend, opponent_star_power, nets_last_5_wins

**Business Recommendations:**
1. Schedule high-profile matchups on weekend nights for maximum attendance
2. Implement dynamic pricing for star opponent games
3. Increase marketing spend during win streaks

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

**Created for**: Brooklyn Sports & Entertainment Digital Fellowship Application

## License

MIT License

## Contact

- Email: siyavyas02@rutgers.edu
- LinkedIn: https://www.linkedin.com/in/siya-vyas
- GitHub: https://github.com/siyavyas
