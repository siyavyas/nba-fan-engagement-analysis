"""
NBA Fan Engagement Analysis - Project Setup
Creates the complete directory structure and initial files
"""

import os

def create_project_structure():
    """Create all necessary directories for the project"""
    
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'results/eda',
        'results/models',
        'tests'
    ]
    
    print("=" * 60)
    print("SETTING UP PROJECT STRUCTURE")
    print("=" * 60)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    print("\n" + "=" * 60)
    print("CREATING INITIAL FILES")
    print("=" * 60)
    
    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files (don't commit large datasets)
*.csv
*.xlsx
*.json
data/raw/*.csv
data/processed/*.csv

# Exception: keep small example/sample files
!data/raw/sample_*.csv
!data/processed/sample_*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Model files (can be large)
*.pkl
*.h5
*.pt
*.pth

# Results (keep structure, not files)
results/**/*.png
results/**/*.jpg
results/**/*.pdf

# Environment variables
.env

# Temporary files
*.tmp
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✓ Created: .gitignore")
    
    # Create requirements.txt
    requirements_content = """# Core dependencies
pandas==2.1.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.2

# Machine Learning
scikit-learn==1.3.0
xgboost==2.0.0

# Web scraping
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3

# Optional: NBA data library
basketball-reference-scraper==1.0.1

# Development tools
jupyter==1.0.0
ipython==8.14.0
pytest==7.4.0

# Code quality
black==23.7.0
flake8==6.1.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    print("✓ Created: requirements.txt")
    
    # Create README.md template
    readme_content = """# NBA Fan Engagement Analysis

Predicting Brooklyn Nets home game attendance using machine learning to generate actionable business insights for Brooklyn Sports & Entertainment.

## Project Overview

This project analyzes Brooklyn Nets home game data (2022-2025 seasons) to predict attendance levels and identify key drivers of fan engagement. The analysis provides data-driven recommendations for pricing, marketing, and scheduling strategies.

## Problem Statement

Brooklyn Sports & Entertainment (BSE) needs to optimize staffing, pricing, and marketing strategies based on predicted attendance patterns. This project builds a classification model to predict attendance tiers (Low/Medium/High) and identifies the most impactful factors driving fan engagement.

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
git clone https://github.com/yourusername/nba-fan-engagement-analysis.git
cd nba-fan-engagement-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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

[Your Name]

**Created for**: Brooklyn Sports & Entertainment Digital Fellowship Application

## License

MIT License

## Contact

- Email: your.email@example.com
- LinkedIn: [Your LinkedIn]
- GitHub: [Your GitHub]
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("✓ Created: README.md")
    
    # Create src/__init__.py
    with open('src/__init__.py', 'w') as f:
        f.write('"""NBA Fan Engagement Analysis Package"""\n')
    print("✓ Created: src/__init__.py")
    
    # Create empty placeholder files
    placeholders = [
        'src/data_collection.py',
        'src/eda_analysis.py',
        'src/preprocessing.py',
        'src/train.py',
        'tests/__init__.py',
        'notebooks/.gitkeep',
    ]
    
    for filepath in placeholders:
        with open(filepath, 'w') as f:
            if filepath.endswith('.py') and not filepath.endswith('__init__.py'):
                f.write('"""\nTODO: Implement this module\n"""\n\n')
            elif filepath.endswith('.gitkeep'):
                f.write('')
        print(f"✓ Created: {filepath}")
    
    print("\n" + "=" * 60)
    print("PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print("\nYour project structure:")
    print("""
nba-fan-engagement-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_collection.py
│   ├── eda_analysis.py
│   ├── preprocessing.py
│   └── train.py
├── results/
│   ├── eda/
│   └── models/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
""")
    
    print("\nNext steps:")
    print("  1. Create a virtual environment: python -m venv venv")
    print("  2. Activate it:")
    print("     - Mac/Linux: source venv/bin/activate")
    print("     - Windows: venv\\Scripts\\activate")
    print("  3. Install dependencies: pip install -r requirements.txt")
    print("  4. Start with data collection: edit src/data_collection.py")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    create_project_structure()
