# Wildfire Risk Analytics and Prediction

## Project overview
This repository demonstrates an end-to-end data analytics and machine-learning workflow for identifying synthetic areas with higher wildfire risk.

**Important:** The included dataset is synthetic and is for academic/portfolio demonstration only. It is not an official wildfire dataset and must not be used for real emergency-response decisions.

## Skills demonstrated
- Data loading and validation
- Data cleaning / preprocessing
- Exploratory Data Analysis (EDA)
- Descriptive statistics
- Predictive analytics
- Random Forest classification
- Model evaluation
- Feature importance
- K-Means clustering
- Data visualization
- Reproducible Python workflow

## Files
- `wildfire_risk_analysis.py` — main Python analysis script
- `wildfire_risk_data.csv` — synthetic dataset
- `requirements.txt` — Python dependencies
- `Project_Report.pdf` — project documentation and results
- `README.md` — project documentation
- `outputs/` — generated figures and analysis tables

## How to run

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python wildfire_risk_analysis.py
```

The script creates an `outputs/` folder containing charts, cluster summaries, and the final analysis dataset.

## Methodology
1. Load and validate the dataset.
2. Calculate descriptive statistics.
3. Explore correlations and regional risk.
4. Split data into training and testing sets.
5. Train a Random Forest classifier.
6. Evaluate accuracy and classification metrics.
7. Inspect feature importance.
8. Apply K-Means clustering.
9. Save results and visualizations.

## Limitations
This is a synthetic demonstration project. Results are not real-world wildfire predictions. Real deployment would require validated data, geospatial validation, bias/fairness analysis, external testing, and expert oversight.

## GitHub
Create a new GitHub repository and upload all files in this project folder. After uploading, paste the repository URL into the internship form's **GitHub Repository** field.

Suggested repository name:
`wildfire-risk-analytics`
