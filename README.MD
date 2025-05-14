# 🗽NYC Airbnb Host Popularity Prediction App

This project predicts **Airbnb host popularity** using a stacked machine learning model that combines **XGBoost** and **Random Forest** with a **Ridge Regression** meta-learner. The notebook constructs a custom popularity score using review volume, listing price, and recent activity, making this a valuable tool for host performance analytics.

## Features

* Custom popularity score based on:

  * Total number of reviews (log-scaled)
  * Listing price
  * Recency of last review
* Ensemble model: XGBoost + Random Forest + Ridge
* Log-transformed regression target
* Clean, normalized feature space
* Feature importance visualization
* Ready for extension to Streamlit or web app

## Project Structure

```
├── notebook.ipynb                # Full notebook for data prep, modeling, and evaluation
├── checkpoint.pth                # Saved model checkpoint
├── popularity_model.pkl          # Final model (optional pickle format)
├── README.md                     # Project overview and setup
```

## Model Performance

| Metric | Value    |
| ------ | -------- |
| MAE    | \~0.0055 |
| RMSE   | \~0.0101 |
| R²     | \~0.9958 |

## Setup Instructions

```bash
git clone https://github.com/berkde/New-York-City-Airbnb
cd New-York-City-Airbnb
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Requirements

See `requirements.txt` for dependencies. Key libraries include:

* `xgboost`
* `scikit-learn`
* `pandas`
* `matplotlib`
* `numpy`


## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. The popularity score is derived from data engineering decisions that may not reflect Airbnb's internal rankings or algorithms. Users should not use this tool for real-world investment, pricing, or host comparison decisions without further validation.

The author assumes no responsibility for any outcomes resulting from use of this project.
