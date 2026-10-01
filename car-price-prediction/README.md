# CarPriceAI — AI-Powered Car Price Prediction

A production-quality end-to-end machine-learning web application that predicts the selling price of a car based on vehicle specifications.

---

## Features

- **6 regression models** trained and compared automatically (Linear, Ridge, Random Forest, Gradient Boosting, Extra Trees, HistGradient Boosting)
- **Best model selected** based on R² and cross-validation performance
- **Complete scikit-learn Pipeline** — preprocessing + model saved together (no data leakage)
- **REST API** via Flask — POST `/predict` returns JSON
- **Modern responsive UI** — dark-navy dashboard aesthetic
- **Interactive charts** — Actual vs Predicted, R² comparison, Feature Importance, Price Distribution
- **No fake metrics** — all displayed numbers come from actual training

---

## Technology Stack

| Layer        | Technology                             |
|-------------|---------------------------------------|
| Language     | Python 3.9+                           |
| Web Framework| Flask                                 |
| ML Library   | scikit-learn                          |
| Data         | pandas, NumPy                         |
| Serialization| joblib                                |
| Frontend     | HTML5, CSS3, Vanilla JavaScript       |
| Charts       | Chart.js (CDN)                        |

---

## Dataset

- **File:** `CarPrice_Assignment.csv`
- **Records:** 205 automobiles
- **Features:** 26 columns (24 predictive features + car_ID + price)
- **Target:** `price` (USD)
- **Manufacturers:** alfa-romero, audi, bmw, buick, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercury, mitsubishi, nissan, peugeot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo

---

## Machine Learning Methodology

### Feature Engineering
- `car_ID` dropped (identifier, not predictive)
- `CarName` → `manufacturer` extracted from the first word, with known typos corrected

### Numerical Features (14)
symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg

### Categorical Features (10)
manufacturer, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem

### Preprocessing Pipeline
- Numerical: `SimpleImputer(median)` → `StandardScaler`
- Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
- Fitted **only** on training data (80/20 split)

### Models Compared
1. Linear Regression
2. Ridge Regression (α=10)
3. Random Forest Regressor (200 estimators)
4. Gradient Boosting Regressor (200 estimators)
5. Extra Trees Regressor (200 estimators)
6. HistGradientBoostingRegressor (200 iterations)

### Selection Criteria
Best model selected by highest test R². Cross-validation (5-fold) is also reported.

---

## Installation

### 1. Clone / download the project

```bash
cd car-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Train the Model

```bash
python train_model.py
```

This will:
1. Load `CarPrice_Assignment.csv`
2. Clean and engineer features
3. Split data 80/20
4. Train 6 regression models with cross-validation
5. Print a comparison table
6. Save the best pipeline to `models/car_price_model.pkl`
7. Save metrics + dropdown values to `models/model_metrics.json`

Expected output:
```
============================================================
  Car Price Prediction — Model Training
============================================================

[1] Loading dataset from: ...
    Records : 205
    Features: 24
    Target  : price

[2] Train size: 164  |  Test size: 41

[3] Training and evaluating models ...

  Linear Regression           R²=0.8xxx  MAE=  x,xxx.xx  RMSE=  x,xxx.xx
  Ridge Regression            R²=0.8xxx  ...
  Random Forest               R²=0.9xxx  ...
  Gradient Boosting           R²=0.9xxx  ...
  Extra Trees                 R²=0.9xxx  ...
  HistGradientBoosting        R²=0.9xxx  ...

[4] Best model: ...  (R²=0.9xxx)
[5] Saving model  → models/car_price_model.pkl
[5] Saving metrics → models/model_metrics.json

✓ Training complete.
============================================================
```

---

## How to Run the Application

```bash
python app.py
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## API Endpoint

### POST `/predict`

**Request:**
```json
{
  "symboling": 1,
  "fueltype": "gas",
  "aspiration": "std",
  "doornumber": "four",
  "carbody": "sedan",
  "drivewheel": "fwd",
  "enginelocation": "front",
  "wheelbase": 95.0,
  "carlength": 175.0,
  "carwidth": 66.0,
  "carheight": 54.0,
  "curbweight": 2500,
  "enginetype": "ohc",
  "cylindernumber": "four",
  "enginesize": 120,
  "fuelsystem": "mpfi",
  "boreratio": 3.2,
  "stroke": 3.0,
  "compressionratio": 9.0,
  "horsepower": 100,
  "peakrpm": 5000,
  "citympg": 25,
  "highwaympg": 30,
  "manufacturer": "toyota"
}
```

**Response:**
```json
{
  "success": true,
  "predicted_price": 12345.67,
  "model_used": "Random Forest",
  "model_r2": 0.9523,
  "currency": "USD"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Missing required field: horsepower"
}
```

---

## Project Structure

```
car-price-prediction/
│
├── app.py                    # Flask web application
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── CarPrice_Assignment.csv   # Source dataset
│
├── models/
│   ├── car_price_model.pkl   # Saved best pipeline (generated)
│   └── model_metrics.json    # Metrics + dropdown data (generated)
│
├── templates/
│   ├── index.html            # Home page
│   ├── predict.html          # Prediction form
│   ├── insights.html         # Model insights dashboard
│   └── about.html            # About page
│
├── static/
│   ├── css/style.css         # Stylesheet
│   ├── js/script.js          # Global JS (nav, prediction)
│   └── js/insights.js        # Chart.js charts for insights page
│
└── data/                     # Data directory (copy of CSV)
```

---

## Example Prediction

**Input:** Toyota sedan, gas, standard aspiration, 4 doors, FWD, front engine,
120cc engine, OHC, 4 cylinders, MPFI fuel system, 100hp, 5000 RPM,
wheelbase 95", length 175", width 66", height 54", curb weight 2500 lbs,
25 city / 30 highway MPG, symboling 1.

**Result:** ~$12,000–$15,000 (varies by selected model)

---

## Limitations

- Trained on 205 records — limited dataset size
- Prices are from an older automotive dataset and may not reflect current market values
- Does not account for: vehicle condition, mileage, regional pricing, taxes, or dealer markup
- Predictions are estimates — not guaranteed market prices
- Extrapolation outside training distribution may be inaccurate

---

## License

This project is provided for educational purposes.
