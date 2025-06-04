# -Rainfall-Prediction-System-using-Random-Forest-Classifier
This system leverages the Random Forest Classifier to analyze historical weather data and predict rainfall patterns with high accuracy. By integrating feature engineering, data preprocessing, and hyperparameter tuning, the model ensures reliable forecasts, aiding in agriculture, disaster management, and water resource planning. Accuracy over 75% has been achieved.

Certainly! Here is a comprehensive **README.md** for your "Rainfall Prediction Using Random Forest" project, ready to upload on GitHub. This README covers project overview, dataset, setup, usage, methodology, results, and more.

---

# Rainfall Prediction Using Random Forest

Predicting rainfall is crucial for agriculture, disaster management, and daily planning. This project uses a Random Forest Classifier to predict whether it will rain on a given day based on meteorological data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Model Saving & Loading](#model-saving--loading)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project leverages machine learning (Random Forest Classifier) to predict rainfall based on weather features such as temperature, humidity, pressure, wind, and more. The model is trained and evaluated using a real-world dataset, with performance metrics and visualizations included.

---

## Dataset

- **Source:** [Rainfall.csv] (local file, not included due to size/privacy)
- **Rows:** 366 (one year of daily data)
- **Columns:** 12 meteorological features + target
- **Target:** `rainfall` (yes/no, later encoded as 1/0)

### Sample Data

| pressure | maxtemp | temparature | mintemp | dewpoint | humidity | cloud | rainfall | sunshine | winddirection | windspeed |
|----------|---------|-------------|---------|----------|----------|-------|----------|----------|---------------|-----------|
| 1025.9   | 19.9    | 18.3        | 16.8    | 13.1     | 72       | 49    | yes      | 9.3      | 80.0          | 26.3      |
| 1022.0   | 21.7    | 18.9        | 17.2    | 15.6     | 81       | 83    | yes      | 0.6      | 50.0          | 15.3      |
| ...      | ...     | ...         | ...     | ...      | ...      | ...   | ...      | ...      | ...           | ...       |

---

## Features

- **pressure:** Atmospheric pressure (hPa)
- **maxtemp:** Maximum temperature (°C)
- **temparature:** Average temperature (°C)
- **mintemp:** Minimum temperature (°C)
- **dewpoint:** Dew point (°C)
- **humidity:** Relative humidity (%)
- **cloud:** Cloud cover (%)
- **rainfall:** Rainfall (yes/no, target variable)
- **sunshine:** Sunshine hours
- **winddirection:** Wind direction (degrees)
- **windspeed:** Wind speed (km/h)

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rainfall-prediction-randomforest.git
cd rainfall-prediction-randomforest
```

### 2. Install Dependencies

Make sure you have Python 3.7+ and pip installed.

```bash
pip install -r requirements.txt
```

**requirements.txt** should include:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### 3. Prepare the Dataset

Place your `Rainfall.csv` file in the project directory.

---

## Usage

You can run the notebook directly:

```bash
jupyter notebook RainfallPredictionUsingRandomForest.ipynb
```

Or run as a script (if you convert the notebook):

```bash
python RainfallPredictionUsingRandomForest.py
```

---

## Methodology

1. **Data Loading & Cleaning**
   - Load CSV data using pandas.
   - Handle missing values and encode categorical variables (e.g., 'yes'/'no' to 1/0).

2. **Exploratory Data Analysis**
   - Visualize feature distributions and correlations using matplotlib/seaborn.

3. **Feature Engineering**
   - Select relevant features for model input.

4. **Train-Test Split**
   - Split data into training and testing sets (e.g., 80/20).

5. **Model Building**
   - Use `RandomForestClassifier` from scikit-learn.
   - Hyperparameter tuning with `GridSearchCV`.

6. **Model Evaluation**
   - Evaluate using accuracy, confusion matrix, and classification report.

7. **Model Saving**
   - Save the trained model using pickle for future use.

---

## Results

- **Best Model:** Random Forest Classifier with tuned hyperparameters.
- **Evaluation Metrics:**
  - Accuracy: *e.g., 0.85*
  - Precision, Recall, F1-score: See notebook for details.
- **Confusion Matrix:**  
  ![Confusion Matrix](assets/confusion_matrix image)*

---

## Model Saving & Loading

```python
# Saving the model
import pickle
with open('rainfall_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Loading the model
with open('rainfall_rf_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

**Note:**  
If you use this project or its ideas, please cite this repository and give credit to the contributors.

---

