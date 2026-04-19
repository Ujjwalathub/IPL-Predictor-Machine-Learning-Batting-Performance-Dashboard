# IPL-Predictor: Batting Performance Analysis & Prediction

IPL-Predictor is an end-to-end machine learning solution designed to forecast the number of runs a cricket player will score in an IPL match. The system processes historical IPL data, trains a sophisticated regression model, and serves predictions through an interactive web dashboard.

## 🚀 Features

* **Automated ETL Pipeline**: Merges disparate CSV files into a master dataset while extracting metadata like Year and Metric via Regex.
* **Predictive Modeling**: Utilizes a `HistGradientBoostingRegressor` to handle categorical data and deliver accurate run predictions.
* **Interactive Dashboard**: A Flask-based web interface for users to input match parameters and receive instant predictions.
* **Model Insights**: Visualizes feature importance (Player vs. Venue vs. Opponent) and actual vs. predicted performance.
* **Live API**: Provides RESTful endpoints for retrieving available players, venues, and model statistics.

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Data Analysis**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (HistGradientBoosting, LabelEncoding)
* **Backend**: Flask, Flask-CORS
* **Visualization**: Matplotlib, Seaborn

## 📂 Project Structure

* `app.py`: The main Flask application containing the prediction API and web server logic.
* `ipl.py`: The core modeling script for training, evaluating, and visualizing model performance.
* `merge_ipl_dataset.py`: ETL script that cleans and consolidates raw IPL performance data.
* `Frontend/`: Contains the HTML templates (`Dashboard-updated.html`, `code-updated.html`) and static assets.

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/yourusername/IPL-Predictor.git](https://github.com/yourusername/IPL-Predictor.git)
   cd IPL-PredictorInstall Dependencies:

Bash
pip install -r requirements.txt
Data Preparation:
Update the DATASET_PATH in merge_ipl_dataset.py to point to your raw CSV files, then run:

Bash
python merge_ipl_dataset.py
Launch the Application:
Run the Flask server:

Bash
python app.py
The application will automatically open in your default browser at http://localhost:5000.

📊 Model Performance
The model focuses on high-impact batting metrics such as "Most Fours," "Most Sixes," and "Fastest Centuries".

Algorithm: HistGradientBoostingRegressor.

Evaluation: The system tracks Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to ensure prediction reliability.

Primary Features: Player Identity, Opponent Team, Venue, and Year.

📝 License
Distributed under the MIT License.
