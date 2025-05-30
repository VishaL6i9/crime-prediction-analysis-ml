# Crime Prediction Analysis with Machine Learning

## Project Overview

This project aims to develop a machine learning model capable of predicting the occurrence of crime based on various input features. The primary objective is to provide insights into factors influencing crime and potentially assist in proactive measures or resource allocation by identifying conditions under which crime is likely to happen.

## Features

* **Data Preprocessing:** Handling of raw crime data, including cleaning, feature engineering, and preparing the dataset for machine learning models.
* **Exploratory Data Analysis (EDA):** Visualizations and statistical summaries to understand crime patterns, distributions, and potential correlations.
* **Machine Learning Model Training:** Development and training of a robust classification model to predict crime occurrence.
* **Model Persistence:** Saving the trained model and necessary preprocessing components (like encoders) for future use and deployment.
* **Model Evaluation:** Comprehensive assessment of model performance using metrics such as classification reports, confusion matrices, and other relevant evaluation plots.
* **Web Application (Flask):** A lightweight web interface to demonstrate the crime prediction model in action.

## Dataset

The dataset used for this project is a trimmed version sourced from Kaggle: [Crime Prediction (95% Accuracy)](https://www.kaggle.com/code/abdelrahmanemad594/crime-prediction-95-accuracy/input).

The specific dataset file used in this project is `output.csv`, which has been preprocessed and reduced in size to optimize training on local machines. It contains various features relevant to crime incidents, which are used as inputs for the prediction model.

## Technologies Used

This project leverages the following technologies and Python libraries:

* **Python:** Programming language
* **Jupyter Notebook / Google Colab:** For interactive development, experimentation, and analysis.
* **Pandas:** For efficient data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib:** For static data visualization.
* **Seaborn:** For enhanced statistical data visualization.
* **Scikit-learn:** For machine learning model building, evaluation, and preprocessing utilities (e.g., for `encoders.pkl`).
* **XGBoost:** For implementing the Extreme Gradient Boosting algorithm, which is a powerful and efficient ensemble method.
* **Joblib:** For saving and loading Python objects, specifically the trained machine learning model (`model.pkl`) and encoders (`encoders.pkl`).
* **Flask:** For building the lightweight web application to serve predictions.
* **Cython:** (Potentially used for performance optimization in some parts of the project, if integrated).

## Project Structure

The project is organized into the following directories and files:

```
.
├── .venv/                      # Python virtual environment
├── templates/                  # HTML templates for the Flask application
│   ├── index.html              # Main page for the web application
│   └── index3.html             # Another HTML page for the web application
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── app.py                      # Main Flask application to serve predictions
├── app2.py                     # (Potentially an alternative or older version of the Flask app)
├── crime-prediction.ipynb      # Jupyter Notebook containing data analysis, model training, and evaluation
├── encoders.pkl                # Pickled object containing pre-trained data encoders (e.g., LabelEncoder, OneHotEncoder)
├── model.pkl                   # The trained machine learning model saved using joblib
├── output.csv                  # The preprocessed and trimmed dataset used for training and prediction
└── requirements.txt            # List of Python dependencies
```

## Machine Learning Models Explored

The `crime-prediction.ipynb` notebook includes experimentation and evaluation, primarily focusing on:

* **Random Forest Classifier:** An ensemble tree-based method known for its robustness and ability to handle various data types.
* **Extreme Gradient Boosting (XGBoost) Classifier:** A highly efficient and effective gradient boosting framework, likely identified as the best-fit model for this prediction task and saved as `model.pkl`.

For the models explored, the notebook provides:

* Model fitting and training.
* Hyperparameter tuning (if applicable).
* Generation of a `classification_report`.
* Calculation of various matrices (e.g., confusion matrix).
* Other relevant evaluation metrics and plots.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/VishaL6i9/crime-prediction-analysis-ml.git](https://github.com/VishaL6i9/crime-prediction-analysis-ml.git)
    cd crime-prediction-analysis-ml
    ```
    
2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    * **On Windows:**

        ```bash
        .venv\Scripts\activate
        ```

    * **On macOS/Linux:**

        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required dependencies:**

    The `requirements.txt` file specifies the exact versions of the libraries used. Install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Jupyter Notebook

To explore the data analysis, model training, and evaluation process, open the Jupyter Notebook:

```bash
jupyter notebook crime-prediction.ipynb
```

### Running the Flask Web Application

To run the Flask application and make predictions via a web interface:

1.  Ensure your virtual environment is activated.

2.  Navigate to the project's root directory.

3.  Run the `app.py` script:

    ```bash
    python app.py
    ```

4.  Open your web browser and go to `http://127.0.0.1:5000/` (or the address shown in your terminal) to access the application.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or suggestions, feel free to reach out:

* **Vishal Kandakatla —** [vishalkandakatla@gmail.com]
