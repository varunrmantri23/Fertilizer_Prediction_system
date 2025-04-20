# Fertilizer Recommendation System 🌾🌱

## Description

This project provides an intelligent recommendation system for agricultural fertilizers. Based on key soil characteristics, environmental conditions, and crop type, the system predicts the most suitable fertilizer to optimize crop yield and health. The prediction model is built using machine learning (Random Forest) and deployed as an interactive web application using Streamlit.

The application allows users to input parameters like soil type, crop type, temperature, humidity, nitrogen, phosphorus, potassium, and moisture levels. It then provides a fertilizer recommendation along with a relevant image and detailed descriptions in both English and Hindi.

## Features

-   **Interactive Input:** User-friendly interface with dropdowns and number inputs for parameters.
-   **Machine Learning Model:** Utilizes a pre-trained Random Forest model for accurate predictions.
-   **Visual Output:** Displays an image of the recommended fertilizer.
-   **Multilingual Descriptions:** Provides detailed information about the recommended fertilizer in English and Hindi.
-   **Easy Deployment:** Built with Streamlit for simple web application deployment.

## Tech Stack

-   **Python:** Core programming language.
-   **Pandas:** Data manipulation and analysis.
-   **Scikit-learn:** Machine learning library (for model training and preprocessing).
-   **Joblib:** Saving and loading the trained model and encoder.
-   **Streamlit:** Building the interactive web application interface.

## Setup and Installation

1.  **Clone the Repository (if applicable):**

    ```bash
    git clone https://github.com/varunrmantri23/Fertilizer_Prediction_system
    cd Fertilizer_Prediction_system
    ```

    _(If you don't have a repository yet, just ensure all files are in one project folder)._

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _(If you don't have a `requirements.txt` file yet, create one or install manually):_

    ```bash
    pip install streamlit pandas scikit-learn joblib
    ```

4.  **Ensure Model Files are Present:**
    Make sure the following files (generated from your Jupyter notebook) are in the main project directory:

    -   `random_forest_model.joblib`
    -   `one_hot_encoder.joblib`
    -   `X_final_columns.joblib`

5.  **Prepare Images:**
    -   Create a folder named `images` in the main project directory.
    -   Place images of the fertilizers (e.g., `urea.jpg`, `dap.jpg`, `10-26-26.jpg`, `default.jpg`, etc.) inside this `images` folder. Ensure the filenames match the keys in the `fertilizer_images` dictionary in `app.py`.

## Running the Application

1.  Navigate to the project directory in your terminal.
2.  Ensure your virtual environment is activated.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application will open automatically in your default web browser.
