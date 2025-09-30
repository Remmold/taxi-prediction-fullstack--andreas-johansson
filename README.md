# 🚕 Taxi Fare Predictor

> **Note:** This is a school project. The dataset used is synthetic and should not be used to predict real taxi prices.

This guide will walk you through setting up and running the project on your local machine.

---

### 🔧 Step 1: Project Setup

First, you'll clone the project, create a virtual environment, and install the required packages.

1.  **Clone the Project**
    Open your terminal and run the following command to download the project files:
    ```bash
    git clone [https://github.com/your-username/taxi-prediction-fullstack--andreas-johansson.git](https://github.com/your-username/taxi-prediction-fullstack--andreas-johansson.git)
    cd taxi-prediction-fullstack--andreas-johansson
    ```

2.  **Create Environment and Install Packages**
    This project uses `uv` to manage the Python environment and dependencies. Run the following command. It will automatically create a `.venv` folder and install all the necessary packages.
    ```bash
    uv sync
    ```

3.  **Configure Your Environment File**
    You need to add your Google Maps API key to run the application.
    -   In the project's root folder, find the file named `.env.example`.
    -   Rename it to `.env`.
    -   Open the `.env` file with a text editor. You will see a placeholder for the API key:
        ```
        GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
        ```
    -   Replace `YOUR_API_KEY_HERE` with your actual Google Maps API key.

---

### 🚀 Step 2: Running the Application

To run the application, you will need to open **two separate terminal windows**: one for the backend server and one for the frontend application.

#### **🖥️ Terminal 1: Start the Backend API**

1.  **Activate the Virtual Environment**
    -   **Windows (PowerShell):**
        ```powershell
        .venv\scripts\activate
        ```
    -   **macOS / Linux (Bash):**
        ```bash
        source .venv/bin/activate
        ```

2.  **Run the Uvicorn Server**
    Once the environment is active, start the FastAPI server. The `--reload` flag will automatically restart the server when you make code changes.
    ```bash
    uvicorn src.taxipred.backend.api:app --reload
    ```
    Keep this terminal window running.

#### **🖥️ Terminal 2: Start the Frontend App**

1.  **Activate the Virtual Environment in the New Terminal**
    -   **Windows (PowerShell):**
        ```powershell
        .venv\scripts\activate
        ```
    -   **macOS / Linux (Bash):**
        ```bash
        source .venv/bin/activate
        ```

2.  **Run the Streamlit App**
    Once the environment is active, start the Streamlit frontend.
    -   **Windows (PowerShell):**
        ```powershell
        streamlit run .\src\taxipred\frontend\app.py
        ```
    -   **macOS / Linux (Bash):**
        ```bash
        streamlit run src/taxipred/frontend/app.py
        ```

Your browser should automatically open with the taxi prediction application running. Enjoy!


