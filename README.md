# taxi-prediction-fullstack--andreas-johansson
note that this is a school project and the dataset used in synthetic data and should not be used to predict real taxiprices

First Clone the project

Then run uv sync
```powershell
uv sync
```
This will create a virtual environment aswell as instlal the required python packages

then you remove the .example from your <img width="149" height="23" alt="image" src="https://github.com/user-attachments/assets/fac8794c-954b-41c2-869a-3276db1d832e" />

so you are left with <img width="85" height="24" alt="image" src="https://github.com/user-attachments/assets/6682fe03-8abb-4456-9fce-ce3440c64d01" />
add your Google_maps_api key instead of placeholder

<img width="595" height="46" alt="image" src="https://github.com/user-attachments/assets/83d5d7aa-3457-408f-b6e8-59a7b354ebc9" />

it should look something like this afterward

<img width="669" height="46" alt="image" src="https://github.com/user-attachments/assets/97ab07dc-e1c9-4421-86ba-420645609502" />

Activate virtual environmentw

```powershell
.venv/scripts/activate
```
Start Uvicorn servers for fastAPI access
From project root folder
```powershell
uvicorn src.taxipred.backend.api:app
```

Open a second terminal window
Activate virtual environment for second terminal

```powershell
.venv/scripts/activate
```

Start Streamlit Frontend

```powershell
streamlit run .\src\taxipred\frontend\app.py
```


