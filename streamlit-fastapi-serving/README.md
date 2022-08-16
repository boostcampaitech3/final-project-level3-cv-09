# FastAPI(BackEnd) <-> Streamlit(FrontEnd) Serving
 Code composed by transferring most of the operation logic to the backend
  

## How to run

1. install requirements.txt


2. FastAPI(BackEnd) run
```
uvicorn server:app --host 0.0.0.0 --port 8000
```

3. Streamlit(FrontEnd) run
```
streamlit run ui.py
```