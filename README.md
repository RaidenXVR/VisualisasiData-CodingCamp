# Analisis Perentalan Sepeda
Analisis ini berdasarkan data [Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) oleh Hadi Fanaee-T menggunakan diagram dan prediksi menggunakan regresi linear.

## Environment Setup - Anaconda
```
conda create --name python-env python=3.11
conda activate  python-env
pip install -r requirements.txt
```

## Environment Setup - Python Virtual Env
```bash
python -m venv venv
# assuming the current directory is where the requirements.txt is.
./venv/scripts/activate 
pip install -r requirements.txt
```

## Run Streamlit App
```
streamlit run dashboard/dashboard.py
```