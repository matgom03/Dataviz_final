import os
import pandas as pd 
from sqlalchemy import create_engine

def load():
    df1 = pd.read_csv("adult.data", header=None, na_values=["?", " ?"])
    df2 = pd.read_csv("adult.test", header=None, na_values=["?", " ?"], skiprows=1)
    df = pd.concat([df1, df2], ignore_index=True)

    cols = [
        "Age", "Workclass", "Fnlwgt", "Education", "Education-num",
        "Marital-status", "Occupation", "Relationship", "Race", "Sex",
        "Capital-gain", "Capital-loss", "Hours-per-week", "Native-country", "Income"
    ]

    df1.columns = cols
    df2.columns = cols
    df.columns = cols

    return df1, df2, df


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL no está definida para cargar datos.")

engine = create_engine(DATABASE_URL)

df1, df2, df = load()

df1.to_sql("adult_train", engine, if_exists="replace", index=False)
df2.to_sql("adult_test", engine, if_exists="replace", index=False)
df.to_sql("adult_completo", engine, if_exists="replace", index=False)
