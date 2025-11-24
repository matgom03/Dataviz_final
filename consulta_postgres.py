import pandas as pd
from sqlalchemy import create_engine, text

# =====================================
# 1. CONEXIÓN A POSTGRESQL
# =====================================

engine = create_engine("postgresql+psycopg2://postgres:Matcrack0800@postgres:5432/dataviz_db")

print("Conectado correctamente a PostgreSQL.")


# Función auxiliar para ejecutar consultas y mostrar resultados
def run_query(query, title=None, limit=10):
    if title:
        print(f"\n===== {title} =====")

    with engine.connect() as connection:
        df = pd.read_sql_query(text(query), connection)

    print(df.head(limit))
    return df


# =====================================
# 2. CONSULTAS SQL SOBRE adult_csv
# =====================================
# ----- 2.1 Conteo total -----
run_query(
    "SELECT COUNT(*) AS total_registros FROM adult_csv;",
    "Total de registros"
)

# ----- 2.2 Distribución de income -----
run_query(
    """
    SELECT "Income", COUNT(*) AS total
    FROM adult_csv
    GROUP BY "Income";
    """,
    "Distribución de Income"
)

# ----- 2.3 Valores únicos por columna -----
run_query(
    """SELECT DISTINCT "Workclass" FROM adult_csv;""",
    """ Valores únicos de Workclass """
)
# ----- 2.4 Horas trabajadas vs probabilidad ingreso alto -----
run_query(
    """
    SELECT "Hours-per-week",
           AVG("Income") AS high_income_rate
    FROM adult_csv
    GROUP BY "Hours-per-week"
    ORDER BY "Hours-per-week";
    """,
    "Horas por semana vs probabilidad de ingreso alto"
)

# ----- 2.5 Ingreso por ocupación -----
run_query(
    """
    SELECT "Occupation",
           AVG("Income") AS pct_above_50k,
           COUNT(*) AS total
    FROM adult_csv
    GROUP BY "Occupation"
    ORDER BY pct_above_50k DESC;
    """,
    "Ingreso promedio por ocupación"
)

# ----- 2.6 Ranking de países según ingreso alto -----
run_query(
    """
    SELECT "Native-country",
           AVG("Income") AS high_income_rate,
           RANK() OVER (ORDER BY AVG("Income") DESC) AS rank
    FROM adult_csv
    GROUP BY "Native-country";
    """,
    "Ranking de países según ingreso alto"
)

# ----- 2.7 Correlación aproximada SQL -----
run_query(
    """
    SELECT corr("Age", "Hours-per-week") AS corr_age_hours
    FROM adult_csv;
    """,
    "Correlación entre edad y horas por semana"
)


print("\nTodas las consultas ejecutadas correctamente.")
