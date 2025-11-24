import dash
from dash import dcc, html,dash_table,Input,Output,State,no_update
import dash_bootstrap_components as dbc
from utils import *
import plotly.express as px
from utils import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from sklearn.base import BaseEstimator, TransformerMixin
class MultiHotDeckImputer(BaseEstimator, TransformerMixin):
    def __init__(self, imputations, random_state=0):
        self.imputations = imputations
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        rng = np.random.default_rng(self.random_state)
        for col, group in self.imputations:
            out = X[col].copy()
            for g, sub in X.groupby(group):
                pool = sub[col].dropna().to_numpy()
                idx = sub.index[sub[col].isna()]
                if pool.size > 0 and len(idx) > 0:
                    out.loc[idx] = rng.choice(pool, size=len(idx), replace=True)
            X[col] = out
        return X
df1_head, df2_head, df = load_data()
df["Education-num"] = df["Education-num"].astype("category")
df["Income"] = df["Income"].apply(lambda x: 1 if str(x).strip().startswith(">50K") else 0)
df["Income"] = df["Income"].astype("category")
cat=df.select_dtypes(include=['object','category']).columns.drop(['Income','Education-num'])
num = df.select_dtypes(include=[np.number]).columns
duplicadas = df[df.duplicated()]
df_importancias = pd.read_csv("feature_importances.csv")
resultados = np.load("modelo/results_rf.npz")
cm = resultados["cm"]
fpr, tpr, roc_auc = resultados["fpr"], resultados["tpr"], resultados["roc_auc"]
cm_fig = plot_confusion_matrix(cm)
roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True, external_scripts=[
        "https://polyfill.io/v3/polyfill.min.js?features=es6",
        {
            "src": "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js",
            "id": "mathjax-script",
            "defer": True
        }
    ]
)
app.title = "EDA Adult Dataset"
server = app.server
data = [
    {"Clase": "0", "Precisión": 0.93, "Recall": 0.86, "F1-score": 0.89, "Soporte": 12435},
    {"Clase": "1", "Precisión": 0.63, "Recall": 0.78, "F1-score": 0.70, "Soporte": 3846},
    {"Clase": "accuracy", "Precisión": "", "Recall": "", "F1-score": 0.84, "Soporte": 16281},
    {"Clase": "macro avg", "Precisión": 0.78, "Recall": 0.82, "F1-score": 0.79, "Soporte": 16281},
    {"Clase": "weighted avg", "Precisión": 0.86, "Recall": 0.84, "F1-score": 0.84, "Soporte": 16281},
]
pipeline_list = dbc.ListGroup([
    dbc.ListGroupItem("Pipeline", color="primary"),

    dbc.ListGroupItem(
        html.Ul([
            html.Li([
                "preprocesador (ColumnTransformer)",
                html.Ul([
                    html.Li([
                        "num",
                        html.Ul([
                            html.Li("StandardScaler()")
                        ])
                    ]),
                    html.Li([
                        "cat",
                        html.Ul([
                            html.Li("MultiHotDeckImputer"),
                            html.Li("SimpleImputer(strategy='most_frequent')"),
                            html.Li("OneHotEncoder(drop='first')")
                        ])
                    ])
                ])
            ]),

            html.Li("SMOTE(random_state=0)"),
            html.Li("RandomForestClassifier()")
        ])
    )
])

subtabs_analisis = html.Div([
    dcc.Tabs(id='subtabs_eda', value='univariado', children=[
        dcc.Tab(label='1. Análisis Univariado', value='univariado'),
        dcc.Tab(label='2. Análisis Bivariado', value='bivariado'),
        dcc.Tab(label='3. Correlaciones y Colinealidad', value='correlaciones'),
        dcc.Tab(label='4. Visualizaciones del Modelo', value='modelo'),
        dcc.Tab(label='5. Indicadores del Modelo', value='indicadores')
    ]),
    html.Div(id='contenido_subtab_eda')
])
# ==========================================================
# CALLBACKS DE LAS SUBTABS DE ANÁLISIS (EDA)
# ==========================================================

@app.callback(
    Output('contenido_subtab_eda', 'children'),
    Input('subtabs_eda', 'value')
)
def render_subtab(subtab_value):
    # -----------------------------
    # 1 Análisis Univariado
    # -----------------------------
    if subtab_value == 'univariado':
        return html.Div([
            html.H3("Análisis Univariado del Dataset"),
            html.Div([
                html.H5("Selecciona el tipo de análisis:"),
                dcc.RadioItems(
                    id='opcion-analisis-univariado',
                    options=[
                        {'label': 'Resumen General', 'value': 'resumen'},
                        {'label': 'Variables Categóricas', 'value': 'categoricas'},
                        {'label': 'Variables Numéricas', 'value': 'numericas'},
                        {'label': 'Variable Individual', 'value': 'individual'}
                    ],
                    value='resumen',
                    inline=True,
                    labelStyle={'marginRight': '10px'}
                )
            ], style={'margin': '20px 0'}),
            html.Div(id='contenido-analisis-univariado')
        ])

    # -----------------------------
    # 2 Análisis Bivariado
    # -----------------------------
    elif subtab_value == 'bivariado':
        return html.Div([
            html.H3("Análisis Bivariado del Dataset"),

            # ================================
            # 🔹 Sección de Prueba de Normalidad
            # ================================
            html.H4("Prueba de Normalidad - Kolmogorov-Smirnov"),
            html.P("""
                Antes de realizar el análisis bivariado, se evalúa la normalidad de las variables numéricas 
                usando la prueba de Kolmogorov-Smirnov (K-S Test). Esto permite decidir entre el uso de ANOVA 
                o pruebas no paramétricas como Kruskal-Wallis.
            """),
            dbc.Button("Ejecutar prueba de normalidad", id="btn-prueba-normalidad", color="secondary", className="mb-3"),
            html.Div(id="resultado-prueba-normalidad"),
            html.Hr(),

            # ================================
            # 🔹 Sección de Análisis Bivariado
            # ================================
            html.H5("Selecciona el tipo de análisis:"),
            dcc.RadioItems(
                id='tipo_bivariado',
                options=[
                    {'label': 'Numérico vs Numérico', 'value': 'num_num'},
                    {'label': 'Categórico vs Categórico', 'value': 'cat_cat'},
                    {'label': 'Categórico vs Numérico', 'value': 'cat_num'},
                ],
                value='num_num',
                inline=True
            ),
            html.Br(),
            dbc.Button("Ejecutar análisis", id="btn_bivariado", color="primary", className="mb-3"),
            html.Div(id="salida_bivariado")
        ])


    # -----------------------------
    # 3 Correlaciones y Colinealidad
    # -----------------------------
    elif subtab_value == 'correlaciones':
        resultados_corr = analizar_colinealidad_y_correlaciones(df)
        corr_vif = resultados_corr["resultados"]["vif"]
        figs = resultados_corr["figuras"]  # antes era "imagenes"

        return html.Div([
            html.H3("Correlaciones y Colinealidad"),
            html.Hr(),

            # --------------------------
            # Tabla VIF
            # --------------------------
            html.Br(),
            html.H4("Factor de Inflación de Varianza (VIF)"),
            dash_table.DataTable(
                data=corr_vif.to_dict('records'),
                columns=[{"name": c, "id": c} for c in corr_vif.columns],
                style_table={'overflowX': 'auto', 'width': '80%', 'margin': 'auto'},
                style_header={
                    'backgroundColor': '#003366',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '8px',
                    'fontFamily': 'Arial',
                    'fontSize': 15
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{VIF} > 10'},
                        'backgroundColor': '#F8D7DA',
                        'color': 'black'
                    },
                    {
                        'if': {'filter_query': '{Colinealidad} = "Alta"'},
                        'backgroundColor': '#FADBD8'
                    }
                ]
            ),

            # --------------------------
            # Gráficos Plotly
            # --------------------------
            html.Br(),
            html.H4("Visualizaciones de Correlaciones"),
            html.Div(
                [
                    dcc.Graph(figure=fig, style={"width": "48%", "display": "inline-block", "margin": "10px"})
                    for fig in figs
                ],
                style={'textAlign': 'center'}
            )
        
        ])
    # ================================
    # Subtab: Visualizaciones del Modelo
    # ================================
    elif subtab_value == 'modelo':
        return html.Div([
            html.H3("Visualizaciones del Modelo - Random Forest"),
            html.Hr(),

            # 🔹 Fila 1: Matriz de Confusión y Curva ROC
            html.Div([
                html.Div([
                    html.H5("Matriz de Confusión"),
                    dcc.Graph(figure=cm_fig, style={"height": "400px"})
                ], style={"width": "48%", "padding": "10px"}),

                html.Div([
                    html.H5("Curva ROC"),
                    dcc.Graph(figure=roc_fig, style={"height": "400px"})
                ], style={"width": "48%", "padding": "10px"})
            ], style={"display": "flex", "justifyContent": "space-between", "flexWrap": "wrap"}),

            html.Hr(),

            # 🔹 Fila 2: Importancia de Características
            html.Div([
                html.H5("Top 20 Feature Importances - Random Forest"),
                dcc.Graph(
                    figure=px.bar(
                        pd.read_csv("feature_importances.csv").head(20),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top 20 Feature Importances - Random Forest"
                    ).update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=500,
                        margin=dict(l=80, r=20, t=60, b=40)
                    )
                )
            ], style={"width": "100%", "padding": "10px"})
        ])

    # ================================
    # Subtab: Indicadores del Modelo
    # ================================
    elif subtab_value == 'indicadores':
        # Crear DataFrame a partir de las métricas definidas
        df_metrics = pd.DataFrame(data)

        return html.Div([
            html.H3("Indicadores Globales del Modelo - Random Forest"),
            html.Hr(),

            html.H4("Reporte de Clasificación"),
            dash_table.DataTable(
                data=df_metrics.to_dict('records'),
                columns=[{"name": c, "id": c} for c in df_metrics.columns],
                style_table={'width': '80%', 'margin': 'auto'},
                style_header={
                    'backgroundColor': '#003366',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'fontSize': 15
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'},
                    {'if': {'filter_query': '{F1-score} >= 0.85'},
                    'backgroundColor': '#D4EDDA'}
                ]
            ),

            html.Br(),
            html.P(
                "Estos indicadores reflejan el desempeño global del modelo Random Forest "
                "entrenado sobre el dataset Adult. Se observa un buen equilibrio general, "
                "con mayor desempeño en la clase 0 (ingresos ≤ 50K) y mejora del recall en la clase 1 "
                "tras aplicar SMOTE.",
                style={"textAlign": "center", "maxWidth": "80%", "margin": "auto"}
            )
    ])
    return html.Div()

@app.callback(
    Output('contenido-analisis-univariado', 'children'),
    Input('opcion-analisis-univariado', 'value')
)
def mostrar_analisis_univariado(opcion):
    if opcion == 'resumen':
        resumen_cat = analizar_categoricas(df)
        resumen_num = resumen_numericas(df)

        return html.Div([
            html.H4("Resumen General del Dataset"),
            html.H5("Variables Categóricas"),
            dash_table.DataTable(
                data=resumen_cat.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen_cat.columns],
                page_size=10,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H5("Variables Numéricas"),
            dash_table.DataTable(
                data=resumen_num.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen_num.columns],
                page_size=10,
                style_table={"overflowX": "auto"}
            )
        ])

    elif opcion == 'categoricas':
        resumen_cat = analizar_categoricas(df)
        graficos_barra = graficar_categoricas(df, tipo="barra")

        return html.Div([
            html.H4("Análisis de Variables Categóricas"),
            dash_table.DataTable(
                data=resumen_cat.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen_cat.columns],
                page_size=10,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H5("Gráficos de Frecuencia"),
            html.Div([html.Img(src=img, style={"width": "40%", "margin": "10px"}) for img in graficos_barra])
        ])

    elif opcion == 'numericas':
        resumen_num = resumen_numericas(df)
        graficos_num = graficar_numericas(df)

        return html.Div([
            html.H4("Análisis de Variables Numéricas"),
            dash_table.DataTable(
                data=resumen_num.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen_num.columns],
                page_size=10,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H5("Distribución Numérica"),
            html.Div([html.Img(src=img, style={"width": "45%", "margin": "10px"}) for img in graficos_num])
        ])

    elif opcion == 'individual':
        return html.Div([
            html.H4("Análisis de Variable Individual"),
            dcc.Dropdown(
                id='variable-individual',
                options=[{'label': c, 'value': c} for c in df.columns],
                value=df.columns[0],
                style={'width': '50%'}
            ),
            html.Div(id='analisis-variable-univariado')
        ])

    return html.Div("Selecciona una opción de análisis.")
@app.callback(
    Output('analisis-variable-univariado', 'children'),
    Input('variable-individual', 'value')
)
def analizar_variable_individual(columna):
    if columna is None:
        return html.Div("Selecciona una variable para analizar.")

    if pd.api.types.is_numeric_dtype(df[columna]):
        resumen = df[[columna]].describe().T
        mediana = df[columna].median()
        varianza = df[columna].var()

        fig_hist = px.histogram(
            df, x=columna, nbins=20, title=f"Histograma de {columna}",
            marginal="box", labels={columna: columna}
        )

        return html.Div([
            html.H5(f"Variable seleccionada: {columna}"),
            dash_table.DataTable(
                data=resumen.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen.columns]
            ),
            html.P(f"Mediana: {mediana:.2f}"),
            html.P(f"Varianza: {varianza:.2f}"),
            dcc.Graph(figure=fig_hist)
        ])

    else:
        conteo = df[columna].value_counts(dropna=False).reset_index()
        conteo.columns = [columna, "Frecuencia"]

        fig_bar = px.bar(
            conteo, x=columna, y="Frecuencia",
            title=f"Frecuencia de {columna}", text="Frecuencia"
        )

        children = [
            html.H5(f"Variable seleccionada: {columna}"),
            dash_table.DataTable(
                data=conteo.to_dict('records'),
                columns=[{"name": i, "id": i} for i in conteo.columns]
            ),
            dcc.Graph(figure=fig_bar)
        ]

        
        if conteo.shape[0] <= 6:
            fig_pie = px.pie(conteo, values="Frecuencia", names=columna,
                             title=f"Distribución de {columna}")
            children.append(dcc.Graph(figure=fig_pie))
        else:
            children.append(html.P(" Demasiadas categorías para gráfico de pastel."))

        return html.Div(children)
@app.callback(
    Output("resultado-prueba-normalidad", "children"),
    Input("btn-prueba-normalidad", "n_clicks"),
    prevent_initial_call=True
)
def ejecutar_prueba_normalidad(n_clicks):
    if n_clicks == 0:
        return no_update
    resultado = prueba_normalidad(df)
    if isinstance(resultado, pd.DataFrame):
        return dash_table.DataTable(
            data=resultado.to_dict("records"),
            columns=[{"name": c, "id": c} for c in resultado.columns],
            page_size=10,
            style_table={"overflowX": "auto"}
        )
    else:
        return html.Pre(str(resultado))
@app.callback(
    Output("salida_bivariado", "children"),
    Input("btn_bivariado", "n_clicks"),
    State("tipo_bivariado", "value"),
    prevent_initial_call=True
)
def mostrar_bivariado(n_clicks, tipo):
    if tipo == 'num_num':
        resultado = analisis_bivariado_numerico(df)
    elif tipo == 'cat_cat':
        resultado = analisis_bivariado_categorico(df)
    else:
        resultado = analisis_bivariado_cat_num(df)

    if "error" in resultado:
        return html.Div(resultado["error"])

    tabla = dash_table.DataTable(
        data=resultado["tabla"].round(4).to_dict('records'),
        columns=[{"name": c, "id": c} for c in resultado["tabla"].columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
    )

    graficos = [dcc.Graph(figure=fig) for fig in resultado["imagenes"]]

    return html.Div([tabla] + graficos)

tab_introduccion = dcc.Tab(
    label='1. Introducción',
    children=[
        html.Div([
            html.H2("Introducción al Proyecto"),
            html.Hr(),

            dcc.Markdown("""
                         Este proyecto ofrece una visión general del contexto socioeconómico asociado a los niveles de ingreso en la población adulta de EE. UU., utilizando como base el dataset Adult / Census Income. 
                         A partir de este conjunto de datos, se realiza un Análisis Exploratorio de Datos (EDA) para identificar patrones, relaciones entre variables y factores que influyen en el nivel de ingresos.
                         
                         Posteriormente, se construye un modelo predictivo basado en Random Forest, una técnica de aprendizaje automático ampliamente utilizada por su capacidad para manejar datos heterogéneos y capturar relaciones no lineales entre variables. 
                         El objetivo del modelo es predecir si una persona tiene un ingreso anual superior a 50 000 dólares, al tiempo que se analiza la importancia de las características más influyentes mediante el uso de feature importance.
                         En conjunto, el proyecto busca explorar, analizar y modelar la información del censo para comprender los determinantes más relevantes del nivel de ingresos y evaluar el desempeño del modelo de clasificación seleccionado.
                         """),

            html.Img(
            src="https://www.pymnts.com/wp-content/uploads/2016/09/US-Income-On-The-Rise.jpg?w=768",
            style={
                "width": "60%",
                "margin-bottom": "10px",
            },
            ),
          html.P(
            "Fuente: PYMNTS - US Income On The Rise",
            style={
                "fontSize": "14px",
                "color": "gray",
                "fontStyle": "italic",
                "marginTop": "0px",
            },
        )
        ], 
        style={"padding": "20px"})
    ]
)
tab_contexto = dcc.Tab(
    label='2. Contexto',
    children=[
        html.Div([
            html.H2("Contexto del Dataset"),
            html.Hr(),
            dcc.Markdown("""
                         El análisis del nivel de ingresos en una población es fundamental para comprender dinámicas laborales, educativas y socioeconómicas. Factores como el nivel educativo, el tipo de empleo, la jornada laboral o la situación familiar pueden influir significativamente en la probabilidad de obtener salarios más altos. 
                         En este contexto, el dataset Adult / Census Income ofrece una base sólida para examinar estas relaciones y construir modelos que permitan caracterizar los elementos más determinantes en la distribución del ingreso.
                         """),

            html.H3("Fuente de los datos"),
            dcc.Markdown("""
                En este trabajo se analiza el dataset **Adult / Census Income**, 
                disponible en el **UCI Machine Learning Repository**.  
                Fue publicado por Barry Becker en 1996 con información del censo de EE. UU. de 1994.  

                - **Instancias:** 48,842  
                - **Atributos:** 14  
                
                **Fuente:** https://archive.ics.uci.edu/dataset/2/adult
            """),

            html.H3("Variables de interés"),
            dcc.Markdown("""
                | Variable | Tipo | Descripción / Operacionalización |
                |---------|------|----------------------------------|
                | **Age** | Numérica | Edad del individuo. |
                | **Education-num** | Numérica | Nivel educativo codificado (1-16). |
                | **Education** | Categórica | Nivel educativo textual. |
                | **Workclass** | Categórica | Tipo de sector laboral del individuo. |
                | **Occupation** | Categórica | Ocupación desempeñada. |
                | **Hours-per-week** | Numérica | Horas trabajadas a la semana. |
                | **Capital-gain** | Numérica | Ganancias de capital. |
                | **Capital-loss** | Numérica | Pérdidas de capital. |
                | **Marital-status** | Categórica | Estado civil. |
                | **Race** | Categórica | Raza. |
                | **Sex** | Categórica | Género. |
                | **Native-country** | Categórica | País de origen. |
                | **Income** | Categórica (objetivo) | {<=50K, >50K}. Ingresos |
            """),
        ],
        style={"padding": "20px"})
    ]
)
tab_problema = dbc.Tab(
    label="3. Planteamiento del Problema",
    children=[
        html.Div([
            html.H2("Planteamiento del Problema"),
            html.Hr(),

            html.P("""
                En Estados Unidos, el ingreso anual de una persona suele estar determinado por 
                factores demográficos, educativos y laborales. Sin embargo, identificar cuáles 
                de estos factores influyen más en que una persona gane más de $50.000 USD al año 
                no es trivial. Analizar estos factores de manera mas profunda es esencial para poder predecir de mejor manera los ingresos en el hogar. 
            """),

            html.P("""
                El dataset Adult permite explorar cómo se relacionan estas características 
                con los niveles de ingreso y construir un modelo que permita predecir dicho nivel. 
            """),

            html.H3("Pregunta problema"),
            dcc.Markdown("""
                **¿Qué tan eficaz es un modelo Random Forest optimizado con GridSearchCV para predecir ingresos superiores a $50.000 USD anuales a partir de variables demográficas y laborales?**  
            """),
        ], style={"padding": "20px"})
    ]
)
tab_objetivos = dbc.Tab(
    label="4. Objetivos y Justificación",
    children=[
        html.Div([
            html.H2("Objetivos del Proyecto"),
            html.Hr(),

            html.H3("Objetivo general"),
            dcc.Markdown("""
                Predecir de manera efectiva si los ingresos de las personas en Estados unidos son mayores de $50.000 USD al año mediante **RandomForestClassifier**, optimizado mediante **GridsearchCV**
            """),

            html.H3("Objetivos específicos"),
            html.Ul([
                html.Li("Realizar un análisis exploratorio detallado de variables numéricas y categóricas del Adult dataset."),
                html.Li("Evaluar relaciones y patrones relevantes entre características y el nivel de ingresos."),
                html.Li("Preparar los datos mediante un preprocesamiento adecuado para realizar el modelo."),
                html.Li("Entrenar un modelo de clasificación de RandomForest optimizado mediante GridsearchCV y evaluar su desempeño."),
                html.Li("Identificar las variables más importantes en la predicción del ingreso."),
            ]),

            html.H3("Justificación"),
            html.P("""
                La predicción de ingresos es un problema clásico en machine learning 
                con aplicaciones en economía laboral, estudios poblacionales y análisis social. 
                Comprender los factores que influyen en el nivel de ingresos permite mejorar 
                modelos de predicción, apoyar decisiones gubernamentales y entender desigualdades 
                estructurales que afectan a la población.
            """),
        ], style={"padding": "20px"})
    ]
)
tab_teorico = dbc.Tab(
    label="5. Marco Teórico",
    children=[
        html.Div([
            html.Hr(),

            dcc.Markdown(
                r"""
## Marco Teórico

La predicción del nivel de ingreso ha sido un tema central dentro de la econometría, la estadística aplicada
y la ciencia de datos. Diversos trabajos han utilizado técnicas de modelado supervisado para determinar los 
factores que explican la probabilidad de que un individuo obtenga ingresos superiores a cierto umbral 
(como el umbral estándar de **50K** en las bases de ingresos del UCI Machine Learning Repository).

En el contexto de modelos modernos de clasificación, enfoques basados en ensambles han demostrado ofrecer 
un rendimiento superior al capturar relaciones no lineales, interacciones complejas y patrones 
estructurales difíciles de modelar mediante técnicas lineales tradicionales.

---

## Análisis Exploratorio de Datos (EDA)

El EDA constituye una etapa esencial para comprender:
- Distribuciones de variables socioeconómicas.
- Evidencia preliminar de correlaciones y asociaciones.
- Identificación de valores atípicos o estructuras anómalas.
- Diferencias entre grupos poblacionales según el nivel de ingreso.

Formalmente, muchas de estas inspecciones se basan en estimadores clásicos como:
- **Estimadores de densidad**:  
$$
\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^n K\left(\frac{x - X_i}{h}\right)
$$
- **Correlación de Pearson**:  
$$
\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$
Estas herramientas permiten caracterizar las propiedades estadísticas de los datos antes del modelado.

---

## Árboles de Decisión
Los árboles de decisión son clasificadores basados en particiones recursivas del espacio de características.  
Dado un conjunto de entrenamiento:

$$
D = \{(x_i, y_i)\}_{i=1}^n, \quad x_i \in \mathbb{R}^p
$$


un árbol divide el espacio en regiones (R_1, R_2,....,R_M) tales que cada región asigna una clase 
mayoritaria. La división se realiza seleccionando una variable (X_j) y un umbral (s) que minimiza 
una medida de impureza, como el **índice Gini**:

$$
G(t) = \sum_{c=1}^{C} p(c|t)\,(1 - p(c|t))
$$

---

## Modelo Random Forest

![Diagrama Random Forest](/assets/random_forest_diagrama.jpeg)

Random Forest es un método de *bagging* que construye múltiples árboles usando:

1. **Muestreo bootstrap** para generar variabilidad:
$$
D^{(b)} \sim \text{Bootstrap}(D)
$$

2. **Selección aleatoria de características** en cada división:
   $$
m \subseteq \{1,\ldots,p\}, \quad |m| = k
$$

3. **Árboles no podados** que capturan toda la estructura de los datos.

Cada árbol produce una predicción 
$$ 
\\hat{y}^{(b)}(x) 
$$.  
La predicción final del bosque es el resultado de una votación mayoritaria:

$$
\hat{y}_{RF}(x) = \operatorname{mode}\left(\hat{y}^{(1)}(x), \ldots, \hat{y}^{(B)}(x)\right)
$$

Si se calcula la probabilidad estimada:

$$
\hat{P}(y = 1 \mid x) = \frac{1}{B} \sum_{b=1}^{B} \hat{P}^{(b)}(y=1\mid x)
$$

---

## Importancia de Variables

Una métrica fundamental para la interpretación es la **disminución promedio de impureza (MDI)**.  
Para una variable \\(X_j\\):

$$
\text{Imp}(X_j) = \sum_{b=1}^{B} \sum_{t \in T_{b,j}} \Delta G_t
$$

donde (T_{b,j}) indica los nodos del árbol (b) donde se utilizó la variable (X_j).  
Esta medida captura cuánto contribuye cada predictor a reducir la impureza a través del bosque.

Otra alternativa es la **Permutation Importance**, basada en el cambio en error fuera-de-bolsa (OOB):

$$
\Delta \text{Error}_j = \text{Err}_{OOB}(X_j^{\text{perm}}) - \text{Err}_{OOB}
$$

lo cual proporciona una medida robusta del impacto real de una característica.

---

## Fundamentación Teórica de su Desempeño

Random Forest posee resultados teóricos que explican su estabilidad:

- La combinación de árboles reduce la **varianza** del modelo sin incrementar significativamente el **sesgo**.
- El uso de variables aleatorias en cada nodo disminuye la **correlación** entre árboles, mejorando el ensamble.
- Bajo ciertas condiciones, Random Forest es **consistente**:
  
$$
\hat{f}_{RF}(x) \xrightarrow[]{P} f(x)
$$

Esto garantiza que el modelo converge a la función verdadera al aumentar (n) y (B).
                """
            ,mathjax=True),
        ])
    ]
)
tab_metodologia = dbc.Tab(
    label="6. Metodología",
    children=[
        html.Div([
            html.H2("Metodología del Proyecto"),
            html.Hr(),

            dbc.Tabs([
                # ----------------------------- #
                #       SUBTAB A
                # ----------------------------- #
                dbc.Tab(
                    label="a. Definición del Problema",
                    children=[
                        html.H3("a. Definición del Problema"),
                        html.P("""
                            El problema es de clasificación binaria, donde se busca predecir si el 
                            ingreso de una persona supera o no los $50.000 USD al año. Las clases 
                            fueron codificadas como 0 y 1 para mayor facilidad de comprensión, 
                            siendo 0 menor a 50 mil dólares y 1 mayor a 50 mil dólares. 
                        """),
                        html.P("Variable objetivo: Income "),
                    ],
                    style={"padding": "20px"}
                ),

                # ----------------------------- #
                #       SUBTAB B
                # ----------------------------- #
                dbc.Tab(
                 label="b. Preprocesamiento de Datos",
                 children=[
                    html.Div(
                    [
                        html.H3("Preprocesamiento de Datos"),
                        pipeline_list
                    ],
                    style={"padding": "20px"}
                    )
                ]
            ),

                # ----------------------------- #
                #       SUBTAB C
                # ----------------------------- #
                dbc.Tab(
                    label="c. Selección del Modelo",
                    children=[
                        html.H3("c. Selección del Modelo"),
                        dcc.Markdown("""
                            El modelo seleccionado para la prediccion de ingresos fue el de **RandomForest**
                            
                            ### Aplicabilidad del modelo al problema del ingreso
                            El modelo de RandomForest es especialmente adecuado para este problema porque:
                            - Maneja relaciones no lineales y efectos de interacción entre variables socioeconómicas.
                            - Es robusto a valores atípicos y ruido.
                            - Requiere poca preparación previa de los datos (no asume linealidad ni normalidad).
                            - Permite interpretar la relevancia relativa de cada característica.
                            
                            ### Proceso de entrenamiento 
                            
                            ![K-Fold](/assets/K-fold.png)
                        """),
                    ],
                    style={"padding": "20px"}
                ),

                # ----------------------------- #
                #       SUBTAB D
                # ----------------------------- #
                dbc.Tab(
    label="d. Evaluacion del modelo",
    children=[
        html.H3("Métricas de Evaluación"),
        html.Hr(),

        dcc.Markdown(r"""
## **1. Accuracy**
Mide la proporción de predicciones correctas sobre todas las muestras:
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

## **2. Precision**
Proporción de predicciones positivas que realmente son positivas:
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

---

## **3. Recall**
Proporción de positivos reales que fueron correctamente identificados:
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

---

## **4. F1-Score**
Media armónica entre Precision y Recall:
$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

## **5. ROC-AUC**
La curva ROC grafica:
### TPR y FPR

$$
TPR = \frac{TP}{TP + FN}
$$

$$
FPR = \frac{FP}{FP + TN}
$$

para distintos umbrales del clasificador.
### Área bajo la curva
El área bajo la curva ROC se define como:
$$
\text{ROC-AUC} = \int_{0}^{1} TPR(FPR)\, d(FPR)
$$
""",mathjax=True)
    ],
    style={"padding": "20px"}
)
,
            ])
        ])
    ]
)
tab_etl = dbc.Tab(
    label="7. ETL - Extracción, Transformación y Carga",
    children=[
        html.Div(
            [
                html.H2("ETL - Extracción, Transformación y Carga, Análisis inicial"),
                html.Hr(),
                html.P(
                    "Se realizó la concatenación de los dos dataframes (train y test) para crear un dataset general, "
                    "agregando también los nombres de las columnas."
                ),

                html.H4("Vista previa de 'adult.data'"),
                dash_table.DataTable(
                    data=df1_head.to_dict("records"),
                    columns=[{"name": str(i), "id": str(i)} for i in df1_head.columns],
                    page_size=5,
                    style_table={"overflowX": "auto"},
                ),

                html.H4("Vista previa de 'adult.test'"),
                dash_table.DataTable(
                    data=df2_head.to_dict("records"),
                    columns=[{"name": str(i), "id": str(i)} for i in df2_head.columns],
                    page_size=5,
                    style_table={"overflowX": "auto"},
                ),
                html.Hr(),
                html.H4("Dataset final concatenado"),
                dash_table.DataTable(
                    data=df.head().to_dict("records"),
                    columns=[{"name": str(i), "id": str(i)} for i in df.columns],
                    page_size=5,
                    style_table={"overflowX": "auto"},
                ),

                html.Br(),
                html.H4("Tipos de datos despues de concatenar y transformar"),
                html.Pre(str(df.dtypes)),

                html.Hr(),
                html.H4("Filas duplicadas"),
                html.P(f"Total de filas duplicadas: {len(duplicadas)}"),

                (
                    dash_table.DataTable(
                        data=duplicadas.to_dict("records"),
                        columns=[{"name": str(i), "id": str(i)} for i in duplicadas.columns],
                        page_size=5,
                        style_table={"overflowX": "auto"},
                    )
                    if len(duplicadas) > 0
                    else html.Div(
                        dbc.Alert("No se encontraron filas duplicadas.", color="success")
                    )
                ),

                html.P(
                    "En general, estas filas duplicadas se deben más a casualidad que a error, "
                    "por lo que no se van a eliminar."
                ),
            ],
            style={"padding": "20px"},
        )
    ],
)
tab_Resultados = dbc.Tab(
    label='8. Resultados y Analisis Final',
    children=[
        html.Div([
            html.H2('Análisis Exploratorio de Datos (EDA) y Modelo'),
            html.Hr(),
            subtabs_analisis
        ], style={"padding": "20px"})
    ]
)
tab_conclusiones = dbc.Tab(
    label="9. Conclusiones",
    children=[
        html.Div(
            [
                html.H2("Conclusiones Generales"),
                html.Hr(),
                dcc.Markdown(
                    """
                    ## Conclusiones generales sobre analisis univariado 
                    - Se observa la presencia de datos atipicos para las variables numericas y presencia de datos Nan para las variables categoricas que deben ser tratados
                    - Se observa que algunos de los graficos como el de Age o Fnlwgt tienen un sesgo positivo, por lo que la mayoria de los datos se concentran en los valores inferiores 
                    - Se observa tambien que hay bastantes valores de 0 en capital gain o capital loss, sin embargo estos valores no los podemos considerar como valores faltantes ya que si son valores reales
                    - Tambien en las variables categoricas se observa que la variable income esta desbalanceada, esto se debe tener en cuenta si se realiza algun modelo con esta variable como objetivo


                    ---

                    ## Conclusiones generales sobre analisis bivariado 
                    - Podemos observar que en general, las correlaciones entre las variables numericas del dataset son bastante bajas, tanto en las correlaciones de pearson y spearman ninguno supera mas de 0.2 de correlacion. 
                    En general al realizar la correlacion con el metodo de spearman se generaron mejores resultados de los que habia a comparación de cuando se calcula con spearman
                    - Podemos observar que en las variables categoricas, el p-valor es menor que 0.05 por lo que todos los pares tienen relaciones significativas. 
                    Sin embargo para un mayor entendimiento se calculo tambien la fuerza mediante cramer para ver cuales eran asociadas con mayor fuerza, las cuales en su mayoria son las variables que estaban mas relacionadas en concepto como raza y pais nativo, las mas moderadas tampoco se pueden ignorar porque estas tambien muestran patrones importantes como la relacion entre ingresos y la educacion 
                    - Se utilizaron las pruebas del test de levene y la normalidad con kstest para determinar si se debia usar ANOVA o Kruskall wallis para el analisis bivariado entre categoricas y numericas, en general al no ser normales y los pares no tener varianzas homogeneas se utilizo kruskall wallis y se nota que en la mayoria de los casos hay diferencias estadisticamente significativas entre los pares, 
                    lo que significa que hay relaciones fuertes entre ellas, la unica excepcion es  income y Fnlwgt cuyo p-valor es mayor con 0.09 por lo que los ingresos no estan relacionados con el peso de muestra poblacional




                    ---
                    ## Conclusiones generales sobre las correlaciones del dataset
                    - Podemos observar que en cuanto a las correlaciones de variables hay varias cosas curiosas por ejemplo Education y Education num tienen correlacion perfecta, por lo que si se llega a realizar algun modelado con el dataset se debe eliminar 1 de ellas para no generar data leakage.
                    - En general las variables categoricas estan muchisimo mas relacionadas que las variables numericas lo cual muestra que las variables numericas son mas independientes entre si en comparacion y se debe tomar en cuenta si se realiza el modelado con este dataset 
                    ---
                    ## Conclusiones del modelo
                    - El modelo realizado tiene una ROC AUC de 0.91 lo que nos muestra que realiza una buena discriminacion entre ambas clases 
                    - El modelo en general reconoce la mayoria de los casos de ambas clases con pocos falsos negativos en la matriz de confusion 
                    - Las metricas generales del modelo rondan alrededor de 0.7 en la clase de ganancias mayores a 50k y 0.8 para la clase de ganancias 0 
                    - Las variables mas importantes para el modelo son edad, estado civil, horas trabajadas y capital-gain, todas intuitivamente relacionadas con ingresos.
                    - **Limitaciones del modelo**: el modelo al ser randomforest es bastante tardado en la ejecucion sobre todo en la busqueda de hiperparametros realizada al ser un dataset de tamaño mediano, ademas de que tiene cierto desbalance el dataset lo cual puede complicar los resultados si no se usa tecnicas de balanceo como SMOTE o ADASYN. 
                    También existen correlaciones entre variables que pueden distorsionar la interpretación de la importancia de las características, por lo que se deben tomar en cuenta estos aspectos si se quieren realizar analisis futuros.
                    
                    ## Conclusion General
                    En conjunto, el proyecto ofrece una visión robusta y detallada del comportamiento del dataset y demuestra la viabilidad de predecir niveles de ingreso con modelos supervisados. 
                    Sin embargo, se recomienda profundizar en técnicas de balanceo, selección de características y optimización computacional para mejorar la interpretabilidad y el rendimiento futuro de los modelos.
                    """
                ),
            ],
            style={"padding": "20px"},
        )
    ],
)

tabs = dcc.Tabs([tab_introduccion,tab_contexto,tab_problema,tab_objetivos,tab_teorico,tab_metodologia,tab_etl,tab_Resultados,tab_conclusiones])
app.layout = dbc.Container([
    html.H1(" Exploratory Data Analysis - Adult Dataset", className="text-center my-4"),
    html.Hr(),
    tabs
], fluid=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)