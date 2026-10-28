import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Aprobación de Estudiantes",
    page_icon="🎓",
    layout="wide"
)

# Cargar el modelo entrenado y el escalador de forma robusta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_estudiantes.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load(MODEL_PATH)
    escalador = joblib.load(SCALER_PATH)
    return modelo, escalador

try:
    model, scaler = cargar_artefactos()
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

# Título y descripción
st.title("🎓 Predicción de Aprobación de Estudiantes (MLP)")
st.markdown("""
Esta aplicación utiliza un modelo de **Deep Learning (Perceptrón Multicapa - MLP)** entrenado para predecir si un estudiante 
aprobará o reprobará la asignatura a partir de su desempeño académico y hábitos de vida.
""")

# Barra lateral con información del modelo
with st.sidebar:
    st.header("ℹ️ Información del Modelo")
    st.markdown("""
    - **Algoritmo:** Perceptrón Multicapa (MLPClassifier)
    - **Arquitectura:** Capas ocultas (64, 32) con activación ReLU y optimizador Adam
    - **Métricas de Rendimiento:**
      - **Accuracy:** ~96.7% - 97.5%
      - **Precision:** ~0.97
      - **Recall:** ~0.97
      - **F1-Score:** ~0.97
    - **Dataset:** 600 registros balanceados (50% aprobados / 50% reprobados)
    """)
    st.divider()
    st.caption("Taller 2 - Inteligencia Artificial")

# Formulario de entrada dividido en 2 columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Desempeño Académico")
    tareas = st.number_input(
        "Tareas (0 - 100)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0,
        help="Calificación acumulada de entregables y talleres prácticos"
    )
    examenes = st.number_input(
        "Exámenes (0 - 100)",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=1.0,
        help="Promedio ponderado de exámenes parciales y final"
    )
    asistencia = st.number_input(
        "Asistencia (0 - 100%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0,
        help="Porcentaje global de asistencia presencial a clases"
    )

with col2:
    st.subheader("⏱️ Hábitos y Entorno")
    estudio_horas = st.number_input(
        "Horas de Estudio Semanales",
        min_value=0.0,
        max_value=50.0,
        value=14.0,
        step=0.5,
        help="Promedio de horas dedicadas al estudio independiente por semana"
    )
    sueno_horas = st.number_input(
        "Horas de Sueño Diarias",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5,
        help="Promedio de horas de descanso diario nocturno"
    )
    extracurriculares = st.selectbox(
        "Actividades Extracurriculares",
        options=[0, 1],
        format_func=lambda x: "Sí (Participa activamente)" if x == 1 else "No (No participa)",
        help="Indicador de participación activa en talleres, deportes o actividades culturales"
    )

st.divider()

# Botón de predicción
if st.button("🚀 Generar Predicción", type="primary", use_container_width=True):
    feature_names = [
        'Tareas',
        'Examenes',
        'Asistencia',
        'Estudio_Horas',
        'Sueno_Horas',
        'Extracurriculares'
    ]

    input_df = pd.DataFrame([[
        tareas,
        examenes,
        asistencia,
        estudio_horas,
        sueno_horas,
        extracurriculares
    ]], columns=feature_names)

    # Escalamiento y predicción
    input_scaled = scaler.transform(input_df)
    prediccion = int(model.predict(input_scaled)[0])
    probas = model.predict_proba(input_scaled)[0]
    prob_apr = probas[1] * 100
    prob_rep = probas[0] * 100

    st.subheader("🎯 Resultado de la Estimación")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(
            label="Pronóstico del Estudiante",
            value="🎉 APROBADO" if prediccion == 1 else "⚠️ REPROBADO"
        )
    with res_col2:
        st.metric(
            label="Probabilidad Estimada de Aprobación",
            value=f"{prob_apr:.1f}%"
        )

    if prediccion == 1:
        st.success(f"Predicción generada con éxito: El estudiante tiene una probabilidad de aprobación del **{prob_apr:.1f}%**. Mantiene hábitos y rendimiento favorables.")
    else:
        st.error(f"Alerta temprana: El estudiante tiene un pronóstico desfavorable con **{prob_rep:.1f}%** de riesgo de reprobar. Se recomienda intervención pedagógica y tutorías de apoyo.")
