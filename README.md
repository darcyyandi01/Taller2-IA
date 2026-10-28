# 🎓 Predicción de Aprobación de Estudiantes (MLP) - Taller 2 IA

Este repositorio contiene el desarrollo del **Taller 2 de Inteligencia Artificial**, enfocado en la construcción, entrenamiento y despliegue de una red neuronal artificial de tipo **Perceptrón Multicapa (MLP)** para predecir si un estudiante aprobará o no una asignatura a partir de su desempeño académico y hábitos de vida.

---

## 📌 Contenido del Proyecto

- **`taller_notas.ipynb`**: Cuaderno Jupyter con todo el flujo de ciencia de datos y redes neuronales:
  1. Carga y exploración de datos.
  2. Imputación de valores faltantes (usando la mediana de cada variable).
  3. Tratamiento de valores atípicos mediante el método de Rango Intercuartílico (IQR).
  4. División en conjuntos de entrenamiento (80%) y prueba (20%) de forma estratificada.
  5. Escalado de variables predictoras con `StandardScaler` (media 0, varianza 1).
  6. Entrenamiento de la Red Neuronal con **Perceptrón Multicapa (MLPClassifier)**.
  7. Evaluación exhaustiva con matriz de confusión, reporte de clasificación (Accuracy, Precision, Recall, F1-Score).
  8. Exportación del modelo y escalador entrenados (`.pkl`).
- **`app.py`**: Aplicación web interactiva desarrollada con **Streamlit** que permite ingresar nuevos parámetros de estudiantes y obtener en tiempo real el pronóstico de aprobación y la probabilidad calculada.
- **`dataset_estudiantes_balanceado.csv`**: Conjunto de datos con 600 registros de estudiantes balanceado al 50%.
- **`modelo_estudiantes.pkl`** y **`scaler.pkl`**: Pesos de la red neuronal MLP y del escalador estándar.
- **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.

---

## 📊 Variables del Modelo

El modelo utiliza las siguientes variables predictoras:

| Variable | Descripción | Rango / Escala |
| :--- | :--- | :--- |
| `Tareas` | Calificación acumulada de entregables prácticos | 0 - 100 |
| `Examenes` | Calificación promedio de parciales y examen final | 0 - 100 |
| `Asistencia` | Porcentaje de asistencia a clases | 0 - 100% |
| `Estudio_Horas` | Promedio de horas semanales de estudio independiente | Horas / semana |
| `Sueno_Horas` | Promedio de horas de descanso nocturno diario | Horas / día |
| `Extracurriculares` | Participación en talleres, deportes o actividades culturales | 0 (No), 1 (Sí) |

**Variable Objetivo:**
- `Aprobado`: Clase binaria balanceada ($1 = \text{Aprobado}$, $0 = \text{Reprobado}$).

*(Nota: La variable `Calificacion_Final` es una variable de salida que se excluye de las características de entrada para evitar filtración de datos).*

---

## 📈 Rendimiento del Modelo (MLP)

Evaluado sobre el conjunto de prueba independiente (20% de los datos):
- **Accuracy (Exactitud):** `0.9667` (~96.7% de aciertos globales)
- **Precision (Precisión):** `0.9516` (Clase Aprobado) / `0.9828` (Clase Reprobado)
- **Recall (Sensibilidad):** `0.9833` (Clase Aprobado) / `0.9500` (Clase Reprobado)
- **F1-Score:** `0.9672` (Clase Aprobado) / `0.9661` (Clase Reprobado)

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/darcyyandi01/Taller2-IA.git
cd Taller2-IA
```

### 2. Crear y activar un entorno virtual (recomendado)
```bash
# En Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# En Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación Interactiva (Streamlit)
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`.

### 5. Ejecutar el Cuaderno Jupyter
```bash
jupyter notebook taller_notas.ipynb
```

---

## 📁 Estructura del Repositorio

```text
├── .gitignore                          # Archivos y carpetas ignorados por Git
├── README.md                           # Documentación principal del proyecto
├── requirements.txt                    # Dependencias del proyecto
├── app.py                              # Interfaz web interactiva con Streamlit
├── taller_notas.ipynb                  # Flujo de análisis y entrenamiento en Jupyter
├── dataset_estudiantes_balanceado.csv  # Conjunto de datos utilizado
├── modelo_estudiantes.pkl              # Red Neuronal MLP exportada
└── scaler.pkl                          # Objeto StandardScaler exportado
```

---

## 👤 Autor
- **Darcy Yandi**
