<div align="center">

<img src="logo_colegio.png" width="110" alt="Logo"/>

# Perceptrón Multicapa (MLP) — Heart Disease Dataset

**Trabajo Práctico PSR-TP01-C2 · Programación de Redes**

Carolina Lobo · Sabrina De Marco

</div>

---

## 📋 Descripción

Aplicación de red neuronal **Perceptrón Multicapa (MLP)** para predecir el riesgo de
enfermedad cardíaca a partir de indicadores clínicos, usando el
[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
de Kaggle. Incluye un dashboard interactivo en **Streamlit** para explorar los datos,
entrenar el modelo con distintos hiperparámetros y predecir en tiempo real.

## 📁 Contenido del repositorio

| Archivo | Descripción |
|---|---|
| [`dataset.csv`](dataset.csv) | Dataset original descargado de Kaggle |
| [`app_red_neuronal.py`](app_red_neuronal.py) | Código fuente en POO (clases `GestorDatos`, `RedNeuronalMLP`, `DashboardApp`) |
| [`Informe_Tecnico_TP.pdf`](Informe_Tecnico_TP.pdf) | Informe técnico (versión final) |
| [`Informe_Tecnico_TP.docx`](Informe_Tecnico_TP.docx) | Informe técnico (versión editable en Word) |
| [`requirements.txt`](requirements.txt) | Dependencias del proyecto |

## 🚀 Cómo correr la aplicación

```bash
pip install -r requirements.txt
python -m streamlit run app_red_neuronal.py
```

Se abre en `http://localhost:8501` con 3 pestañas:

1. **Vista de Datos** — dataset original vs. normalizado, distribución de la variable objetivo
2. **Entrenamiento y Convergencia** — hiperparámetros ajustables, curva de error y métricas
3. **Predicción** — matriz de confusión + formulario para predecir un paciente nuevo

## 🧠 Arquitectura del modelo

| Capa | Neuronas | Activación |
|---|---|---|
| Entrada | 13 | — |
| Oculta 1 | 16 | ReLU |
| Oculta 2 | 8 | ReLU |
| Salida | 1 | Sigmoide |

**División:** 70% entrenamiento / 30% prueba (estratificada) · **Optimizador:** Adam · **Épocas:** 500

## 📊 Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 0.813 |
| Precision | 0.848 |
| Recall | 0.796 |
| F1-score | 0.821 |

Detalle completo del análisis (convergencia, overfitting, conclusiones) en el
[informe técnico](Informe_Tecnico_TP.pdf).
