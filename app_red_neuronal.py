"""
app_red_neuronal.py

Trabajo Practico PSR-TP01-C2 - Desarrollo de una Aplicacion con Redes Neuronales
sobre Datasets de Kaggle.

Grupo: Carolina Lobo y Sabrina De Marco
Dataset: Heart Disease Dataset (Kaggle - johnsmith88)
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Aplicacion en Streamlit, implementada bajo el paradigma Orientado a Objetos (POO):
- GestorDatos: carga, limpieza, normalizacion y particionamiento (70/30) del dataset.
- RedNeuronalMLP: entrenamiento, evaluacion y prediccion del Perceptron Multicapa.
- DashboardApp: interfaz interactiva que orquesta las dos clases anteriores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATASET_PATH = "dataset.csv"
TARGET_COL = "target"
TEST_SIZE = 0.30  # RF3: division 70% train / 30% test


class GestorDatos:
    """Carga, limpia, normaliza y particiona el dataset (RF1, RF2, RF3)."""

    def __init__(self, ruta_csv: str = DATASET_PATH, columna_objetivo: str = TARGET_COL):
        self.ruta_csv = ruta_csv
        self.columna_objetivo = columna_objetivo
        self.df_crudo = None
        self.df_limpio = None
        self.columnas = None
        self.scaler = StandardScaler()

    def cargar(self) -> pd.DataFrame:
        self.df_crudo = pd.read_csv(self.ruta_csv)
        return self.df_crudo

    def limpiar(self) -> pd.DataFrame:
        self.df_limpio = self.df_crudo.drop_duplicates().dropna().reset_index(drop=True)
        self.columnas = [c for c in self.df_limpio.columns if c != self.columna_objetivo]
        return self.df_limpio

    def obtener_datos_normalizados(self) -> pd.DataFrame:
        """RF2: dataframe con las features normalizadas, para comparar contra el crudo."""
        X = self.df_limpio[self.columnas]
        X_norm = StandardScaler().fit_transform(X)
        df_norm = pd.DataFrame(X_norm, columns=self.columnas).round(4)
        df_norm[self.columna_objetivo] = self.df_limpio[self.columna_objetivo].values
        return df_norm

    def dividir_train_test(self, test_size: float = TEST_SIZE, random_state: int = 42):
        """RF3: division automatica 70% train / 30% test, estratificada."""
        X = self.df_limpio[self.columnas]
        y = self.df_limpio[self.columna_objetivo]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train_esc = self.scaler.fit_transform(X_train)
        X_test_esc = self.scaler.transform(X_test)

        return X_train_esc, X_test_esc, y_train.values, y_test.values

    def escalar_muestra(self, fila: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(fila[self.columnas])


class RedNeuronalMLP:
    """Perceptron Multicapa (MLP): entrenamiento, evaluacion y prediccion (RF4)."""

    def __init__(
        self,
        hidden_layer_sizes=(16, 8),
        activation: str = "relu",
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        self.modelo = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )

    def entrenar(self, X_train, y_train) -> list:
        """Entrena la red y devuelve el error (loss) registrado en cada epoca."""
        self.modelo.fit(X_train, y_train)
        return list(getattr(self.modelo, "loss_curve_", []))

    def evaluar(self, X_test, y_test) -> dict:
        y_pred = self.modelo.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
        }

    def predecir(self, X):
        return self.modelo.predict(X), self.modelo.predict_proba(X)


class DashboardApp:
    """Interfaz interactiva en Streamlit (RF5): datos, convergencia y prediccion."""

    FEATURE_INFO = {
        "age": ("Edad (anios)", 29, 77, 54, 1),
        "sex": ("Sexo (1 = hombre, 0 = mujer)", 0, 1, 1, 1),
        "cp": ("Tipo de dolor de pecho (0-3)", 0, 3, 0, 1),
        "trestbps": ("Presion arterial en reposo (mm Hg)", 90, 200, 130, 1),
        "chol": ("Colesterol serico (mg/dl)", 120, 570, 246, 1),
        "fbs": ("Glucemia en ayunas > 120 mg/dl (1 = si)", 0, 1, 0, 1),
        "restecg": ("Resultado ECG en reposo (0-2)", 0, 2, 1, 1),
        "thalach": ("Frecuencia cardiaca maxima alcanzada", 70, 205, 150, 1),
        "exang": ("Angina inducida por ejercicio (1 = si)", 0, 1, 0, 1),
        "oldpeak": ("Depresion del ST inducida por ejercicio", 0.0, 6.5, 1.0, 0.1),
        "slope": ("Pendiente del segmento ST (0-2)", 0, 2, 1, 1),
        "ca": ("Numero de vasos coloreados (0-4)", 0, 4, 0, 1),
        "thal": ("Talasemia (1-3)", 0, 3, 2, 1),
    }

    def __init__(self, ruta_csv: str = DATASET_PATH):
        self.ruta_csv = ruta_csv
        self.gestor = None

    def _cargar_datos(self):
        """Reutiliza el mismo GestorDatos (y su scaler ya ajustado) a traves de los
        reruns de Streamlit, en vez de crear una instancia nueva cada vez."""
        if "gestor" not in st.session_state:
            gestor = GestorDatos(self.ruta_csv)
            gestor.cargar()
            gestor.limpiar()
            st.session_state["gestor"] = gestor
        self.gestor = st.session_state["gestor"]

    def run(self):
        st.set_page_config(page_title="MLP - Heart Disease (PSR-TP01-C2)", layout="wide")
        st.title("Perceptron Multicapa (MLP) - Prediccion de Enfermedad Cardiaca")
        st.caption(
            "Trabajo Practico PSR-TP01-C2 | Grupo: Carolina Lobo y Sabrina De Marco | "
            "Dataset: Heart Disease Dataset (Kaggle - johnsmith88)"
        )

        self._cargar_datos()

        tab_datos, tab_entrenamiento, tab_prediccion = st.tabs(
            ["1. Vista de Datos", "2. Entrenamiento y Convergencia", "3. Prediccion"]
        )

        with tab_datos:
            self._vista_datos()
        with tab_entrenamiento:
            self._vista_entrenamiento()
        with tab_prediccion:
            self._vista_prediccion()

    # ---------------- RF5.1: Vista de Datos ----------------
    def _vista_datos(self):
        st.subheader("Dataset original (crudo)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas (tras limpieza)", len(self.gestor.df_limpio))
        col2.metric("Columnas", self.gestor.df_limpio.shape[1])
        col3.metric(
            "Casos positivos (target=1)",
            int((self.gestor.df_limpio[TARGET_COL] == 1).sum()),
        )
        st.dataframe(self.gestor.df_limpio.head(15), use_container_width=True)

        st.subheader("Dataset normalizado (StandardScaler)")
        st.caption("Mismos registros que arriba, con las features transformadas: z = (x - media) / desvio_estandar")
        df_norm = self.gestor.obtener_datos_normalizados()
        st.dataframe(df_norm.head(15), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.write("Estadisticas - datos crudos")
            st.dataframe(self.gestor.df_limpio[self.gestor.columnas].describe().round(2), use_container_width=True)
        with c2:
            st.write("Estadisticas - datos normalizados")
            st.dataframe(df_norm[self.gestor.columnas].describe().round(2), use_container_width=True)

        st.subheader("Distribucion de la variable objetivo")
        fig, ax = plt.subplots(figsize=(4, 3))
        self.gestor.df_limpio[TARGET_COL].value_counts().sort_index().plot(
            kind="bar", ax=ax, color=["#4C72B0", "#DD8452"]
        )
        ax.set_xlabel("target (0 = sin enfermedad, 1 = con enfermedad)")
        ax.set_ylabel("Cantidad")
        st.pyplot(fig, use_container_width=False)

    # ---------------- RF4 + RF5.2: Entrenamiento y Convergencia ----------------
    def _vista_entrenamiento(self):
        st.caption("Division automatica y estratificada: 70% entrenamiento / 30% prueba.")

        st.subheader("Hiperparametros de la red")
        c1, c2, c3 = st.columns(3)
        with c1:
            capa1 = st.slider("Neuronas capa oculta 1", 2, 64, 16)
            capa2 = st.slider("Neuronas capa oculta 2 (0 = sin segunda capa)", 0, 64, 8)
        with c2:
            activation = st.selectbox("Funcion de activacion", ["relu", "tanh", "logistic"], index=0)
            solver = st.selectbox("Optimizador", ["adam", "sgd", "lbfgs"], index=0)
        with c3:
            learning_rate = st.number_input(
                "Tasa de aprendizaje (learning rate)", 0.0001, 1.0, 0.001, step=0.0001, format="%.4f"
            )
            max_iter = st.slider("Numero de epocas (epochs)", 50, 2000, 500, step=50)

        entrenar = st.button("Entrenar modelo", type="primary")

        if entrenar or "modelo" in st.session_state:
            if entrenar:
                hidden_layers = (capa1,) if capa2 == 0 else (capa1, capa2)
                X_train, X_test, y_train, y_test = self.gestor.dividir_train_test()

                red = RedNeuronalMLP(
                    hidden_layer_sizes=hidden_layers,
                    activation=activation,
                    solver=solver,
                    learning_rate_init=learning_rate,
                    max_iter=max_iter,
                )
                with st.spinner("Entrenando la red neuronal..."):
                    loss_curve = red.entrenar(X_train, y_train)
                metrics = red.evaluar(X_test, y_test)

                st.session_state["modelo"] = red
                st.session_state["loss_curve"] = loss_curve
                st.session_state["metrics"] = metrics
                st.session_state["train_size"] = len(X_train)
                st.session_state["test_size"] = len(X_test)

            red = st.session_state["modelo"]
            loss_curve = st.session_state["loss_curve"]
            metrics = st.session_state["metrics"]

            st.success(
                f"Modelo entrenado: {st.session_state['train_size']} pacientes de train, "
                f"{st.session_state['test_size']} de test."
            )

            st.subheader("Grafica de evolucion del error (Loss Curve)")
            if loss_curve:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(loss_curve, color="#4C72B0")
                ax.set_xlabel("Epoca")
                ax.set_ylabel("Error (Binary Cross-Entropy)")
                ax.set_title("Convergencia del entrenamiento")
                st.pyplot(fig, use_container_width=False)

                epoca_estable = self._detectar_epoca_convergencia(loss_curve)
                st.caption(
                    f"El error se estabiliza aproximadamente en la epoca {epoca_estable} "
                    f"(a partir de ahi se mantiene dentro del 2% del error final)."
                )
            else:
                st.info("El optimizador seleccionado (lbfgs) no genera curva de error por epoca.")

            st.subheader("Metricas sobre el 30% de prueba")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m2.metric("Precision", f"{metrics['precision']:.3f}")
            m3.metric("Recall", f"{metrics['recall']:.3f}")
            m4.metric("F1-score", f"{metrics['f1']:.3f}")

            with st.expander("Reporte de clasificacion completo"):
                st.text(metrics["classification_report"])
        else:
            st.info("Configura los hiperparametros y presiona 'Entrenar modelo' para comenzar.")

    @staticmethod
    def _detectar_epoca_convergencia(loss_curve, tolerancia=0.02):
        """Primera epoca a partir de la cual el error se mantiene, en adelante,
        dentro de una banda de tolerancia respecto del error final (estabilidad
        sostenida, no una unica caida puntual)."""
        loss_final = loss_curve[-1]
        rango = max(loss_curve) - loss_final
        if rango <= 0:
            return 0
        banda = tolerancia * rango
        for i in range(len(loss_curve)):
            if all(abs(l - loss_final) <= banda for l in loss_curve[i:]):
                return i
        return len(loss_curve) - 1

    # ---------------- RF5.3: Prediccion sobre el 30% de prueba ----------------
    def _vista_prediccion(self):
        if "modelo" not in st.session_state:
            st.warning("Primero entrena un modelo en la pestana 'Entrenamiento y Convergencia'.")
            return

        red: RedNeuronalMLP = st.session_state["modelo"]
        metrics = st.session_state["metrics"]

        col_form, col_matriz = st.columns([3, 1])

        with col_matriz:
            st.caption("Matriz de confusion (30% prueba)")
            cm = metrics["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(2.2, 2.2))
            ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color="black")
            ax.set_xlabel("Prediccion", fontsize=7)
            ax.set_ylabel("Real", fontsize=7)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=7)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)

        with col_form:
            st.subheader("Prediccion en tiempo real sobre un nuevo paciente")
            valores = {}
            cols = st.columns(3)
            for i, col_name in enumerate(self.gestor.columnas):
                label, minv, maxv, default, step = self.FEATURE_INFO.get(col_name, (col_name, 0, 100, 0, 1))
                with cols[i % 3]:
                    if isinstance(step, float):
                        valores[col_name] = st.number_input(
                            label, min_value=float(minv), max_value=float(maxv),
                            value=float(default), step=step, key=f"input_{col_name}"
                        )
                    else:
                        valores[col_name] = st.number_input(
                            label, min_value=int(minv), max_value=int(maxv),
                            value=int(default), step=step, key=f"input_{col_name}"
                        )

            if st.button("Predecir", type="primary"):
                fila = pd.DataFrame([valores])[self.gestor.columnas]
                fila_escalada = self.gestor.escalar_muestra(fila)
                pred, proba = red.predecir(fila_escalada)

                if pred[0] == 1:
                    st.error(f"Resultado: Riesgo de enfermedad cardiaca (probabilidad {proba[0][1]:.1%})")
                else:
                    st.success(f"Resultado: Sin riesgo detectado (probabilidad {proba[0][0]:.1%})")
                st.progress(float(proba[0][1]), text=f"Probabilidad de enfermedad: {proba[0][1]:.1%}")


if __name__ == "__main__":
    app = DashboardApp()
    app.run()
