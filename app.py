import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as parquet
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mejor_modelo_regresion_logistica.zip"
HEALTHY_IMAGE_PATH = BASE_DIR / "Paciente_Sano.jpg"
WARNING_IMAGE_PATH = BASE_DIR / "Posible_Ataque_Cardiaco.jpg"


@dataclass(frozen=True)
class ModelParameters:
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    threshold: float


def _read_metadata(archive: zipfile.ZipFile, class_name: str) -> dict:
    for filename in archive.namelist():
        if filename.startswith("stages/") and filename.endswith(".txt"):
            metadata = json.loads(archive.read(filename))
            if metadata.get("class", "").endswith(class_name):
                return metadata
    raise ValueError(f"No se encontró la metadata del modelo {class_name}.")


def _read_stage_data(
    archive: zipfile.ZipFile, stage_uid: str
) -> dict:
    filename = next(
        (
            name
            for name in archive.namelist()
            if name.startswith("stages/")
            and f"_{stage_uid}/data/" in name
            and name.endswith(".parquet")
        ),
        None,
    )
    if filename is None:
        raise ValueError(f"No se encontraron los datos del componente {stage_uid}.")
    table = parquet.read_table(io.BytesIO(archive.read(filename)))
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"El componente {stage_uid} no contiene parámetros.")
    return rows[0]


def _load_model_parameters() -> ModelParameters:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH.name}")

    with zipfile.ZipFile(MODEL_PATH) as archive:
        assembler_metadata = _read_metadata(archive, "VectorAssembler")
        scaler_metadata = _read_metadata(archive, "StandardScalerModel")
        logistic_metadata = _read_metadata(archive, "LogisticRegressionModel")
        scaler = _read_stage_data(archive, scaler_metadata["uid"])
        logistic = _read_stage_data(archive, logistic_metadata["uid"])

    input_columns = tuple(assembler_metadata["paramMap"]["inputCols"])
    if input_columns != ("edad_imp", "colesterol_imp"):
        raise ValueError("El modelo no espera edad y colesterol en ese orden.")

    means = tuple(scaler["mean"]["values"])
    standard_deviations = tuple(scaler["std"]["values"])
    coefficients = tuple(logistic["coefficientMatrix"]["values"])
    intercept_values = logistic["interceptVector"]["values"]

    if not means or len(means) != len(standard_deviations):
        raise ValueError("El escalador del modelo contiene parámetros incompatibles.")
    if len(coefficients) != len(means) or len(intercept_values) != 1:
        raise ValueError("El clasificador no coincide con las variables de entrada.")
    if logistic["numClasses"] != 2 or logistic["numFeatures"] != len(means):
        raise ValueError("El archivo no contiene un clasificador binario compatible.")
    if any(value == 0 for value in standard_deviations):
        raise ValueError("El modelo contiene una desviación estándar igual a cero.")

    threshold = logistic_metadata.get("defaultParamMap", {}).get("threshold", 0.5)
    return ModelParameters(
        means=means,
        standard_deviations=standard_deviations,
        coefficients=coefficients,
        intercept=intercept_values[0],
        threshold=float(threshold),
    )


@st.cache_resource
def load_model_parameters() -> ModelParameters:
    return _load_model_parameters()


def predict_risk(
    age: float, cholesterol: float, model: ModelParameters
) -> tuple[bool, float]:
    inputs = (age, cholesterol)
    if len(inputs) != len(model.means):
        raise ValueError("El modelo no espera exactamente dos variables de entrada.")

    margin = model.intercept + sum(
        coefficient * (value - mean) / standard_deviation
        for value, mean, standard_deviation, coefficient in zip(
            inputs,
            model.means,
            model.standard_deviations,
            model.coefficients,
        )
    )
    if margin >= 0:
        probability = 1 / (1 + math.exp(-margin))
    else:
        exp_margin = math.exp(margin)
        probability = exp_margin / (1 + exp_margin)
    return probability >= model.threshold, probability


def main() -> None:
    st.set_page_config(
        page_title="Pulso Vital | Evaluación orientativa",
        page_icon="❤️",
        layout="centered",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(160deg, #0b1220 0%, #1a1724 100%);
                color: #f1f5f9;
            }
            .hero {
                padding: 1.8rem 1rem 1rem;
                text-align: center;
            }
            .hero h1 {
                color: #f8fafc;
                font-size: clamp(2.2rem, 6vw, 3.5rem);
                margin-bottom: 0.4rem;
            }
            .hero p {
                color: #cbd5e1;
                font-size: 1.1rem;
            }
            .footer {
                color: #94a3b8;
                font-size: 0.9rem;
                margin-top: 3rem;
                padding: 1.2rem 0;
                text-align: center;
                border-top: 1px solid #334155;
            }
        </style>
        <div class="hero">
            <h1>❤️ Pulso Vital</h1>
            <p>Una mirada orientativa a tu salud cardiovascular</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "Ingresa la edad y el colesterol total para obtener una estimación del modelo."
    )

    with st.form("cardiac_risk_form"):
        age = st.number_input(
            "Edad (años)",
            min_value=1,
            max_value=120,
            value=45,
            step=1,
        )
        cholesterol = st.number_input(
            "Colesterol total (mg/dL)",
            min_value=50,
            max_value=1000,
            value=200,
            step=1,
        )
        submitted = st.form_submit_button(
            "Consultar estimación", type="primary", width="stretch"
        )

    if submitted:
        model = load_model_parameters()
        possible_risk, probability = predict_risk(age, cholesterol, model)

        with st.container(border=True):
            if possible_risk:
                st.warning(
                    "El modelo detecta una posible señal de riesgo cardiovascular. "
                    "Esta estimación no confirma un ataque cardíaco."
                )
                st.image(
                    str(WARNING_IMAGE_PATH),
                    caption="Posible señal de riesgo: busca orientación médica.",
                    width="stretch",
                )
            else:
                st.success(
                    "Según este modelo, el corazón se clasifica como saludable "
                    "con los datos ingresados."
                )
                st.image(
                    str(HEALTHY_IMAGE_PATH),
                    caption="Resultado orientativo del modelo.",
                    width="stretch",
                )
            st.metric("Probabilidad estimada por el modelo", f"{probability:.1%}")

    st.info(
        "Esta herramienta es solo orientativa: usa únicamente edad y colesterol, "
        "no diagnostica ni reemplaza la evaluación de un profesional de salud. "
        "Ante dolor de pecho, dificultad para respirar u otros síntomas de urgencia, "
        "contacta inmediatamente a los servicios de emergencia."
    )

    st.markdown(
        '<div class="footer">Desarrollado por Santiago Alberto Matheus Quintero</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
