# Pulso Vital

Aplicación Streamlit de estimación cardiovascular orientativa. Utiliza el
modelo de regresión logística incluido en
`mejor_modelo_regresion_logistica.zip` y las imágenes proporcionadas.

## Ejecutar localmente

Se recomienda Python 3.11 o posterior.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Para publicar en Streamlit Community Cloud, sube `app.py`, `requirements.txt`,
`mejor_modelo_regresion_logistica.zip`, `Paciente_Sano.jpg` y
`Posible_Ataque_Cardiaco.jpg` al repositorio y selecciona `app.py` como archivo
principal.

La estimación usa solo edad y colesterol total. No es un diagnóstico médico ni
reemplaza la evaluación de un profesional de salud.
