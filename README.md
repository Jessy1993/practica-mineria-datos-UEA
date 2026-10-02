# Práctica 1 - Minería de Datos (UEA)

**Caso:** Clasificación automática de dígitos manuscritos para apoyar la digitalización documental empresarial.

**Autora:** Jessica Alexandra Vicente Vicente  
**Carrera:** Tecnologías de la Información (TICS) - sexto semestre  
**Asignatura:** Minería de Datos

## Objetivo
Aplicar un proceso completo de minería de datos: exploración, control de calidad, preprocesamiento, feature engineering, modelado, evaluación y validación cruzada.

## Datos
Se utiliza el conjunto **Digits** incluido en scikit-learn, con 1.797 registros de imágenes de dígitos manuscritos. El script genera automáticamente `digits_procesado.csv` con las 64 variables de píxeles, la clase objetivo y dos variables derivadas.

Fuente oficial: scikit-learn, `load_digits`.

## Modelos implementados
- Regresión logística
- Árbol de decisión
- Bosque aleatorio

Se usa una partición estratificada 80/20 y validación cruzada estratificada de 5 particiones.

## Ejecución
```bash
pip install -r requirements.txt
python analisis_mineria_datos.py
```

Al ejecutarse se generan `digits_procesado.csv` y `resultados_modelos.csv`.

## Resultados principales
La regresión logística obtuvo 97,78 % de accuracy en prueba; el bosque aleatorio, 97,22 %; y el árbol de decisión, 80,83 %.

## Reproducibilidad
Se utiliza `random_state=42` en la partición, la validación y los modelos que admiten este parámetro.
