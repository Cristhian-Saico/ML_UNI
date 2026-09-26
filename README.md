# Actividades resueltas — Semanas 1 y 2

Material: los cuatro HTML suministrados del curso Machine Learning, IEEE CIS UNI. Datos: `Flujo_vehicular.csv` y `Telco_customer_churn.xlsx` (hoja `Telco_Churn`).

## Leer los resultados

Abre `reports/Informe_actividades_ML.html` en un navegador. Es autónomo: tablas, gráficos, respuestas y código desplegable están incluidos, sin necesidad de internet. Los resultados estructurados están en `reports/resultados_*.json`.

## Notebooks incluidos

1. `notebooks/01_semana1_peajes.ipynb`: práctica pendiente completada con los datos originales, EDA, faltantes, imputación y pipeline.
2. `notebooks/02_semana2_peajes.ipynb`: todas las actividades de ambas clases de semana 2, con resultados reales y sensibilidad de detectores.
3. `notebooks/03_semana2_telco.ipynb`: aplicación a la versión Excel suministrada, comparación de imputación de Semana 1 y actividades de Semana 2.

Los tres conservan salidas y gráficas de ejecución. Adult Income no está incluido porque aún falta `adult.data`. Telco estaba propuesto en la introducción: esta aplicación adicional no sustituye por sí misma el requisito original de elegir un dataset distinto al de clase.

## Reproducir

Usa Python 3.11 o superior. La comprobación se realizó con Python 3.12.14.

1. Copia tus archivos originales en `data/raw/Flujo_vehicular.csv` y `data/raw/Telco_customer_churn.xlsx`.
2. Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyter
python -m jupyter notebook
```

En fish, activa con `source .venv/bin/activate.fish`. En Windows: `.venv\Scripts\activate`.

Abre el notebook, selecciona el entorno y ejecuta todas las celdas. No hay rutas absolutas en los notebooks: funcionan desde la raíz o desde `notebooks/`.

La comparación numérica está congelada con las versiones de `requirements.txt`. Jupyter es una dependencia de interfaz adicional. Puedes registrar todo el entorno instalado con `python -m pip freeze > requirements-lock.txt`.

## Datos y Git

Los datos originales no se incluyen nuevamente en el ZIP. Copia los adjuntos que ya tienes. No se modificaron los archivos fuente. Las reglas de `.gitignore` excluyen datos y modelos, y mantienen las carpetas mediante `.gitkeep`.

El ZIP es un proyecto local. No se ha creado un repositorio remoto ni se ha verificado una cuenta GitHub. Publica desde tu cuenta cuando hayas revisado el contenido y registra cambios reales, sin simular un historial anterior. Antes de publicar:

```bash
git status
git check-ignore data/raw/Flujo_vehicular.csv data/raw/Telco_customer_churn.xlsx
git log --oneline
```

Si quieres evitar versionar outputs grandes del notebook, usa una revisión separada o una herramienta para limpiar salidas; los notebooks de esta entrega sí llevan resultados para facilitar la revisión del docente.

## Método y límites

- Una partición base aleatoria por dataset: peajes 80/20 con semilla 42; Telco 80/20 estratificada por Churn Label.
- Todos los parámetros aprendidos para test se ajustan con train: imputación, escalado, lambdas, IQR y p99.
- El EDA global describe el archivo; ranking de columnas y selección de parámetros posteriores se hace en train.
- La recuperación de valores ocultados es una evaluación didáctica interna. No demuestra recuperación de faltantes reales con otro mecanismo.
- Los resúmenes de peajes con total cero se convierten en una copia según el supuesto del curso; la bandera no prueba falta de reporte.
- Se comparan detectores sobre la misma población cuando se informa su intersección. Los datos incompletos quedan sin evaluar, no como normales.
- LOF advirtió sobre perfiles repetidos en peajes. Se conserva la advertencia y se realiza una sensibilidad excluyendo totales cero del subconjunto de detección, sin borrar las filas originales.
- Se conserva el valor original; la columna capada es demostrativa y la decisión principal es marcar para revisión.
- La transformación y el número de detecciones no se interpretan como mejora predictiva o exactitud sin etiquetas verificadas.
- El split aleatorio no es una validación de pronóstico temporal ni de generalización a nuevos peajes.
- Se excluyen de Telco Churn Value, Churn Reason y Churn Score. CLTV queda fuera hasta verificar cómo/cuándo se calculó.

## Modelos y módulos

`src/data.py` contiene carga y normalización; `src/features.py` compara imputadores y construye el pipeline de semana 1.

Los `.joblib` están incluidos en el paquete de entrega, pero ignorados por Git:

- `preprocesador_flujo.joblib`: pipeline de semana 1 para siete numéricas y dos categóricas; incorpora normalización de texto.
- `preprocesador_semana2_peajes.joblib` y `preprocesador_semana2_telco.joblib`: ColumnTransformer sobre entradas ya limpiadas según su notebook. No reciben los archivos crudos directamente: primero aplicar las reglas deterministas de carga, espacios y conversión numérica.
- `detector_peajes.joblib` y `detector_telco.joblib`: diccionario con scaler, Isolation Forest, nombres de features, p99 y vallas IQR. El detector se ajustó con casos completos y requiere validar ese esquema en entradas nuevas.

No son APIs listas para producción. Son objetos ajustados para el ejercicio, con su contrato de entrada visible en el código.

## Verificación realizada

Se ejecutaron 41 celdas de cálculo sin errores sobre los datos reales. Se capturaron salidas de tablas y gráficos en notebooks estándar sin iniciar un servidor Jupyter; se verificaron sus conteos de ejecución y ausencia de salidas de error. Los cálculos de formas, particiones e índices incluyen comprobaciones en código. Se inspeccionaron gráficos exportados. No se comprobó una interfaz Jupyter interactiva ni se obtuvo una captura del HTML en navegador, al no estar instalado un navegador local.

Los hashes SHA-256 de ambos datos están en los JSON de resultados. Diferentes copias, versiones de dependencias o protocolos de partición pueden producir cifras distintas de las salidas ilustrativas del material.
