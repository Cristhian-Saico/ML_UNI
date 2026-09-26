"""Lectura y reglas deterministas: no estiman parámetros sobre la muestra."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PESADOS = ['VEH_PESADOS__2E', 'VEH_PESADOS_3E', 'VEH_PESADOS_4E',
           'VEH_PESADOS_5E', 'VEH_PESADOS_6E', 'VEH_PESADOS_7E']
FEATURES = ['VEH_LIGEROS_AUTOMOVILES'] + PESADOS
RESUMEN = ['VEH_LIGEROS_TOTAL', 'VEH_PESADOS_TOTAL', 'VEH_TOTAL',
           'VEH_LIGEROS_IMD', 'VEH_PESADOS_IMD', 'VEH_IMD']
ADULT_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

def root_path():
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / 'src/data.py').is_file() and (p / 'notebooks').is_dir():
            return p
    raise FileNotFoundError('Ejecuta desde la raíz del proyecto o notebooks/.')

def read_flujo(root):
    p = Path(root) / 'data/raw/Flujo_vehicular.csv'
    if not p.exists():
        raise FileNotFoundError(f'Falta {p}. Copia el CSV original del curso; los HTML no lo incluyen.')
    df = pd.read_csv(p, encoding='latin-1')
    required = FEATURES + RESUMEN + ['ANIO', 'MES', 'ADMINIST', 'DEPARTAMENTO', 'NOMBRE_PEAJE']
    missing = sorted(set(required) - set(df))
    if missing:
        raise ValueError(f'El esquema no coincide con el HTML. Faltan: {missing}')
    return df

class NormalizeText(BaseEstimator, TransformerMixin):
    """Regla fija: espacios, placeholders y mapa de departamentos del material."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        result = X.copy()
        for c in result.select_dtypes(include=['object', 'string']).columns:
            result[c] = result[c].map(lambda v: v.strip() if isinstance(v, str) else v)
            result[c] = result[c].replace({'': np.nan, '?': np.nan})
        if 'DEPARTAMENTO' in result:
            result['DEPARTAMENTO'] = result['DEPARTAMENTO'].replace(
                {'San Martin': 'San Martín', 'Apurimac': 'Apurímac'})
        return result
