"""Imputación comparable y preprocesamiento ajustado solo con entrenamiento."""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data import NormalizeText

# Todos los métodos reciben exactamente la misma matriz y la misma máscara.
# La escala se aprende con valores visibles, nunca con los valores ocultados.
def compare_imputers(X_train, column, seed=42, fraction=0.15):
    if not X_train.index.is_unique:
        raise ValueError('Se requiere índice único para alinear máscara y verdad.')
    if X_train.isna().all().any():
        raise ValueError('Hay una columna completamente vacía en train; revisar antes de imputar.')
    observed = X_train.index[X_train[column].notna()]
    if len(observed) < 8:
        raise ValueError('Insuficientes valores observados para evaluar recuperación.')
    rng = np.random.RandomState(seed)
    hidden = rng.choice(observed, size=max(1, int(len(observed)*fraction)), replace=False)
    masked = X_train.copy()
    truth = X_train.loc[hidden, column].copy()
    masked.loc[hidden, column] = np.nan
    candidates = {
        'Mediana': (None, SimpleImputer(strategy='median')),
        'KNN sin escala': (None, KNNImputer(n_neighbors=5)),
        'KNN estandarizado': (StandardScaler(), KNNImputer(n_neighbors=5)),
        'Iterativa estandarizada': (StandardScaler(), IterativeImputer(random_state=seed, max_iter=20)),
    }
    results, recovered = [], {}
    for name, (scaler, imputer) in candidates.items():
        arr = masked if scaler is None else scaler.fit_transform(masked)
        arr = imputer.fit_transform(arr)
        if scaler is not None:
            arr = scaler.inverse_transform(arr)
        frame = pd.DataFrame(arr, index=masked.index, columns=masked.columns)
        predicted = frame.loc[hidden, column]
        results.append({'metodo': name, 'n_ocultos': len(hidden),
                        'RMSE': float(np.sqrt(mean_squared_error(truth, predicted))),
                        'MAE': float(mean_absolute_error(truth, predicted)),
                        'predicciones_negativas': int((predicted < 0).sum())})
        recovered[name] = frame
    return pd.DataFrame(results).sort_values('RMSE'), truth, recovered

def build_preprocessor(num_cols, cat_cols, method='Mediana'):
    if method == 'KNN sin escala':
        num = Pipeline([('impute', KNNImputer(n_neighbors=5)), ('scale', StandardScaler())])
    elif method == 'KNN estandarizado':
        num = Pipeline([('scale', StandardScaler()), ('impute', KNNImputer(n_neighbors=5))])
    elif method == 'Iterativa estandarizada':
        num = Pipeline([('scale', StandardScaler()),
                        ('impute', IterativeImputer(random_state=42, max_iter=20))])
    else:
        num = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    columns = ColumnTransformer([('num', num, num_cols), ('cat', cat, cat_cols)], remainder='drop',
                                verbose_feature_names_out=True)
    return Pipeline([('normalize', NormalizeText()), ('columns', columns)])
