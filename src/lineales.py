"""Utilidades de Pardo para modelos lineales (Sprint 2). No modifica el contrato de Joan.

``PrepLineal`` envuelve ``build_preprocessor(escalar=True, ...)`` y le añade, fuera de la función,
una codificación extra de la hora (One-Hot o splines periódicos), tal como pide el traspaso del Sprint 2.
Todo lo que aprende (escalas, categorías, medianas) lo aprende en ``fit`` con los datos que recibe (train del fold).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from src.features import build_preprocessor

CODIF_HORA = ("entero", "sin_cos", "onehot", "splines")


def encoder_hora(codif: str, n_knots: int = 13):
    """One-Hot de las 24 horas o B-splines cúbicos periódicos (periodo 24 h).

    Con ``n_knots`` nudos equiespaciados en [0, 24] y ``extrapolation='periodic'`` salen ``n_knots - 1`` columnas.
    """
    if codif == "onehot":
        return OneHotEncoder(categories=[list(range(24))], sparse_output=False, handle_unknown="ignore")
    if codif == "splines":
        return SplineTransformer(degree=3, n_knots=n_knots, knots=np.linspace(0, 24, n_knots).reshape(-1, 1),
                                 extrapolation="periodic")
    raise ValueError(codif)


class PrepLineal(BaseEstimator, TransformerMixin):
    """Preprocesador para Ridge/Lasso.

    codif_hora: ``'entero'`` | ``'sin_cos'`` (pasan a ``build_preprocessor``) | ``'onehot'`` | ``'splines'``
        (la hora sale de ``build_preprocessor`` como sin/cos y se le añade la codificación extra).
    con_hr_workingday: si ``False``, quita las columnas One-Hot de ``hr_workingday`` del contrato, para medir
        la codificación de la hora por sí sola.
    interaccion_workingday: si ``True``, multiplica cada columna de la codificación extra por ``workingday``
        (perfil horario distinto en laborables, como en el ejemplo de scikit-learn).
    interaccion_temp: si ``True``, multiplica cada columna de la codificación extra por ``temp_c`` ya escalada
        (el efecto de la temperatura cambia según la hora).
    lags: columnas ``lag_<h>`` de ``agregar_lags``; sus NaN se imputan con la mediana de train del fold.
    """

    def __init__(self, codif_hora="entero", con_hr_workingday=True, interaccion_workingday=False,
                 interaccion_temp=False, n_knots=13, lags=()):
        self.codif_hora = codif_hora
        self.con_hr_workingday = con_hr_workingday
        self.interaccion_workingday = interaccion_workingday
        self.interaccion_temp = interaccion_temp
        self.n_knots = n_knots
        self.lags = lags

    def _hora_base(self):
        return self.codif_hora if self.codif_hora in ("entero", "sin_cos") else "sin_cos"

    def fit(self, X, y=None):
        assert self.codif_hora in CODIF_HORA, f"codif_hora debe ser una de {CODIF_HORA}"
        self.base_ = build_preprocessor(escalar=True, hora=self._hora_base(), lags=tuple(self.lags)).fit(X, y)
        self.extra_ = None
        if self.codif_hora in ("onehot", "splines"):
            self.extra_ = encoder_hora(self.codif_hora, self.n_knots).fit(X[["hr"]])
        if self.lags:
            Z = self.base_.transform(X)
            self.medianas_lags_ = Z[list(self.lags)].median()
        return self

    def transform(self, X):
        Z = self.base_.transform(X)
        if not self.con_hr_workingday:
            Z = Z.loc[:, ~Z.columns.str.startswith("hr_workingday")]
        if self.lags:
            Z[list(self.lags)] = Z[list(self.lags)].fillna(self.medianas_lags_)
        if self.extra_ is not None:
            H = np.asarray(self.extra_.transform(X[["hr"]]), dtype=float)
            nombres = [f"hr_{self.codif_hora}_{i}" for i in range(H.shape[1])]
            partes = [pd.DataFrame(H, index=Z.index, columns=nombres)]
            if self.interaccion_workingday:
                wd = X["workingday"].to_numpy(dtype=float)[:, None]
                partes.append(pd.DataFrame(H * wd, index=Z.index, columns=[f"{n}_x_wd" for n in nombres]))
            if self.interaccion_temp:
                t = Z["temp_c"].to_numpy(dtype=float)[:, None]
                partes.append(pd.DataFrame(H * t, index=Z.index, columns=[f"{n}_x_temp" for n in nombres]))
            Z = pd.concat([Z, *partes], axis=1)
        assert not Z.isna().any().any(), "quedan NaN en la entrada del modelo lineal"
        return Z
