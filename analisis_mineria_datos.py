"""Práctica 1 - Minería de Datos - UEA
Caso: clasificación automática de dígitos manuscritos para digitalización documental.
Autora: Jessica Alexandra Vicente Vicente
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

D = load_digits(as_frame=True)
df = D.frame.rename(columns={"target": "digito"}).copy()
pixeles = [c for c in df.columns if c != "digito"]

# Feature engineering
df["intensidad_total"] = df[pixeles].sum(axis=1)
df["pixeles_activos"] = (df[pixeles] > 0).sum(axis=1)

# Calidad de datos
print("Dimensiones:", df.shape)
print("Nulos (%):\n", (df.isna().mean() * 100).round(2))
print("Duplicados:", df.duplicated().sum())
q1, q3 = df["intensidad_total"].quantile([.25, .75])
iqr = q3 - q1
atipicos = ((df["intensidad_total"] < q1 - 1.5 * iqr) |
            (df["intensidad_total"] > q3 + 1.5 * iqr)).sum()
print("Atípicos IQR:", atipicos)

# Guardar dataset procesado para evidencia y reproducibilidad
df.to_csv("digits_procesado.csv", index=False)

X = df[pixeles + ["intensidad_total", "pixeles_activos"]]
y = df["digito"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y
)

modelos = {
    "Regresión logística": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "Árbol de decisión": DecisionTreeClassifier(
        max_depth=12, min_samples_split=4, random_state=42
    ),
    "Bosque aleatorio": RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = []

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    val = cross_validate(
        modelo, X_train, y_train, cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
        n_jobs=-1
    )
    resultados.append({
        "Modelo": nombre,
        "Accuracy prueba": accuracy_score(y_test, pred),
        "F1 macro prueba": f1_score(y_test, pred, average="macro"),
        "Precisión macro": precision_score(y_test, pred, average="macro"),
        "Recall macro": recall_score(y_test, pred, average="macro"),
        "Accuracy CV (media)": val["test_accuracy"].mean(),
        "F1 CV (media)": val["test_f1"].mean(),
        "Desv. CV accuracy": val["test_accuracy"].std()
    })

tabla = pd.DataFrame(resultados).sort_values(
    "F1 macro prueba", ascending=False
)
tabla.to_csv("resultados_modelos.csv", index=False)
print("\nResultados comparativos:\n", tabla.to_string(index=False))
