import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================
# FUNÇÃO DE DADOS
# =========================
def load_and_prepare_data(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )


# =========================
# FUNÇÃO DE TREINO + MLflow
# =========================
def train_and_evaluate(model, X_train, X_test, y_train, y_test, run_name):
    with mlflow.start_run(run_name=run_name):

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(run_name)
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1:", f1)
        print("-"*40)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    path = r"C:\Users\fernanda\Documents\Projetos Pessoal\trabalhos pós\pd cluster 3\archive (3)\creditcard.csv"
    X_train, X_test, y_train, y_test = load_and_prepare_data(path)

    # =========================
    # MODELOS SEM REDUÇÃO
    # =========================
    train_and_evaluate(
        DecisionTreeClassifier(random_state=42),
        X_train, X_test, y_train, y_test,
        run_name="DecisionTree - Sem Redução"
    )

    train_and_evaluate(
        RandomForestClassifier(random_state=42),
        X_train, X_test, y_train, y_test,
        run_name="RandomForest - Sem Redução"
    )

    # =========================
    # RANDOMFOREST + PCA (PIPELINE)
    # =========================
    pipeline_rf_pca = Pipeline([
        ("pca", PCA(n_components=10, random_state=42)),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    train_and_evaluate(
        pipeline_rf_pca,
        X_train, X_test, y_train, y_test,
        run_name="RandomForest + PCA"
    )

    # =========================
    # MODELOS COM LDA
    # =========================
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    train_and_evaluate(
        DecisionTreeClassifier(random_state=42),
        X_train_lda, X_test_lda, y_train, y_test,
        run_name="DecisionTree + LDA"
    )

    train_and_evaluate(
        RandomForestClassifier(random_state=42),
        X_train_lda, X_test_lda, y_train, y_test,
        run_name="RandomForest + LDA"
    )