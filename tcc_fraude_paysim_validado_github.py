# %% [markdown]
# TCC - Detecção de fraudes em transações financeiras
# Comparação entre Regressão Logística e XGBoost
#
# Fluxo da análise:
# - leitura, filtragem e auditoria do PaySim
# - preparação das variáveis e divisão temporal
# - tratamento do desbalanceamento
# - treinamento da Regressão Logística e do XGBoost
# - seleção dos thresholds na validação
# - avaliação no conjunto de teste
# - análise de sensibilidade à prevalência
# - interpretação complementar com SHAP

# %%
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
import joblib

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "dados"
RESULTS_DIR = BASE_DIR / "resultados"
MODELS_DIR = BASE_DIR / "modelos"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Arquivos de entrada e base filtrada.
RAW_FILE = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
FILTERED_FILE = DATA_DIR / "paysim_transfer_cashout.parquet"

# False = execução final; True = teste rápido com amostra de 10%.
MODO_RAPIDO = True
FRACAO_MODO_RAPIDO = 0.10
RANDOM_STATE = 42

# Ativa ou desativa a análise complementar com SHAP.
RODAR_SHAP = True

print("Pasta do projeto:", BASE_DIR)
print("Arquivo esperado:", RAW_FILE)

# %% [markdown]
# ETAPA 1 - LEITURA E FILTRAGEM DA BASE
# Lê o CSV em blocos, calcula a distribuição de fraude e mantém TRANSFER e CASH_OUT.

# %%
USECOLS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud",
]

DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "oldbalanceDest": "float32",
    "isFraud": "int8",
}

if not FILTERED_FILE.exists():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n{RAW_FILE}\n\n"
            "Coloque o CSV do PaySim dentro da pasta 'dados' "
            "ou altere a variável RAW_FILE no início do script."
        )

    print("Lendo a base original em blocos. Isso pode levar alguns minutos...")
    filtered_chunks = []
    summary_chunks = []

    for i, chunk in enumerate(
        pd.read_csv(
            RAW_FILE,
            usecols=USECOLS,
            dtype=DTYPES,
            chunksize=500_000,
        ),
        start=1,
    ):
        resumo = (
            chunk.groupby("type", observed=True)["isFraud"]
            .agg(total="size", fraudes="sum")
            .reset_index()
        )
        summary_chunks.append(resumo)

        filtrado = chunk[chunk["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
        filtered_chunks.append(filtrado)

        print(
            f"Bloco {i:02d}: {len(chunk):,} registros lidos | "
            f"{len(filtrado):,} mantidos"
        )

    resumo_tipo = pd.concat(summary_chunks, ignore_index=True)
    resumo_tipo = (
        resumo_tipo.groupby("type", observed=True)[["total", "fraudes"]]
        .sum()
        .reset_index()
    )

    resumo_tipo["taxa_fraude_pct"] = (
        resumo_tipo["fraudes"] / resumo_tipo["total"] * 100
    )

    resumo_tipo.to_csv(
        RESULTS_DIR / "01_distribuicao_fraude_por_tipo_toda_base.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks

    df["type"] = df["type"].cat.remove_unused_categories()
    df.to_parquet(FILTERED_FILE, index=False)

    print("\nArquivo filtrado salvo em:", FILTERED_FILE)

else:
    print("Arquivo filtrado já existe. Lendo diretamente do Parquet...")
    df = pd.read_parquet(FILTERED_FILE)

print("\nDimensão da base filtrada:", df.shape)
print(df.head())

# %% [markdown]
# ETAPA 2 - AUDITORIA E ANÁLISE EXPLORATÓRIA

# %%
n_total = len(df)
n_fraudes = int(df["isFraud"].sum())
taxa_fraude = n_fraudes / n_total

print(f"Total filtrado: {n_total:,}")
print(f"Fraudes: {n_fraudes:,}")
print(f"Taxa de fraude: {taxa_fraude:.4%}")

resumo_filtrado = (
    df.groupby("type", observed=True)["isFraud"]
    .agg(total="size", fraudes="sum")
    .reset_index()
)

resumo_filtrado["taxa_fraude_pct"] = (
    resumo_filtrado["fraudes"] / resumo_filtrado["total"] * 100
)

print("\nResumo por tipo:")
print(resumo_filtrado)

resumo_filtrado.to_csv(
    RESULTS_DIR / "02_distribuicao_fraude_transfer_cashout.csv",
    index=False,
    encoding="utf-8-sig",
)

resumo_amount = (
    df.groupby("isFraud")["amount"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

resumo_amount.to_csv(
    RESULTS_DIR / "03_estatisticas_amount_por_classe.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nEstatísticas de amount por classe:")
print(resumo_amount)

plt.figure(figsize=(7, 4))
plt.bar(
    resumo_filtrado["type"].astype(str),
    resumo_filtrado["taxa_fraude_pct"],
)
plt.xlabel("Tipo de transação")
plt.ylabel("Taxa de fraude (%)")
plt.title("Taxa de fraude por tipo de transação")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "grafico_taxa_fraude_por_tipo.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ETAPA 3 - AUDITORIA DE VALORES ZERO
# Valores zero são avaliados separadamente de nulos e não são removidos automaticamente.

# %%
print("\n" + "=" * 80)
print("AUDITORIA COMPLETA DE VALORES ZERO")
print("=" * 80)

VARIAVEIS_ZERO = ["amount", "oldbalanceOrg", "oldbalanceDest"]

# Resumo geral
auditoria_zero_geral = pd.DataFrame(
    {
        "variavel": VARIAVEIS_ZERO,
        "total_registros": [len(df)] * len(VARIAVEIS_ZERO),
        "quantidade_zeros": [
            int((df[col] == 0).sum()) for col in VARIAVEIS_ZERO
        ],
    }
)

auditoria_zero_geral["percentual_zeros"] = (
    auditoria_zero_geral["quantidade_zeros"]
    / auditoria_zero_geral["total_registros"]
    * 100
)

print("\n1. RESUMO GERAL")
print(auditoria_zero_geral.to_string(index=False))

auditoria_zero_geral.to_csv(
    RESULTS_DIR / "auditoria_zeros_resumo_geral.csv",
    index=False,
    encoding="utf-8-sig",
)

# Zeros por classe
registros_zero_classe = []

for variavel in VARIAVEIS_ZERO:
    for classe, grupo in df.groupby("isFraud", observed=True):
        total = len(grupo)
        quantidade_zeros = int((grupo[variavel] == 0).sum())

        registros_zero_classe.append(
            {
                "variavel": variavel,
                "isFraud": int(classe),
                "total": total,
                "quantidade_zeros": quantidade_zeros,
                "percentual_zeros": (
                    quantidade_zeros / total * 100 if total else np.nan
                ),
            }
        )

auditoria_zero_classe = pd.DataFrame(registros_zero_classe)

print("\n2. ZEROS POR CLASSE")
print(auditoria_zero_classe.to_string(index=False))

auditoria_zero_classe.to_csv(
    RESULTS_DIR / "auditoria_zeros_por_classe.csv",
    index=False,
    encoding="utf-8-sig",
)

# Zeros por tipo e classe
registros_zero_tipo_classe = []

for variavel in VARIAVEIS_ZERO:
    for (tipo, classe), grupo in df.groupby(
        ["type", "isFraud"],
        observed=True,
    ):
        total = len(grupo)
        quantidade_zeros = int((grupo[variavel] == 0).sum())

        registros_zero_tipo_classe.append(
            {
                "variavel": variavel,
                "type": str(tipo),
                "isFraud": int(classe),
                "total": total,
                "quantidade_zeros": quantidade_zeros,
                "percentual_zeros": (
                    quantidade_zeros / total * 100 if total else np.nan
                ),
            }
        )

auditoria_zero_tipo_classe = pd.DataFrame(
    registros_zero_tipo_classe
)

print("\n3. ZEROS POR TIPO DE TRANSAÇÃO E CLASSE")
print(auditoria_zero_tipo_classe.to_string(index=False))

auditoria_zero_tipo_classe.to_csv(
    RESULTS_DIR / "auditoria_zeros_por_tipo_e_classe.csv",
    index=False,
    encoding="utf-8-sig",
)

# Saldos de origem e destino iguais a zero
saldo_origem_destino_zero = (
    (df["oldbalanceOrg"] == 0)
    & (df["oldbalanceDest"] == 0)
)

quantidade_saldos_zero = int(saldo_origem_destino_zero.sum())
percentual_saldos_zero = (
    quantidade_saldos_zero / len(df) * 100
)

print("\n4. SALDOS DE ORIGEM E DESTINO SIMULTANEAMENTE ZERO")
print(f"Quantidade: {quantidade_saldos_zero:,}")
print(f"Percentual: {percentual_saldos_zero:.6f}%")

auditoria_saldos_zero_classe = (
    pd.DataFrame(
        {
            "isFraud": df["isFraud"],
            "saldos_origem_destino_zero": saldo_origem_destino_zero.astype("int8"),
        }
    )
    .groupby("isFraud")["saldos_origem_destino_zero"]
    .agg(["count", "sum", "mean"])
)

auditoria_saldos_zero_classe["percentual"] = (
    auditoria_saldos_zero_classe["mean"] * 100
)

print("\nPor classe:")
print(auditoria_saldos_zero_classe)

auditoria_saldos_zero_classe.reset_index().to_csv(
    RESULTS_DIR / "auditoria_saldos_origem_destino_zero_por_classe.csv",
    index=False,
    encoding="utf-8-sig",
)

# Fraudes com amount igual a zero
fraudes_amount_zero = df[
    (df["isFraud"] == 1) & (df["amount"] == 0)
].copy()

fraudes_amount_zero.to_csv(
    RESULTS_DIR / "auditoria_fraudes_amount_zero.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nArquivos salvos na pasta resultados.")

# %% [markdown]
# ETAPA 4 - PREPARAÇÃO DAS VARIÁVEIS
# Identificadores, isFlaggedFraud e saldos posteriores não entram na modelagem.
# step é usado apenas para a separação temporal.

# %%
df["type_TRANSFER"] = (
    df["type"].astype(str) == "TRANSFER"
).astype("int8")

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type_TRANSFER",
]

TARGET = "isFraud"

print("\nValores nulos:")
print(
    df[["step"] + FEATURES + [TARGET]]
    .isna()
    .sum()
)

if df[["step"] + FEATURES + [TARGET]].isna().any().any():
    raise ValueError(
        "Foram encontrados valores nulos. "
        "Investigue antes de modelar."
    )

# Validações básicas da base antes da modelagem
if set(df[TARGET].unique()) - {0, 1}:
    raise ValueError("A variável-alvo contém valores diferentes de 0 e 1.")

if not np.isfinite(df[FEATURES].to_numpy(dtype="float64")).all():
    raise ValueError("Foram encontrados valores infinitos nas variáveis explicativas.")

if not set(df["type"].astype(str).unique()).issubset({"TRANSFER", "CASH_OUT"}):
    raise ValueError("A base filtrada contém tipos diferentes de TRANSFER e CASH_OUT.")

# %% [markdown]
# ETAPA 5 - DIVISÃO TEMPORAL
# 70% dos steps para treino, 15% para validação e 15% para teste.
# Os percentuais se referem aos períodos, não ao número de transações.

# %%
steps = np.sort(df["step"].unique())

idx_train = max(1, int(len(steps) * 0.70))
idx_val = max(idx_train + 1, int(len(steps) * 0.85))

train_last_step = steps[idx_train - 1]
val_last_step = steps[idx_val - 1]

train_df = df[
    df["step"] <= train_last_step
].copy()

val_df = df[
    (df["step"] > train_last_step)
    & (df["step"] <= val_last_step)
].copy()

test_df = df[
    df["step"] > val_last_step
].copy()

# Confere se não há sobreposição temporal entre os conjuntos.
if not (
    train_df["step"].max() < val_df["step"].min()
    and val_df["step"].max() < test_df["step"].min()
):
    raise ValueError("Foi detectada sobreposição temporal entre treino, validação e teste.")

print("Último step de treino:", train_last_step)
print("Último step de validação:", val_last_step)


def mostrar_distribuicao(nome, base):
    n = len(base)
    f = int(base[TARGET].sum())
    taxa = f / n if n else np.nan

    print(
        f"{nome:12s} | "
        f"registros={n:,} | "
        f"fraudes={f:,} | "
        f"taxa={taxa:.4%}"
    )


mostrar_distribuicao("Treino", train_df)
mostrar_distribuicao("Validação", val_df)
mostrar_distribuicao("Teste", test_df)

for nome, base in [
    ("Treino", train_df),
    ("Validação", val_df),
    ("Teste", test_df),
]:
    if base[TARGET].nunique() < 2:
        raise ValueError(
            f"O conjunto {nome} não contém as duas classes. "
            "Será necessário rever o corte temporal."
        )

# %% [markdown]
# ETAPA 6 - MODO RÁPIDO
# Usado apenas para testar o fluxo com uma amostra; não compõe os resultados finais.

# %%
if MODO_RAPIDO:
    train_model = train_df.sample(
        frac=FRACAO_MODO_RAPIDO,
        random_state=RANDOM_STATE,
    )

    val_model = val_df.sample(
        frac=FRACAO_MODO_RAPIDO,
        random_state=RANDOM_STATE,
    )

    test_model = test_df.sample(
        frac=FRACAO_MODO_RAPIDO,
        random_state=RANDOM_STATE,
    )

    print("\nATENÇÃO: MODO RÁPIDO ATIVADO.")

else:
    train_model = train_df
    val_model = val_df
    test_model = test_df

mostrar_distribuicao("Treino usado", train_model)
mostrar_distribuicao("Val. usada", val_model)
mostrar_distribuicao("Teste usado", test_model)

X_train = train_model[FEATURES]
y_train = train_model[TARGET].astype(int)

X_val = val_model[FEATURES]
y_val = val_model[TARGET].astype(int)

X_test = test_model[FEATURES]
y_test = test_model[TARGET].astype(int)

# %% [markdown]
# ETAPA 7 - FUNÇÕES DE AVALIAÇÃO
# Seleciona o threshold de maior F1 na validação e calcula as métricas.
# Average Precision [AP] resume a curva Precision-Recall.

# %%
def melhor_threshold_f1(y_true, probas):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probas,
    )

    precision_t = precision[:-1]
    recall_t = recall[:-1]

    f1 = (
        2 * precision_t * recall_t
        / (precision_t + recall_t + 1e-12)
    )

    idx = int(np.nanargmax(f1))

    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision_t[idx]),
        "recall": float(recall_t[idx]),
        "f1": float(f1[idx]),
    }


def calcular_metricas(
    y_true,
    probas,
    threshold,
    modelo,
    conjunto,
):
    pred = (probas >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "modelo": modelo,
        "conjunto": conjunto,
        "threshold": float(threshold),
        "precision": precision_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "average_precision": average_precision_score(
            y_true,
            probas,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probas,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# %% [markdown]
# ETAPA 8 - REGRESSÃO LOGÍSTICA
# Padronização das variáveis e ajuste com pesos balanceados entre as classes.

# %%
modelo_lr = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

print("\nTreinando Regressão Logística...")

modelo_lr.fit(
    X_train,
    y_train,
)

proba_val_lr = modelo_lr.predict_proba(
    X_val
)[:, 1]

threshold_lr = melhor_threshold_f1(
    y_val,
    proba_val_lr,
)

print(
    "\nMelhor threshold LR na validação:",
    threshold_lr,
)

proba_test_lr = modelo_lr.predict_proba(
    X_test
)[:, 1]

# %% [markdown]
# ETAPA 9 - XGBOOST
# O peso da classe positiva é calculado a partir do conjunto de treinamento.

# %%
n_neg = int(
    (y_train == 0).sum()
)

n_pos = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    n_neg / n_pos
)

print(
    "\nscale_pos_weight:",
    scale_pos_weight,
)

modelo_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=100 if MODO_RAPIDO else 300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

print("\nTreinando XGBoost...")

modelo_xgb.fit(
    X_train,
    y_train,
)

proba_val_xgb = modelo_xgb.predict_proba(
    X_val
)[:, 1]

threshold_xgb = melhor_threshold_f1(
    y_val,
    proba_val_xgb,
)

print(
    "\nMelhor threshold XGBoost na validação:",
    threshold_xgb,
)

proba_test_xgb = modelo_xgb.predict_proba(
    X_test
)[:, 1]

# %% [markdown]
# ETAPA 10 - AVALIAÇÃO FINAL NO TESTE
# Compara o threshold padrão (0,50) com o definido na validação.
# Average Precision [AP] é a medida-resumo da curva Precision-Recall.

# %%
resultados = [
    calcular_metricas(
        y_test,
        proba_test_lr,
        0.50,
        "Regressão Logística",
        "Teste - threshold 0.50",
    ),
    calcular_metricas(
        y_test,
        proba_test_lr,
        threshold_lr["threshold"],
        "Regressão Logística",
        "Teste - threshold otimizado",
    ),
    calcular_metricas(
        y_test,
        proba_test_xgb,
        0.50,
        "XGBoost",
        "Teste - threshold 0.50",
    ),
    calcular_metricas(
        y_test,
        proba_test_xgb,
        threshold_xgb["threshold"],
        "XGBoost",
        "Teste - threshold otimizado",
    ),
]

metricas_df = pd.DataFrame(
    resultados
)

print("\nRESULTADOS NO TESTE:")
print(
    metricas_df.to_string(
        index=False
    )
)

sufixo = (
    "_MODO_RAPIDO"
    if MODO_RAPIDO
    else "_FINAL"
)

metricas_df.to_csv(
    RESULTS_DIR / f"04_metricas_modelos{sufixo}.csv",
    index=False,
    encoding="utf-8-sig",
)

# %% [markdown]
# ETAPA 11 - ANÁLISE DE SENSIBILIDADE À PREVALÊNCIA
# Estima a precisão para a prevalência observada no treino,
# mantendo recall e FPR medidos no teste. Não representa um novo teste empírico.

# %%
def precisao_para_prevalencia(y_true, probas, threshold, prevalencia_alvo):
    if not 0 < prevalencia_alvo < 1:
        raise ValueError("A prevalência-alvo deve estar entre 0 e 1.")

    pred = (probas >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    denominador = (
        recall * prevalencia_alvo
        + fpr * (1 - prevalencia_alvo)
    )

    precisao_estimada = (
        recall * prevalencia_alvo / denominador
        if denominador > 0
        else np.nan
    )

    return {
        "prevalencia_alvo": float(prevalencia_alvo),
        "recall_teste": float(recall),
        "fpr_teste": float(fpr),
        "precisao_estimada": float(precisao_estimada),
    }


prevalencia_treino = float(y_train.mean())

sensibilidade_lr = precisao_para_prevalencia(
    y_test,
    proba_test_lr,
    threshold_lr["threshold"],
    prevalencia_treino,
)

sensibilidade_xgb = precisao_para_prevalencia(
    y_test,
    proba_test_xgb,
    threshold_xgb["threshold"],
    prevalencia_treino,
)

sensibilidade_prevalencia_df = pd.DataFrame(
    [
        {"modelo": "Regressão Logística", **sensibilidade_lr},
        {"modelo": "XGBoost", **sensibilidade_xgb},
    ]
)

print("\nANÁLISE DE SENSIBILIDADE À PREVALÊNCIA")
print(f"Prevalência de fraude no treinamento: {prevalencia_treino:.6%}")
print(sensibilidade_prevalencia_df.to_string(index=False))

sensibilidade_prevalencia_df.to_csv(
    RESULTS_DIR / f"07_sensibilidade_prevalencia{sufixo}.csv",
    index=False,
    encoding="utf-8-sig",
)

# Checagens finais das métricas e thresholds.
if not metricas_df[[
    "precision", "recall", "f1_score", "average_precision", "roc_auc"
]].apply(np.isfinite).all().all():
    raise ValueError("Há métricas finais não finitas. Revise a execução.")

for nome_modelo, info_threshold in [
    ("Regressão Logística", threshold_lr),
    ("XGBoost", threshold_xgb),
]:
    if not 0 <= info_threshold["threshold"] <= 1:
        raise ValueError(f"Threshold inválido para {nome_modelo}.")

validacoes_execucao = pd.DataFrame(
    [
        {"validacao": "modo_final", "valor": str(not MODO_RAPIDO)},
        {"validacao": "shap_ativado", "valor": str(RODAR_SHAP)},
        {"validacao": "prevalencia_treino", "valor": prevalencia_treino},
        {"validacao": "train_last_step", "valor": int(train_last_step)},
        {"validacao": "val_last_step", "valor": int(val_last_step)},
        {"validacao": "n_treino", "valor": len(y_train)},
        {"validacao": "n_validacao", "valor": len(y_val)},
        {"validacao": "n_teste", "valor": len(y_test)},
        {"validacao": "threshold_lr", "valor": threshold_lr["threshold"]},
        {"validacao": "threshold_xgb", "valor": threshold_xgb["threshold"]},
    ]
)

validacoes_execucao.to_csv(
    RESULTS_DIR / f"00_validacoes_execucao{sufixo}.csv",
    index=False,
    encoding="utf-8-sig",
)

# %% [markdown]
# ETAPA 12 - MATRIZES DE CONFUSÃO

# %%
def salvar_matriz_confusao(
    y_true,
    probas,
    threshold,
    titulo,
    nome_arquivo,
):
    pred = (
        probas >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legítima",
            "Fraude",
        ],
    )

    disp.plot(
        values_format="d"
    )

    plt.title(
        titulo
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / nome_arquivo,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


salvar_matriz_confusao(
    y_test,
    proba_test_lr,
    threshold_lr["threshold"],
    "Regressão Logística - Matriz de Confusão",
    f"matriz_confusao_lr{sufixo}.png",
)

salvar_matriz_confusao(
    y_test,
    proba_test_xgb,
    threshold_xgb["threshold"],
    "XGBoost - Matriz de Confusão",
    f"matriz_confusao_xgb{sufixo}.png",
)

# %% [markdown]
# ETAPA 13 - CURVAS PRECISION-RECALL
# A legenda apresenta Average Precision [AP].

# %%
prec_lr, rec_lr, _ = precision_recall_curve(
    y_test,
    proba_test_lr,
)

prec_xgb, rec_xgb, _ = precision_recall_curve(
    y_test,
    proba_test_xgb,
)

ap_lr = average_precision_score(
    y_test,
    proba_test_lr,
)

ap_xgb = average_precision_score(
    y_test,
    proba_test_xgb,
)

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    rec_lr,
    prec_lr,
    label=(
        "Regressão Logística "
        f"(AP={ap_lr:.4f})"
    ),
)

plt.plot(
    rec_xgb,
    prec_xgb,
    label=(
        "XGBoost "
        f"(AP={ap_xgb:.4f})"
    ),
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Curva Precision-Recall"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / f"curva_precision_recall{sufixo}.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()

# %% [markdown]
# ETAPA 14 - INTERPRETAÇÃO INICIAL DAS VARIÁVEIS
# Coeficientes da Regressão Logística e importância das variáveis no XGBoost.

# %%
coef_lr = (
    modelo_lr.named_steps["model"]
    .coef_[0]
)

importancia_lr = pd.DataFrame(
    {
        "variavel": FEATURES,
        "coeficiente_padronizado": coef_lr,
        "coeficiente_abs": np.abs(
            coef_lr
        ),
    }
).sort_values(
    "coeficiente_abs",
    ascending=False,
)

print(
    "\nCoeficientes - Regressão Logística:"
)

print(
    importancia_lr
)

importancia_lr.to_csv(
    RESULTS_DIR / f"05_coeficientes_regressao_logistica{sufixo}.csv",
    index=False,
    encoding="utf-8-sig",
)

importancia_xgb = pd.DataFrame(
    {
        "variavel": FEATURES,
        "feature_importance": modelo_xgb.feature_importances_,
    }
).sort_values(
    "feature_importance",
    ascending=False,
)

print(
    "\nFeature importance - XGBoost:"
)

print(
    importancia_xgb
)

importancia_xgb.to_csv(
    RESULTS_DIR / f"06_feature_importance_xgboost{sufixo}.csv",
    index=False,
    encoding="utf-8-sig",
)

# %% [markdown]
# ETAPA 15 - SALVAR MODELOS
# Salva os modelos ajustados para manter a rastreabilidade da execução.

# %%
joblib.dump(
    modelo_lr,
    MODELS_DIR / f"modelo_regressao_logistica{sufixo}.joblib",
)

joblib.dump(
    modelo_xgb,
    MODELS_DIR / f"modelo_xgboost{sufixo}.joblib",
)

print(
    "\nModelos salvos na pasta:",
    MODELS_DIR,
)

# %% [markdown]
# ETAPA 16 - SHAP
# Análise complementar de interpretabilidade; não interfere no treino nem nos thresholds.
# Usa até 5.000 registros do teste e, na Regressão Logística, 100 registros de treino como referência.

# %%
if RODAR_SHAP:
    import shap

    print(
        "\nIniciando análise complementar SHAP..."
    )

    n_shap = min(
        5000,
        len(X_test),
    )

    X_shap = X_test.sample(
        n=n_shap,
        random_state=RANDOM_STATE,
    )

    # XGBoost
    explainer_xgb = shap.TreeExplainer(
        modelo_xgb
    )

    shap_values_xgb = explainer_xgb(
        X_shap
    )

    shap.summary_plot(
        shap_values_xgb,
        X_shap,
        show=False,
    )

    plt.tight_layout()

    arquivo_shap_xgb = (
        RESULTS_DIR
        / f"shap_xgb{sufixo}.png"
    )

    print(
        "Salvando SHAP do XGBoost em:",
        arquivo_shap_xgb,
    )

    plt.savefig(
        str(arquivo_shap_xgb),
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    # Regressão Logística
    scaler = (
        modelo_lr.named_steps["scaler"]
    )

    lr_estimator = (
        modelo_lr.named_steps["model"]
    )

    X_shap_scaled = scaler.transform(
        X_shap
    )

    n_background = min(
        100,
        len(X_train),
    )

    X_background = X_train.sample(
        n=n_background,
        random_state=RANDOM_STATE,
    )

    X_background_scaled = scaler.transform(
        X_background
    )

    explainer_lr = shap.LinearExplainer(
        lr_estimator,
        X_background_scaled,
    )

    shap_values_lr = explainer_lr(
        X_shap_scaled
    )

    shap.summary_plot(
        shap_values_lr.values,
        X_shap_scaled,
        feature_names=FEATURES,
        show=False,
    )

    plt.tight_layout()

    arquivo_shap_lr = (
        RESULTS_DIR
        / f"shap_lr{sufixo}.png"
    )

    print(
        "Salvando SHAP da Regressão Logística em:",
        arquivo_shap_lr,
    )

    plt.savefig(
        str(arquivo_shap_lr),
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(
        "Análises SHAP concluídas."
    )

else:
    print(
        "\nSHAP não executado "
        "(RODAR_SHAP=False)."
    )

print(
    "\nFIM DO SCRIPT."
)
