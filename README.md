# 🎓 Passos Mágicos — Tech Challenge 5
> Análise preditiva longitudinal de risco acadêmico para apoio à tomada de decisão pedagógica.

---

## 📌 Visão Geral

Este projeto realiza o ciclo completo de ciência de dados sobre os dados da ONG **Passos Mágicos** (2022–2024): desde o tratamento e unificação das bases brutas até a modelagem preditiva de risco futuro dos alunos, com exportação de pipeline para produção via Streamlit.

O objetivo central é **identificar, com antecedência de um ano (T+1), quais alunos apresentam risco de regressão acadêmica** — queda no INDE ou piora na defasagem de nível — permitindo intervenção preventiva pela equipe pedagógica e psicossocial.

---

## 🗂️ Estrutura do Projeto

```
├── notebook_v2.ipynb           # Notebook principal (EDA + ML)
├── base.xlsx                   # Base bruta original (PEDE 2022, 2023, 2024)
├── base_unificada.xlsx         # Base tratada e persistida (output do pré-processamento)
└── deploy/
    └── model_artifacts/
        ├── pipeline.joblib         # Pipeline treinado (pré-processamento + modelo)
        └── features_metadata.json  # Lista de features esperadas pelo modelo
```

---

## ⚙️ Instalação e Dependências

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy shap joblib openpyxl
```

**Python:** 3.9+

---

## 🔄 Pipeline de Dados

### 1. Leitura e Concatenação
- Leitura das abas `PEDE2022`, `PEDE2023` e `PEDE2024` do arquivo `base.xlsx`
- Padronização de colunas via normalização Unicode + regex
- Concatenação longitudinal com coluna `ano_coleta`

### 2. Limpeza e Transformação
| Etapa | Descrição |
|---|---|
| Remoção de colunas | Variáveis administrativas, redundantes e com data leakage removidas |
| Normalização morfológica | `fase` e `fase_ideal` extraídas via regex; `defasagem` preservada da fonte oficial |
| Harmonização categórica | `genero`, `pedra` e `instituicao_de_ensino` unificados em vocabulário único |
| Correção numérica | Notas clipadas entre 0–10; `ing` e `ipp` com ausência tratada como `NaN` |
| Pruning dinâmico | Alunos com >40% de dados faltantes ou sem INDE removidos |
| Normalização de idade | Correção de anomalias do parser do Excel para datas de nascimento |

**Base final:** 2.845 registros × 24 colunas

---

## 📊 Análise Exploratória (EDA)

9 perguntas analíticas investigadas com visualizações e testes estatísticos:

| # | Pergunta | Método |
|---|---|---|
| 3.1 | Perfil de defasagem (IAN) ao longo dos anos | Countplot + Boxplot |
| 3.2 | Evolução do desempenho acadêmico (IDA) por fase | Lineplot longitudinal |
| 3.3 | Relação engajamento (IEG) × desempenho (IDA) e IPV | Scatterplot + Correlação |
| 3.4 | Coerência da autoavaliação (IAA) com desempenho real | Scatterplot + Correlação |
| 3.5 | IPS como preditor antecedente de quedas | Boxplot + Teste Mann-Whitney |
| 3.6 | Alinhamento IPP × defasagem (IAN) | Boxplot por grupo |
| 3.7 | Drivers do Ponto de Virada (IPV) | Random Forest Regressor |
| 3.8 | Combinações de indicadores que elevam o INDE | Random Forest Regressor |
| 3.9 | Efetividade do programa por estágio (Pedras) | Boxplot por categoria |

---

## 🤖 Modelagem Preditiva

### Target
```python
risco_futuro = (inde_proximo_ano < inde) | (defasagem_proximo_ano < defasagem)
```
**Classe 1** = aluno apresenta regressão acadêmica no ano seguinte (T+1).

### Features
- **Base (9):** `iaa`, `ieg`, `ips`, `ida`, `ipp`, `ipv`, `ian`, `idade_aprox`, `fase`
- **Deltas T vs T-1 (8):** variação anual de cada indicador
- **Categóricas (2):** `instituicao_de_ensino`, `genero`

### Estratégia de Treino/Teste
- Dados de **2023** usados exclusivamente (únicos com deltas de 2022 e target de 2024)
- Split estratificado: **70% treino / 30% teste**
- `RandomizedSearchCV` com validação cruzada (3-fold) para seleção de hiperparâmetros

### Modelos Avaliados
| Modelo | ROC-AUC |
|---|---|
| Gradient Boosting | ~0.694 |
| Random Forest | ~0.675 |
| Regressão Logística | — |

### Pré-processamento no Pipeline
- Numéricas: `SimpleImputer (median)` → `StandardScaler`
- Categóricas: `SimpleImputer (most_frequent)` → `OneHotEncoder`

---

## 📈 Interpretabilidade (SHAP)

O modelo é interpretado via **SHAP values** (TreeExplainer), revelando:

- **IAN e IAA** com maior impacto individual nas previsões
- Valores altos de IAN/IAA → SHAP positivo (mais risco) — alunos em posição mais elevada têm maior espaço para regressão relativa
- **Variáveis de evolução (deltas)** capturam tendências de mudança com relevância preditiva
- **Gênero e tipo de ensino** têm impacto mínimo no modelo

---

## 🚀 Deploy

O pipeline treinado é exportado via `joblib` e pode ser consumido diretamente:

```python
import joblib
import pandas as pd

pipeline = joblib.load('deploy/model_artifacts/pipeline.joblib')

# X_novo deve conter as mesmas colunas definidas em features_metadata.json
y_proba = pipeline.predict_proba(X_novo)[:, 1]
risco = (y_proba >= 0.60).astype(int)  # threshold ajustado para 0.60
```

> O threshold de **0.60** foi escolhido deliberadamente para reduzir falsos negativos — priorizando a detecção de alunos em risco real.

---

## 🔑 Principais Achados

- O programa é **comprovadamente eficaz**: INDE cresce ~57% de Quartzo a Topázio
- **IDA + IEG** explicam >70% da composição do INDE
- **IPS** funciona como termômetro preventivo de risco de desempenho (T+1)
- A **autoavaliação (IAA)** é coerente com a realidade, mas com margem de intervenção pedagógica
- O modelo preditivo opera em produção com **ROC-AUC ~0.675–0.694**

---

## 👥 Contexto

Projeto desenvolvido como **Tech Challenge 5** em parceria com a ONG [Passos Mágicos](https://passosmagicos.org.br/), que transforma a vida de crianças e jovens em situação de vulnerabilidade social por meio da educação.
