# Case Study 01 — Credit Scoring: Retail Portfolio

## Objetivo

Construir un **modelo de credit scoring** sobre datos reales de tarjetas de crédito, siguiendo la metodología estándar de la industria bancaria: WOE/IV feature engineering, regresión logística, escalamiento PDO y validación con métricas regulatorias (KS, Gini, AUC, CAP, PSI).

## Dataset

| Campo | Detalle |
|-------|---------|
| **Nombre** | Default of Credit Card Clients |
| **Fuente** | [UCI ML Repository — id 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| **Autores** | Yeh, I. (2009). DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H) |
| **Licencia** | CC BY 4.0 |
| **Observaciones** | 30,000 clientes de tarjeta de crédito en Taiwan |
| **Período** | Abril–Septiembre 2005 |
| **Target** | `DEFAULT` = incumplimiento de pago en octubre 2005 |
| **Default rate** | ~22.1% |

### Variables clave

```
LIMIT_BAL   : Monto de crédito aprobado (TWD)
SEX, EDUCATION, MARRIAGE, AGE : Demografía
PAY_0..PAY_6 : Historial de pagos (meses -1 a -6)
BILL_AMT1..6 : Estado de cuenta mensual
PAY_AMT1..6  : Monto pagado mensualmente
DEFAULT     : 1 = incumplimiento (TARGET)
```

## Setup

```bash
# Desde la raíz del repo
bash 01_credit_scoring/data/download.sh   # descarga vía ucimlrepo API
```

Requiere `ucimlrepo` (incluido en `pyproject.toml`). No se sube data al repo (ver `.gitignore`).

## Metodología

```
1. EDA              → distribuciones, default rate por segmento, correlaciones
2. Feature Eng.     → WOE binning + IV screen (umbral IV > 0.02)
3. Model Dev.       → Logistic Regression (class_weight='balanced')
4. Scorecard        → escalamiento PDO: base=600, PDO=20, Score = Offset + Factor × log-odds
5. Validation       → KS, Gini, AUC, CAP Ratio, PSI (train vs test)
6. Report           → HTML exportado con nbconvert → LinkedIn post
```

## Targets de Validación

| Métrica | Umbral mínimo | Referencia |
|---------|--------------|------------|
| KS | > 0.30 | Basel II IRB guidelines |
| Gini | > 0.40 | Siddiqi (2006) |
| AUC | > 0.70 | Thomas et al. (2002) |
| PSI | < 0.10 | Industry standard |
| CAP Ratio | > 0.60 | OCC Model Risk Guidance |

## Referencias

- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Thomas, L., Edelman, D., Crook, J. (2002). *Credit Scoring and Its Applications*. SIAM.
- Basel Committee on Banking Supervision (2006). *IRB Approach for Credit Risk*.
