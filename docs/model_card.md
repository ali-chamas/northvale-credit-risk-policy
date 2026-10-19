<!-- Generated from docs/templates/docs/model_card.md.j2 by `make docs`. Edit the template, not this file. -->
# Model card: Northvale Credit default model (LightGBM)

Northvale Credit is fictional; this card documents a learning project on public Lending Club data, in the format a high-risk credit model would need.

**Purpose.** Estimate the probability that a new 36-month unsecured personal loan applicant defaults (Charged Off or Default) over the life of the loan, to support an approve / decline decision at a PD cutoff of 27.9%.

**Not for:** pricing, credit limits, 60-month loans, joint applications, applicants below FICO 660, or any population outside Lending Club's 2012-2015 approved borrowers.

**Training data.** 306,462 resolved loans issued 2012-2014 (13.2% default). Validation 120,790 loans (Jan-Jun 2015), test 161,997 loans (Jul-Dec 2015). Only fields known at application are used (63 features; see `data_dictionary.md` and `feature_audit.csv`).

**Model.** LightGBM, 15 leaves, 780 trees, early stopping on validation, Platt recalibration on validation. Challenger: logistic WoE scorecard with 13 characteristics.

**Performance on the test window.**

| Metric | LightGBM | Scorecard | LC sub-grade (incumbent) |
|---|---|---|---|
| AUC | 0.701 | 0.675 | 0.681 |
| KS | 0.292 | 0.253 | 0.267 |
| Brier | 0.1171 | 0.1192 | 0.1189 |
| Mean PD / observed | 14.6% / 14.7% | 14.5% / 14.7% | 14.5% / 14.7% |
| Score PSI, train vs test | 0.005 | 0.004 | 0.023 |

Default rate by LightGBM decile ranges from 2.9% (safest) to 33.8% (riskiest). Largest decile calibration gap: 0.9 pp.

**Explainability.** SHAP values per applicant; the three largest risk-raising contributions become plain-English reason codes (for adverse-action notices). Top global drivers: FICO score at application; Loan amount / annual income; Accounts opened, last 24 months; Debt-to-income ratio (%); Annual income.

**Fairness and regulation.** No protected attributes are used or available. Geography (zip code, state) and job title are excluded as potential proxies. Outcome testing by protected group has not been done and is required before any use. In the EU, creditworthiness scoring of individuals is a high-risk use under the AI Act (risk management, data governance, documentation, human oversight, monitoring). In the US, ECOA / Regulation B require specific adverse-action reasons.

**Known limitations.** Trained on approved loans only (no reject inference); one credit cycle (2012-2015); losses per default rose in 2015 (39.4% vs 35.9% in training); prices held fixed.

**Monitoring plan.** Monthly: score PSI and top-feature PSI (watch at 0.10, act at 0.25), approval rate, early delinquency (30+ days past due at 3 and 6 months on book) by vintage, PD vs observed by decile once outcomes mature, override rate and reasons. Recalibrate when mean PD and observed default rate diverge by more than one percentage point for two consecutive quarters.
