<!-- Generated from docs/templates/docs/decision_log.md.j2 by `make docs`. Edit the template, not this file. -->
# Decision log

Choices that shaped the analysis, in the order they were made, including where the work departs from the original brief. Numbers are rendered from `reports/metrics.json`.

## Scope and target

1. **Window changed from 2015-2018 to 36-month loans issued 2012-2015.** At the data snapshot 28% of 36-month 2016 loans and 60% of 2017 loans were still running. Keeping only their resolved loans would select early defaults and early payoffs. In the chosen window only 0.02% of loans are unresolved (`reports/tables/censoring_by_vintage.csv`). The 2015 vintage was checked separately: 0.05% unresolved, so no adjustment to the window was needed.
2. **60-month loans excluded** (33% of the 2015 vintage still running, different risk profile). Next step once more vintages resolve.
3. **Joint applications excluded** (239 loans). They appear only from October 2015, so the model could not learn them.
4. **Target:** Charged Off or Default = 1, Fully Paid = 0. The 147 loans still current, in grace or late are excluded and counted by vintage (`reports/tables/scope_flow.csv`).
5. **Split by issue month:** train 2012-01 to 2014-12, validation 2015-01 to 2015-06, test 2015-07 to 2015-12. No random split anywhere. No sampling: all 589,249 loans are used.

## Features

6. **Every raw column has a written decision** (`docs/feature_audit.csv`, 151 rows). Kept: 58 application fields plus 4 transformed into derived features.
7. **Lending Club's grade, sub-grade, interest rate and instalment are not features.** They are the incumbent model; the sub-grade is the benchmark and the rate drives the P&L.
8. **Proxy fields dropped:** `zip_code`, `addr_state` (geography) and `emp_title` (free text, very high cardinality).
9. **Late-2015 bureau fields dropped** (14 columns): empty for the whole training window.
10. **Missing values kept as information.** `mths_since_last_delinq` is missing when the applicant has never been delinquent; LightGBM learns a direction for missing values and the scorecard gives them their own bin.

## Models

11. **Scorecard selection:** IV at least 0.02, drop the weaker of any pair with WoE correlation above 0.7, add characteristics in IV order only if validation AUC rises by 0.0005 or more and every coefficient keeps the expected sign. Result: 13 characteristics (`reports/tables/scorecard_selection.csv`).
12. **LightGBM:** 6-configuration grid on leaves and minimum leaf size, learning rate 0.05, early stopping on validation AUC, 4 threads, deterministic mode. Best: 15 leaves, 1000 minimum rows per leaf, 780 trees.
13. **Recalibration on validation (Platt) for all three rankers.** Trained on a calmer period, the models under-predicted 2015 defaults.

## Business layer

14. **Cash definitions checked before use:** `total_pymnt` = principal + interest + late fees + recoveries on every loan (largest gap 0.01 dollars). Lender cash = `total_pymnt` minus `collection_recovery_fee`.
15. **G and L measured on training loans only**, amount-weighted; low/high = best and worst training year.
16. **Cost of funds and opex are stated assumptions** (base 4.0% a year and 1.0% of principal), editable in the simulator.
17. **Cutoff chosen on validation** by expected profit with a 90% approval floor; test used once.
18. **Plain PD cutoff preferred to a price-aware margin rule**, which earned -$2.7M relative to it on test.
19. **Headline impact uses realized cash**, because expected profit overstated realized profit by 1.1% of the test book.
20. **Annualisation:** twice the six-month test window, stated wherever an annual number appears.
