<!-- Generated from docs/templates/INTERVIEW_NOTES.md.j2 by `make docs`. Edit the template, not this file. -->
# Northvale Credit: interview notes

**Personal prep notes for Ali.** Northvale Credit is a fictional company; the data is public Lending Club loans. Every number below is rendered from `reports/metrics.json`, so it matches the README and the deck.

## The 5 numbers to know by heart

1. **12.3% to 14.9%**: default rate of 36-month loans, 2013 vintage vs 2015 vintage. Grade mix moved it -0.6 pp; risk inside grades +3.1 pp. The grade drifted.
2. **AUC 0.701 vs 0.681**: LightGBM vs Lending Club sub-grade on the test window (161,997 loans, Jul-Dec 2015). Scorecard 0.675. KS 0.29 vs 0.27.
3. **0.70 to 0.95**: test AUC when one post-origination field (refreshed FICO) is added. The leakage demonstration.
4. **PD > 27.9% = decline**: keeps 91.0% of applicants, default rate 14.7% to 12.8%.
5. **-$1.1M to $12.7M per half-year**, so +$27.6M a year and $52.1M of credit losses avoided a year (base assumptions, annual = 2 x the test half-year).

## Decisions: what, why, and what I rejected

| Decision | Why | Rejected alternative and why |
|---|---|---|
| 36-month loans issued 2012-2015 | 0.02% unresolved, so default rates are not censored | 2015-2018 (brief's first idea): 28% of 2016 loans still running; keeping only finished ones over-samples early defaults |
| Exclude 60-month loans | Different risk and too few finished vintages (33% of 2015 still running) | Pooling terms: mixes two risk profiles and adds censoring back |
| Default = Charged Off or Default; good = Fully Paid; drop the 147 unresolved | Only final outcomes are labels | Treat "Late 31-120" as default: a label the snapshot date decides, not the borrower |
| Out-of-time split: train 2012-14, validate 2015 H1, test 2015 H2 | Mimics deployment; the 2015 default rate is higher, which a random split would hide | Random split: leaks future vintages into training and overstates AUC |
| Grade, sub-grade, int_rate, installment excluded as features | They are the incumbent model's output; including them copies Lending Club | Include them: AUC would rise but Northvale would not have its own view of risk; used as benchmark and in the P&L instead |
| Drop zip_code, addr_state, emp_title | Geography and job titles can proxy for protected characteristics (ECOA / Reg B disparate impact, EU AI Act high-risk) | Keep them: zip_code alone reaches a single-field AUC of 0.56, a small gain for a large compliance risk |
| Drop 14 late-2015 bureau fields | Empty for every training loan | Impute them: the model would learn nothing and the test would see different inputs |
| Scorecard: IV >= 0.02, abs(r) < 0.7, stepwise on validation AUC, all WoE coefficients negative | Readable scorecard of 13 characteristics that a credit committee can sign off | All 24 IV-passing features: correlated pairs flip signs and the points table stops making sense |
| LightGBM: 6-point grid, early stopping on validation | Validation AUC moved by only 0.0012 across the grid; a bigger search would tune noise | Large Bayesian search: costs time and risks fitting the validation window |
| Platt recalibration on validation for every model | Models trained on 2012-14 under-predicted 2015 (mean PD 13.5% vs 14.7% observed); Platt shifts the level and keeps the ranking | Isotonic: more flexible but step-shaped PDs; no recalibration: the money calculation would be wrong |
| G and L from training loans, amount-weighted | Measured, not assumed: G by rate band, L = 35.9% | Textbook LGD of 60-80%: ignores that defaulters repaid 42.0% of principal first |
| Cutoff chosen on validation by expected profit, volume floor 90% | Profit is the committee's objective; the floor is their volume constraint | Max-KS point or a chosen AUC threshold: ignores price and loss; choosing on test would be optimistic |
| Plain PD cutoff | Beat a price-aware margin ranking by $2.7M on test at the same approval rate | Margin ranking: high prices did not cover high risk in 2015 |
| Recommend balanced, not conservative | Meets the volume brief and limits how far we tighten on a model trained on approved loans only | Conservative (82.5% approval) earned more on test (+$18.8M) but gives up 16% of volume |

## Weak points and how to answer

- **"The AUC gain over the grade is tiny."** It is 0.019. Lending Club's grade already uses FICO and bureau data, so a big jump would be suspicious (see the leakage demo). The value shows elsewhere: inside grade C the model separates 12.1% from 29.9% default rates, and at the approval rate of a "decline D-G" rule it earns $16.6M vs $7.9M.
- **"Your expected profit is higher than realized."** Yes, by 1.1% of the test book, because 2015 defaults lost 39.4% of principal vs 35.9% in training. That is why I report realized cash as the headline, and why the adverse scenario exists.
- **"Validation is used three times."** Early stopping, recalibration and the cutoff all use January-June 2015. The test window was touched once, at the end. With more data I would split 2015 H1 into a tuning month range and a calibration range.
- **"You never saw rejected applicants."** Correct: no reject inference. The policy only declines loans Lending Club approved, so it can only tighten. Expanding approval needs a controlled test.
- **"The cost assumptions are made up."** Cost of funds and opex are assumptions; G and L are measured. The simulator lets the committee change them, and the README shows low/base/high: realized gain +$21.2M to +$40.5M a year.
- **"Doubling six months is crude."** It is a stated scaling basis, not a forecast. The six-month realized number (+$13.8M) is the measured one.
- **"The scorecard is worse than the grade."** Slightly (-0.007 AUC). It uses 13 binned characteristics and no interactions. It is the fallback if a regulator or the committee refuses a gradient-boosted model.
- **"Your declined example was graded A5 by Lending Club."** That is the point of the example: the grade said low risk, but the model's top reasons ("FICO score of 667 (median applicant 692)"; "7 credit accounts opened in the last 24 months (median 4)") put its PD at 34.6%. The model and the grade disagree on a minority of loans, and that is where the value is.
- **"Fairness?"** I removed obvious proxies, but I have not measured outcomes by protected group because the data has no such attribute. Before deployment: BISG-style proxy analysis, approval and error rates by group, and an adverse-impact ratio.

## Likely interview questions

**1. Why an out-of-time split instead of a random split?**
A credit model is always used on applicants who arrive after it was trained, so the test should look like that. Random splitting puts 2015 loans in training and lets the model learn the higher 2015 default level; it also hides drift. Here the default rate was 13.2% in the 2012-2014 training window and 14.7% in the test window. Before recalibration, LightGBM's mean PD on test was 13.5%: a random split would never have shown me that gap. The cost is that I evaluate on one half-year only, so I report the result as one out-of-time check, not a guarantee.

**2. Why 36-month loans from 2012 to 2015?**
Because the outcome has to be finished. The file stops at the end of 2018. For 36-month loans issued in 2016, 28% were still running. If I kept only the finished ones, loans that defaulted early or prepaid early would be over-represented, which biases the default rate. Loans issued 2012 to 2015 had at least three years to finish: only 0.02% were unresolved. 60-month loans are excluded because 33% of the 2015 vintage was still running and their risk profile differs; they are a next step.

**3. How do you know there is no leakage?**
Three controls. First, a written decision for all 151 columns in `docs/feature_audit.csv`: is it known when the lender decides? 38 post-origination columns are dropped. Second, a single-column screen: anything above an AUC of 0.70 on its own is flagged. 9 columns were flagged and all of them were post-origination; the best real application field, FICO, scores 0.60. Third, a demonstration: adding the refreshed FICO to the final model lifts test AUC from 0.701 to 0.948. My final model's AUC of 0.701 is in the plausible range for application data.

**4. Why not use the grade and interest rate as features?**
They are Lending Club's own risk model: the interest rate is set from the sub-grade, and the instalment is a function of the rate. A model that includes them mostly copies the incumbent, so Northvale would not learn what its own data says, and its decisions would break whenever the incumbent changes its grading, which is exactly what happened between 2013 and 2015. So the sub-grade is the benchmark (test AUC 0.681) and the rate feeds the P&L. A blend of LightGBM and the sub-grade reaches 0.710, which shows the two carry partly different information.

**5. Scorecard or LightGBM: which would you deploy?**
LightGBM for the decision, with the scorecard kept as the challenger and fallback. LightGBM is better on every test metric (AUC 0.701 vs 0.675) and at the recommended approval rate it earns +$13.8M vs +$9.5M for the scorecard over six months. The scorecard is easier to govern: 13 characteristics, monotonic bins, points a credit officer can read. LightGBM needs SHAP for reason codes and monitoring of its shape. If the regulator or the committee requires a fully transparent model, the scorecard is the answer and the cost is visible.

**6. What is KS, and why look at calibration as well as discrimination?**
KS is the largest gap between the cumulative score distributions of defaulters and non-defaulters: 0.29 here means that at the best threshold, the riskier side of the cut holds 29 percentage points more of all defaulters than of all good loans. AUC and KS measure ranking. Calibration measures whether a PD of 20% really means 20 defaults in 100. The cutoff is set in money, PD times loss against (1 - PD) times interest, so a well-ranked but badly calibrated model picks the wrong cutoff. After recalibration on early 2015, mean PD is 14.6% vs 14.7% observed on test, and no decile is off by more than 0.9 pp.

**7. What is PSI and how would you monitor this policy after launch?**
The population stability index compares the distribution of a score or feature between a reference period and now, bin by bin; under 0.10 is stable, above 0.25 needs action. The LightGBM score has a PSI of 0.005 between training and test, and the most shifted top feature is `percent_bc_gt_75` at 0.041. After launch I would track monthly: score and feature PSI, approval rate, early delinquency by vintage (30+ days past due at 3 and 6 months, since defaults take a year to show), calibration of PD vs observed by decile, and manual override rates.

**8. How did you estimate G and L?**
From the training window only, amount-weighted. I first checked the cash fields: total payment equals principal plus interest plus late fees plus recoveries on every loan, and fully paid loans repaid exactly their principal. G is the net interest earned per dollar on loans that paid off, by interest-rate band: 9.5% for loans priced under 8% up to 29.1% for 20% and above; prepayment is already inside it. L is the net loss per dollar on defaulted loans after interest and recoveries: 35.9%, because defaulters had repaid 42.0% of principal first. Costs are cost of funds times 1.30 principal-years plus opex.

**9. Why should profit, not AUC, choose the cutoff?**
AUC is the same at every cutoff; it says nothing about where to draw the line. The line depends on what a good loan earns and a bad loan costs. With base assumptions a loan priced at 11-14% breaks even at a PD of 20.2%, one priced at 20% or more at 35.2%. I chose the cutoff on validation by expected profit with a volume floor, then checked realized cash on test. On test, approving everyone made -$1.1M; the balanced cutoff made $12.7M.

**10. What about the applicants Lending Club rejected?**
Their outcomes do not exist, so I cannot do reject inference honestly. Methods such as parcelling or augmentation guess how rejected applicants would have performed, and the guess drives the answer. My policy therefore only declines applicants Lending Club approved; it can tighten but not expand. To approve new segments, Northvale would need a controlled test: approve a small random share of applicants just above the cutoff and measure their defaults.

**11. How do regulation and fairness affect this design?**
In the US, ECOA and Regulation B require specific reasons for an adverse action, which is why each decline comes with SHAP-based reason codes such as "FICO score of 667 (median applicant 692)". In the EU, the AI Act classifies creditworthiness scoring of individuals as high-risk: documentation, data governance, human oversight and monitoring are required, which the feature audit, model card and monitoring plan start to cover. I dropped zip code, state and job title as proxy risks. What is missing is an outcome test by protected group, which I would run before any launch.

## 60-second summary

"Northvale is a fictional online lender; I used 589k real Lending Club loans to answer its credit committee: which applicants should we stop approving, and what is it worth? First I showed the problem: default rates rose from 12.3% to 14.9% between the 2013 and 2015 vintages. The grade mix actually got safer, so the whole rise came from loans inside the same grade getting riskier while prices fell: grade rules are outdated. I scoped to loans with finished outcomes, audited all 151 columns for leakage, and compared Lending Club's sub-grade, a scorecard and LightGBM on a later period. LightGBM ranked best, with a test AUC of 0.70. Then I let profit pick the cutoff: declining applicants above a 28% PD keeps 91% of volume, cuts defaults from 14.7% to 12.8% and is worth about $27.6M a year under stated assumptions. A Streamlit simulator lets a credit manager test other cutoffs."
