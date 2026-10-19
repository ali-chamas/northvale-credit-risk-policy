# Deploying the simulator to Streamlit Community Cloud

The app (`app/streamlit_app.py`) reads only the small files in `app/data/` (about 4 MB, committed) and installs only `app/requirements.txt` (Streamlit, pandas, pyarrow, numpy, altair). No model library is needed at runtime. Northvale Credit is a fictional company and the app says so at the top.

## Before you start

- `make analysis` has been run, so `app/data/scored_test.parquet`, `assumptions.json` and `reason_codes.json` exist and are committed.
- The app runs locally: `make app`, then open http://localhost:8501.

## Steps

1. Create an empty public GitHub repository named `northvale-credit-risk-policy` under the `ali-chamas` account (no README, no licence, no .gitignore: the repo already has them).
2. In this folder, add the remote and push:
   ```bash
   git remote add origin https://github.com/ali-chamas/northvale-credit-risk-policy.git
   git push -u origin main
   ```
3. Open https://share.streamlit.io and sign in with the same GitHub account. Authorise Streamlit to read your repositories when asked.
4. Click **Create app**, then choose **Deploy a public app from GitHub**.
5. Fill in:
   - Repository: `ali-chamas/northvale-credit-risk-policy`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - App URL: pick a subdomain, for example `northvale-credit`
6. Open **Advanced settings** and set **Python version** to `3.11`. Leave secrets empty. Save.
7. Click **Deploy**. Streamlit installs `app/requirements.txt` (it looks next to the main file before the repository root) and starts the app. The first build takes a few minutes.
8. Check the deployed app: the fictional-company notice is at the top, the Policy impact tab shows the same approval rate, default rate and profit as the README, and the Applicant explanations tab loads.
9. Copy the app URL. In `docs/templates/README.md.j2`, replace the simulator link target `app/streamlit_app.py` in the links line near the top with the URL, then run `make docs` and commit and push.

## If something goes wrong

- **Build installs LightGBM or SHAP:** the main file path is wrong, so Streamlit fell back to the root `requirements.txt`. Set it to `app/streamlit_app.py`.
- **`FileNotFoundError` for `app/data/...`:** the data files were not committed. Run `make analysis`, then `git add app/data` and push.
- **The app is asleep:** community apps sleep after a period without visitors; open the URL and click the wake-up button.
