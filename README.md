# AI-Assisted Customer Churn Prediction and Explainable Risk Monitoring System

A beginner-friendly starter MVP built with Python, Streamlit, Pandas, and scikit-learn.

## Features
- Upload a customer churn CSV or use the included clearly labelled demo data.
- Train a baseline Random Forest classifier.
- Show churn probability, configurable Low/Medium/High risk thresholds, and top contributing input signals.
- Flag unusual usage changes compared with a customer's previous usage when baseline columns are available.
- Record customer interventions in a local SQLite database.
- Generate evidence-linked retention playbooks from observed customer fields.
- Optionally send deduplicated Slack and SMTP email alerts for new high-risk transitions.
- Deploy the dashboard in a non-root Docker container with persistent application data.

## 1. Run on Windows (VS Code)
Open this folder in VS Code. In the terminal, run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `python` is not recognized, install Python 3.11+ from python.org and select **Add Python to PATH** during setup.

## 2. Docker deployment
With Docker Desktop running, build and start the dashboard:

```powershell
docker compose up --build -d
```

Open `http://localhost:8501`. Set `PORT` in the environment to publish a different host port. Compose stores the SQLite database and alert CSV in the named `churn-data` volume mounted at `/app/data`; keep this volume when upgrading the container. For a local bind mount instead, replace the volume entry with `./app-data:/app/data` and ensure the directory is writable by container UID 10001.

The container runs as a non-root user and includes a Streamlit health check. Uploaded customer datasets are processed by the app; they are not included in the image. Back up the persistent data volume and restrict access to uploaded data appropriately.

## 3. Optional Slack and email notifications
Alerts are disabled by default. Configure the selected channel using environment variables, then set `AUTO_ALERTS_ENABLED=true` to opt in. Example PowerShell configuration (values are placeholders):

```powershell
$env:AUTO_ALERTS_ENABLED = "true"
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
docker compose up --build -d
```

For SMTP, set `SMTP_HOST`, `SMTP_PORT` (default `587`), `SMTP_FROM`, `ALERT_EMAIL_TO` (comma-separated), and optionally `SMTP_USER` and `SMTP_PASSWORD`. STARTTLS is enabled by default; set `SMTP_USE_SSL=true` for implicit TLS and `SMTP_USE_STARTTLS=false` when using it. Never place real credentials in source control, Docker build arguments, or the image. Use a protected environment/secrets manager for deployment; a local untracked `.env` file can be used with Docker Compose.

Only uploaded datasets trigger external alerts; demo-data alerts are suppressed. A notification is generated when a customer first enters High risk or the score rises materially, not on every Streamlit rerun. Failed delivery retries are cooldown-limited. The app stores hashed customer keys and delivery status locally, but notification content includes customer IDs, risk scores, and recommended actions; send only to approved destinations. Review delivery history in the dashboard.

## 4. Dataset format
Upload a CSV where each row represents a customer. The target column should indicate whether the customer churned (for example, `Churn` with `Yes`/`No` values). Other columns can include numeric and categorical customer signals.

For the demo, click **Use demo data**. Demo records are for demonstrating the app only and are not real customer data or valid evidence of model performance.

## 5. Important notes
- This is a learning MVP, not a production decision system.
- Model probabilities are estimates, not certainty.
- Feature importance indicates which inputs the model used overall; it does not prove causation.
- Retention playbooks match configured signals in uploaded fields to operational suggestions. They do not explain the model prediction and do not establish causal churn drivers.
- The included baseline behaviour check only runs if both `usage_previous` and `usage_recent` columns exist.
- Do not upload sensitive real customer information without appropriate authorization and safeguards.
