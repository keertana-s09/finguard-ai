# FinGuard AI: optional React frontend

A second user interface for the same backend. It shows exactly what the Streamlit app (`app.py`) shows, because both call `FinGuardPipeline`. **Streamlit remains the main app for the faculty demo**; this frontend is optional.

```
React (Vite dev server, :5173) --/api--> FastAPI (api.py, :8000) --> FinGuardPipeline --> SQLite
```

## Requirements
- Everything from the main README (Python venv with `requirements.txt` installed, datasets and models created).
- **Node.js 18 or newer** (https://nodejs.org; check with `node --version`).

## Install (once)
From the project root, with the virtual environment activated:

```powershell
pip install -r requirements-api.txt
cd frontend
npm ci          # installs the exact versions from package-lock.json
cd ..
```

## Run (two terminals)

**Terminal 1** (project root, venv activated): the API
```powershell
uvicorn api:app --port 8000
```

**Terminal 2**: the React app
```powershell
cd frontend
npm run dev
```

Open **http://localhost:5173**. `?page=1` … `?page=7` opens a page directly, like the Streamlit app.

The API and Streamlit use the same `finguard.db`, so data loaded in one appears in the other. Avoid running both at the same time while changing data.

## Troubleshooting
| Message | Fix |
|---|---|
| "Cannot reach the API" | Terminal 1 is not running, or uvicorn is on a different port than 8000 |
| "The API returned an error (500 …)" | The API crashed; the error is printed in Terminal 1 |
| A validation message such as `savings_goal: Input should be greater than or equal to 0` | The input was rejected; nothing was saved |
| `npm ci` fails | Use Node 18+, or delete `node_modules` and run `npm ci` again |

## Tests
`python -m pytest` (from the project root) includes `tests/test_api.py`. It runs when the packages from `requirements-api.txt` are installed and is skipped otherwise.
