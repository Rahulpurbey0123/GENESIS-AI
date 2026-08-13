# GENESIS-AI Week 8 — Frontend Dashboard, Application Integration & Scientific Integrity Hardening

## 1. Executive Summary & Objective
Week 8 integrates all previous GENESIS-AI research modules (Weeks 1–7) into a unified, high-performance web dashboard (React + Tailwind CSS + Plotly) and FastAPI REST orchestration layer, hardened for strict scientific integrity.

The core research paradigm remains uncompromised:
```
Dataset Ingestion
       ↓
Dataset Intelligence Profile (DIP) v1.1
       ↓
Intelligent Search Space Reduction
       ↓
Evolutionary Pipeline Optimization
       ↓
Integrated Research Evaluation (Held-Out Test Set Metrics)
       ↓
Post-Hoc SHAP & Evidence-Grounded LLM Explainability
```

---

## 2. Scientific Integrity & Hardening Enforcement
1. **Zero Fabricated Metrics**: All classification metrics (`accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `roc_auc`) and regression metrics (`mae`, `rmse`, `r2`) are calculated from actual model predictions `y_test_pred` evaluated on held-out test data (`X_test`, `y_test`).
2. **Zero Synthetic SHAP Values**: Position-based feature ranking formulas have been eliminated. Global and local explanations are derived exclusively from the Week 5 `ExplainabilityEngine` (`backend/explainability/engine.py`).
3. **Honest Explanation Method Identification**: The dashboard explicitly identifies the explainability method used (`SHAP`, `Permutation Importance`, `Linear Coefficients`, or `Native Tree`).
4. **Real Diagnostic Plots**: Confusion matrices, ROC curves, and regression residual plots are generated from actual test predictions. Uncomputable plots gracefully return `null` (`N/A`).
5. **Read-Only Dataset Recommendations**: Viewing model recommendations (`/api/datasets/{id}/recommendations`) is strictly read-only and does not launch optimization experiments. Experiments start ONLY when explicitly triggered.
6. **Real-Time GA Progress Tracking**: Progress tracking receives live generation callbacks directly from `EvolutionaryOptimizer`.
7. **Zero Frontend Fake Fallbacks**: Scientific fields use strict `null`/`undefined` checks displaying `"N/A"` when unavailable rather than synthetic fallback defaults.

---

## 3. Architecture & Data Flow

```
                             USER
                               │
                               ▼
                    ┌─────────────────────┐
                    │  React Dashboard    │
                    │  (Port 5173 / Vite) │
                    └──────────┬──────────┘
                               │ (REST API / JSON)
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    │  (Port 8000 / Uvicorn)
                    └──────────┬──────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
 Dataset Loader        Background Job Manager      SQLite Database
(backend/dataset)        (backend/jobs.py)      (data/genesis.db)
       │                       │                       │
       ▼                       ▼                       ▼
  DIP Engine             Evolutionary GA           Persistence
(backend/dataset)     (backend/optimization)   (Datasets/Exps/Chats)
       │                       │                       │
       ▼                       ▼                       ▼
Recommendation        SHAP Explainability        Grounded LLM
 (backend/rec)       (backend/explainability)    (backend/llm)
```

---

## 4. Frontend Architecture (`frontend/`)

```
frontend/
├── src/
│   ├── components/
│   │   ├── Navbar.jsx         # Header navigation bar with active tab tracking
│   │   ├── Footer.jsx         # Disclaimer footer with scientific notice
│   │   └── PlotViewer.jsx     # Interactive Plotly chart renderer
│   ├── pages/
│   │   ├── HomePage.jsx               # Screen 1: Landing Page & Framework Overview
│   │   ├── UploadPage.jsx             # Screen 2: Drag-and-Drop CSV Ingestion & Target Selection
│   │   ├── DipDashboardPage.jsx       # Screen 3: Dataset Intelligence Profile (DIP) Dashboard
│   │   ├── RecommendationsPage.jsx    # Screen 4: Read-Only Model Priority Scores & Search Space Reduction
│   │   ├── OptimizationPage.jsx       # Screen 5: Real-Time GA Progress & Live Convergence Chart
│   │   ├── ResultsPage.jsx            # Screen 6: Best Pipeline Test Metrics & Efficiency Breakdown
│   │   ├── ExplainabilityPage.jsx     # Screen 7: SHAP Summary & Global Feature Importance
│   │   ├── AssistantPage.jsx          # Screen 8: Evidence-Grounded LLM Chat Interface
│   │   └── HistoryPage.jsx            # Screen 9: SQLite Persistent Experiment Run History
│   ├── services/
│   │   └── api.js             # Centralized fetch API service layer
│   ├── App.jsx                # Core application container & state manager
│   ├── main.jsx               # React DOM entrypoint
│   └── index.css              # Dark glassmorphism styling & Tailwind CSS
├── vite.config.js             # Vite builder configuration & dev API proxy
└── package.json               # Node package configuration
```

---

## 5. Implemented FastAPI REST Endpoints

### Dataset Management
- `POST /api/datasets/upload`: Validates CSV structure, sanitizes filename, computes SHA-256 hash, stores file in `data/uploads/`, and returns auto-suggested target column.
- `GET /api/datasets/{id}`: Retrieves stored metadata for an uploaded dataset.

### DIP Profiling & Recommendations (Read-Only)
- `POST /api/datasets/{id}/profile`: Executes DIP v1.1 engine for a confirmed target column and returns structured profile.
- `GET /api/datasets/{id}/profile`: Retrieves stored DIP profile.
- `GET /api/datasets/{id}/recommendations` & `POST /api/datasets/{id}/recommendations`: Returns read-only model recommendations and search-space reduction statistics without launching an experiment.

### Experiment & Optimization Orchestration
- `POST /api/experiments`: Submits background optimization job (`RUNNING`, `COMPLETED`, `FAILED`).
- `GET /api/experiments`: Lists all saved experiments from SQLite database.
- `GET /api/experiments/{id}`: Returns real-time job status, generation progress, best fitness score, runtime, and search space reduction.
- `GET /api/experiments/{id}/recommendations`: Returns model priority scores for completed experiment.
- `GET /api/experiments/{id}/results`: Returns best pipeline configuration, test set evaluation metrics, and execution efficiency.
- `GET /api/experiments/{id}/explanations`: Returns SHAP summary, global feature importance rankings, and diagnostic plots.
- `POST /api/experiments/{id}/chat`: Context-grounded LLM assistant querying stored experiment evidence safely.

---

## 6. How to Run the Application

### Backend Startup
```bash
# From workspace root
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Startup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Dashboard will be accessible at: `http://localhost:5173`

---

## 7. Verification Results
- **Pytest Suite**: All 219 existing Week 1–7 research tests + Week 8 REST API and hardening tests pass cleanly (**231 passed**).
- **Frontend Production Build**: `npm run build` completes with 0 errors.
