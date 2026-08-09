# Architecture

## High-Level Flow
User → Upload CSV → Dataset Intelligence Profile (DIP) → Recommendation Engine → Genetic Optimizer → Best Pipeline → Explainability → LLM Assistant → Dashboard

## Backend Modules
- `dataset/`: Loader, Validator, Profiler, Complexity Score, DIP Builder
- `recommendation/`: Rules, Scorer, Candidate Generator (Week 3)
- `optimizer/`: GA Chromosome, Fitness, Genetic Algorithm (Week 4+)
- `explainability/`: Metrics, SHAP Engine
- `llm/`: Prompts, Assistant
- `api/`: FastAPI routes
- `database/`: SQLite models & migrations

## Technology Stack
- Frontend: React + Tailwind + Plotly
- Backend: FastAPI, Python 3.x
- Data/ML: pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, CatBoost
- Optimization: DEAP or in-house GA
- Explainability: SHAP
- Database: SQLite
