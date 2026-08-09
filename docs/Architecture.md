# GENESIS-AI System Architecture

## High-Level Pipeline Flow
User → Upload CSV → Dataset Intelligence Profile (DIP v1.1) → Signal Normalizer → Candidate Registry → Compatibility Filters → Deterministic Rule Scorer → Top-K Ranker → Recommendation Report → [Future Week 4/5 Evolutionary Optimizer]

## Backend Modules
- `dataset/`: Loader, Validator, Profiler, Complexity Score, DIP Builder v1.1 *(Completed — Week 2 Baseline)*
- `recommendation/`: Schemas, Registry, Normalizer, Compatibility Filters, Rule Engine, Scorer, Ranker, Recommendation Engine *(Completed — Week 3)*
- `optimizer/`: GA Chromosome, Fitness, Genetic Algorithm *(Pending — Week 4/5)*
- `explainability/`: Metrics, SHAP Engine *(Pending — Future)*
- `llm/`: Prompts, Natural Language Explanations *(Pending — Future)*
- `api/`: FastAPI routes (`/health`, `/dip`, `/recommend`) *(Active)*
- `database/`: SQLite models & migrations *(Pending — Future)*

## Technology Stack
- Frontend: React + Tailwind + Plotly (Future)
- Backend: FastAPI, Python 3.x, Pydantic
- Data/ML: pandas, NumPy, SciPy, scikit-learn
- Optimization: DEAP or in-house GA (Future Week 4/5)
- Explainability: SHAP (Future)
- Database: SQLite (Future)

