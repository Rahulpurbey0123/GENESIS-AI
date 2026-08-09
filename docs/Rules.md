# Development Rules

## Tech Stack Rules
- Use Python 3.x, pandas, NumPy, SciPy, scikit-learn, FastAPI, uvicorn, pytest.
- Avoid reinventing basic ML algorithms.
- Avoid hardcoded dataset assumptions.
- Avoid random pipeline generation outside recommendation stage.

## Error Handling
- Validate CSV strictly before profiling.
- Handle missing target column gracefully.
- Log every experiment run.
- Return human-readable error messages from API endpoints.

## Research & AI Assistant Rules
- Explain results, metrics, and feature importance based strictly on empirical output.
- Never fabricate results or modify datasets automatically.
- DIP engine only observes and measures data; it does not transform or impute datasets silently.
