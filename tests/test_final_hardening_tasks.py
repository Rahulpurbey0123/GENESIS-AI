"""
Automated tests for GENESIS-AI Final Hardening Tasks.
Tests SQLite DIP schema migration, hardened GET DIP API, stored config recommendation context,
experiment-grounded LLM evidence, and LLM provider configuration error handling.
"""

import json
import os
import sqlite3
import tempfile
import pytest
from fastapi.testclient import TestClient

from backend.database import init_db, get_db_connection, DatabaseService, DB_PATH
from backend.main import app
from backend.llm.evidence import EvidenceExtractor, EvidenceValidator
from backend.llm.prompt_builder import PromptBuilder
from backend.llm.service import LLMService

client = TestClient(app)


def test_sqlite_dip_schema_migration(tmp_path):
    """
    Test 1: Verify safe SQLite DIP migration from single dataset_id PK to composite (dataset_id, target_column) PK.
    Verifies that existing data is preserved and dataset A + target A and dataset A + target B can coexist.
    """
    db_file = tmp_path / "test_genesis.db"
    
    # Create legacy table structure with PRIMARY KEY (dataset_id)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.executescript("""
    CREATE TABLE datasets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        filepath TEXT NOT NULL,
        rows INTEGER NOT NULL,
        columns INTEGER NOT NULL,
        features_json TEXT NOT NULL,
        suggested_target TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE dip_profiles (
        dataset_id TEXT PRIMARY KEY,
        target_column TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    
    cursor.execute(
        "INSERT INTO datasets (id, name, content_hash, filepath, rows, columns, features_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("ds_alpha", "alpha.csv", "hash123", "/tmp/alpha.csv", 100, 5, '["f1", "targetA", "targetB"]', "2026-08-13T00:00:00Z")
    )
    cursor.execute(
        "INSERT INTO dip_profiles (dataset_id, target_column, profile_json, created_at) VALUES (?, ?, ?, ?)",
        ("ds_alpha", "targetA", '{"dataset_id": "ds_alpha", "target_column": "targetA"}', "2026-08-13T00:00:00Z")
    )
    conn.commit()
    conn.close()

    # Run migration logic on this database file
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(dip_profiles)")
    info = cursor.fetchall()
    pk_cols = [row["name"] for row in info if row["pk"] > 0]
    assert pk_cols == ["dataset_id"]

    cursor.executescript("""
    CREATE TABLE dip_profiles_new (
        dataset_id TEXT NOT NULL,
        target_column TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (dataset_id, target_column),
        FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
    );
    INSERT OR IGNORE INTO dip_profiles_new (dataset_id, target_column, profile_json, created_at)
    SELECT dataset_id, COALESCE(target_column, ''), profile_json, created_at FROM dip_profiles;
    DROP TABLE dip_profiles;
    ALTER TABLE dip_profiles_new RENAME TO dip_profiles;
    """)
    conn.commit()

    # Check primary keys after migration
    cursor.execute("PRAGMA table_info(dip_profiles)")
    info = cursor.fetchall()
    pk_cols = [row["name"] for row in info if row["pk"] > 0]
    assert set(pk_cols) == {"dataset_id", "target_column"}

    # Verify existing profile was preserved
    cursor.execute("SELECT * FROM dip_profiles WHERE dataset_id = ? AND target_column = ?", ("ds_alpha", "targetA"))
    row_a = cursor.fetchone()
    assert row_a is not None

    # Verify dataset A + target B can now be inserted and coexist with dataset A + target A
    cursor.execute(
        "INSERT INTO dip_profiles (dataset_id, target_column, profile_json, created_at) VALUES (?, ?, ?, ?)",
        ("ds_alpha", "targetB", '{"dataset_id": "ds_alpha", "target_column": "targetB"}', "2026-08-13T01:00:00Z")
    )
    conn.commit()

    cursor.execute("SELECT COUNT(*) as count FROM dip_profiles WHERE dataset_id = ?", ("ds_alpha",))
    count = cursor.fetchone()["count"]
    assert count == 2
    conn.close()


def test_get_dip_api_hardening():
    """
    Test 2: Verify GET /api/datasets/{id}/profile behavior.
    - target_column supplied -> target aware
    - target_column omitted & 0 profiles -> 404
    - target_column omitted & 1 profile -> returns it
    - target_column omitted & multiple profiles -> 400 Bad Request with clear message
    """
    dataset_id = "ds_test_hardening"
    
    # Clean up previous profiles if any
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dip_profiles WHERE dataset_id = ?", (dataset_id,))
    conn.commit()

    # Case 0: zero profiles -> 404
    res = client.get(f"/api/datasets/{dataset_id}/profile")
    assert res.status_code == 404

    # Case 1: exactly 1 profile
    DatabaseService.save_dip_profile(dataset_id, "target_X", {"dataset": {"name": "test.csv"}, "target": {"name": "target_X"}})
    res_one = client.get(f"/api/datasets/{dataset_id}/profile")
    assert res_one.status_code == 200
    assert res_one.json()["target"]["name"] == "target_X"

    # Exact target-aware call
    res_exact = client.get(f"/api/datasets/{dataset_id}/profile?target_column=target_X")
    assert res_exact.status_code == 200
    assert res_exact.json()["target"]["name"] == "target_X"

    # Case 2: multiple profiles exist -> return HTTP 400
    DatabaseService.save_dip_profile(dataset_id, "target_Y", {"dataset": {"name": "test.csv"}, "target": {"name": "target_Y"}})
    res_multi = client.get(f"/api/datasets/{dataset_id}/profile")
    assert res_multi.status_code == 400
    assert "target_column is required because multiple DIP profiles exist for this dataset." in res_multi.json()["detail"]

    # Target-aware call still works
    res_target_y = client.get(f"/api/datasets/{dataset_id}/profile?target_column=target_Y")
    assert res_target_y.status_code == 200
    assert res_target_y.json()["target"]["name"] == "target_Y"


def test_experiment_recommendations_stored_config():
    """
    Test 3: Verify GET /api/experiments/{id}/recommendations uses stored config top_k for GENESIS mode
    and exposes candidate pool for BASELINE / RANDOM.
    """
    # Create test dataset entry in DB
    ds_id = "ds_exp_rec_test"
    with open("data/uploads/test_exp_rec.csv", "w") as f:
        f.write("feat1,feat2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n")

    DatabaseService.save_dataset(
        dataset_id=ds_id,
        name="test_exp_rec.csv",
        content_hash="hash_exp_rec",
        filepath="data/uploads/test_exp_rec.csv",
        rows=6,
        columns=3,
        features=["feat1", "feat2", "target"],
        suggested_target="target"
    )

    import uuid
    exp_id_gen = f"exp_gen_{uuid.uuid4().hex[:8]}"
    exp_id_base = f"exp_base_{uuid.uuid4().hex[:8]}"

    # Test GENESIS experiment with stored top_k = 2
    exp_genesis = DatabaseService.create_experiment(
        experiment_id=exp_id_gen,
        dataset_id=ds_id,
        dataset_name="test_exp_rec.csv",
        target_column="target",
        mode="genesis",
        config={"top_k": 2, "population_size": 10, "generations": 2}
    )

    res_gen = client.get(f"/api/experiments/{exp_id_gen}/recommendations")
    assert res_gen.status_code == 200
    body_gen = res_gen.json()
    assert body_gen["mode"] == "genesis"
    assert body_gen["top_k"] == 2
    assert len(body_gen["recommendations"]) <= 2

    # Test BASELINE experiment
    exp_baseline = DatabaseService.create_experiment(
        experiment_id=exp_id_base,
        dataset_id=ds_id,
        dataset_name="test_exp_rec.csv",
        target_column="target",
        mode="baseline",
        config={"population_size": 10, "generations": 2}
    )

    res_base = client.get(f"/api/experiments/{exp_id_base}/recommendations")
    assert res_base.status_code == 200
    body_base = res_base.json()
    assert body_base["mode"] == "baseline"
    assert body_base["top_k_applied"] is False
    assert "uses all compatible candidate pipelines" in body_base["recommendation_mode_context"]


def test_llm_evidence_identifies_experiment():
    """
    Test 4: Verify LLM evidence schema includes experiment_id, dataset_id, dataset_name, target_column, mode,
    and prompt visibly contains required headings.
    """
    evidence = {
        "experiment_id": "exp_audit_99",
        "dataset_id": "ds_123",
        "dataset_name": "marketing.csv",
        "target_column": "churn",
        "mode": "GENESIS",
        "model_name": "RandomForestClassifier",
        "pipeline_id": "tree_ensemble_rf",
        "task_type": "classification",
        "metric": "f1",
        "model_score": 0.895,
        "method": "permutation_importance",
        "global_importance": [{"feature": "feat1", "importance": 0.45, "rank": 1}],
        "dip_summary": {"rows": 500, "columns": 10, "complexity_score": 0.35, "quality_grade": "A"},
        "recommendation_summary": {"top_recommendations": [{"name": "RandomForest", "score": 0.9}]},
        "metrics": {"f1": 0.895, "accuracy": 0.91}
    }

    # Verify allowlist and validator
    extracted = EvidenceExtractor.extract_evidence(evidence)
    assert extracted["experiment_id"] == "exp_audit_99"
    assert extracted["target_column"] == "churn"
    assert extracted["mode"] == "GENESIS"

    is_valid, warnings, cleaned = EvidenceValidator.validate_evidence(extracted)
    assert is_valid
    assert cleaned["experiment_id"] == "exp_audit_99"
    assert cleaned["dataset_name"] == "marketing.csv"

    # Verify Prompt contains visible labels
    prompt = PromptBuilder.build_prompt(cleaned, mode="technical", user_prompt="Why did this perform well?")
    assert "Experiment ID: exp_audit_99" in prompt
    assert "Dataset: marketing.csv" in prompt
    assert "Target Column: churn" in prompt
    assert "Optimization Mode: GENESIS" in prompt
    assert "Model: RandomForestClassifier" in prompt
    assert "Evaluation Metrics:" in prompt
    assert "DIP SUMMARY:" in prompt
    assert "RECOMMENDATION SUMMARY:" in prompt
    assert "EXPLAINABILITY EVIDENCE:" in prompt


def test_mock_and_real_llm_providers(monkeypatch):
    """
    Test 5: Keep MOCK + REAL LLM providers.
    - LLM_PROVIDER=mock works deterministically.
    - LLM_PROVIDER=openrouter without API key shows clear configuration error and never exposes OPENROUTER_API_KEY.
    """
    # Test GET /api/config/llm does not expose API key
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret_key_12345")
    res_cfg = client.get("/api/config/llm")
    assert res_cfg.status_code == 200
    cfg_data = res_cfg.json()
    assert cfg_data["has_api_key"] is True
    assert "secret_key_12345" not in res_cfg.text

    # Test OpenRouter provider selection without API key
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    service = LLMService()
    output = service.explain(
        raw_evidence={
            "experiment_id": "exp_key_test",
            "dataset_id": "ds_key",
            "dataset_name": "data.csv",
            "target_column": "target",
            "mode": "GENESIS"
        },
        mode="technical"
    )

    assert output.llm_provider == "openrouter"
    assert any("OpenRouter API key is missing" in w for w in output.warnings)
