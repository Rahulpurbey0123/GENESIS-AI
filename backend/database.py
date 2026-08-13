"""
Database persistence module for GENESIS-AI Week 8 Application.
Provides thread-safe SQLite storage for datasets, DIP profiles, experiments, results, explanations, and chats.
"""

import json
import math
import sqlite3
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("genesis.database")

DB_DIR = Path("data")
DB_PATH = DB_DIR / "genesis.db"


def get_db_connection() -> sqlite3.Connection:
    """Create and return a thread-safe connection to the SQLite database."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS datasets (
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

    CREATE TABLE IF NOT EXISTS dip_profiles (
        dataset_id TEXT NOT NULL,
        target_column TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (dataset_id, target_column),
        FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        dataset_name TEXT NOT NULL,
        target_column TEXT NOT NULL,
        mode TEXT NOT NULL,
        status TEXT NOT NULL,
        config_json TEXT NOT NULL,
        progress_json TEXT NOT NULL,
        error_message TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS experiment_results (
        experiment_id TEXT PRIMARY KEY,
        best_pipeline_json TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        efficiency_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS experiment_explanations (
        experiment_id TEXT PRIMARY KEY,
        shap_json TEXT NOT NULL,
        global_importance_json TEXT NOT NULL,
        local_importance_json TEXT NOT NULL,
        evaluation_plots_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        prompt TEXT NOT NULL,
        response_json TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    """)
    conn.commit()

    # Migration check for existing dip_profiles schema (migrates single dataset_id PK to composite PK)
    cursor.execute("PRAGMA table_info(dip_profiles)")
    info = cursor.fetchall()
    if info:
        pk_cols = [row["name"] for row in info if row["pk"] > 0]
        if pk_cols == ["dataset_id"]:
            logger.info("Migrating dip_profiles table to composite primary key (dataset_id, target_column)...")
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

    conn.close()
    logger.info(f"Database initialized successfully at {DB_PATH}")


# Initialize DB on module load
init_db()


class DatabaseService:
    """Helper service for database CRUD operations."""

    @staticmethod
    def save_dataset(
        dataset_id: str,
        name: str,
        content_hash: str,
        filepath: str,
        rows: int,
        columns: int,
        features: List[str],
        suggested_target: Optional[str]
    ) -> Dict[str, Any]:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Deduplication check: return existing dataset record if content_hash matches
        cursor.execute("SELECT * FROM datasets WHERE content_hash = ?", (content_hash,))
        existing = cursor.fetchone()
        if existing:
            existing_id = existing["id"]
            cursor.execute(
                """
                UPDATE datasets
                SET name = ?, filepath = ?, rows = ?, columns = ?, features_json = ?, suggested_target = COALESCE(?, suggested_target)
                WHERE id = ?
                """,
                (name, filepath, rows, columns, json.dumps(features), suggested_target, existing_id)
            )
            conn.commit()
            conn.close()
            return {
                "id": existing_id,
                "name": name,
                "content_hash": content_hash,
                "filepath": filepath,
                "rows": rows,
                "columns": columns,
                "features": features,
                "suggested_target": suggested_target or existing["suggested_target"],
                "created_at": existing["created_at"]
            }

        created_at = datetime.now(timezone.utc).isoformat()
        features_json = json.dumps(features)
        
        cursor.execute(
            """
            INSERT INTO datasets (id, name, content_hash, filepath, rows, columns, features_json, suggested_target, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, name, content_hash, filepath, rows, columns, features_json, suggested_target, created_at)
        )
        conn.commit()
        conn.close()

        return {
            "id": dataset_id,
            "name": name,
            "content_hash": content_hash,
            "filepath": filepath,
            "rows": rows,
            "columns": columns,
            "features": features,
            "suggested_target": suggested_target,
            "created_at": created_at
        }

    @staticmethod
    def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "id": row["id"],
            "name": row["name"],
            "content_hash": row["content_hash"],
            "filepath": row["filepath"],
            "rows": row["rows"],
            "columns": row["columns"],
            "features": json.loads(row["features_json"]),
            "suggested_target": row["suggested_target"],
            "created_at": row["created_at"]
        }

    @staticmethod
    def save_dip_profile(dataset_id: str, target_column: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        conn = get_db_connection()
        created_at = datetime.now(timezone.utc).isoformat()
        profile_json = json.dumps(profile)
        
        cursor = conn.cursor()
        cursor.execute(
            "SELECT dataset_id FROM dip_profiles WHERE dataset_id = ? AND target_column = ?",
            (dataset_id, target_column)
        )
        if cursor.fetchone():
            cursor.execute(
                "UPDATE dip_profiles SET profile_json = ?, created_at = ? WHERE dataset_id = ? AND target_column = ?",
                (profile_json, created_at, dataset_id, target_column)
            )
        else:
            cursor.execute(
                "INSERT OR REPLACE INTO dip_profiles (dataset_id, target_column, profile_json, created_at) VALUES (?, ?, ?, ?)",
                (dataset_id, target_column, profile_json, created_at)
            )
        conn.commit()
        conn.close()
        return profile

    @staticmethod
    def get_dip_profile(dataset_id: str, target_column: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        if target_column:
            cursor.execute(
                "SELECT profile_json FROM dip_profiles WHERE dataset_id = ? AND target_column = ? ORDER BY created_at DESC",
                (dataset_id, target_column)
            )
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            return json.loads(row["profile_json"])
        else:
            cursor.execute(
                "SELECT profile_json FROM dip_profiles WHERE dataset_id = ? ORDER BY created_at DESC",
                (dataset_id,)
            )
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return None
            if len(rows) > 1:
                raise ValueError("target_column is required because multiple DIP profiles exist for this dataset.")
            return json.loads(rows[0]["profile_json"])

    @staticmethod
    def create_experiment(
        experiment_id: str,
        dataset_id: str,
        dataset_name: str,
        target_column: str,
        mode: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        conn = get_db_connection()
        created_at = datetime.now(timezone.utc).isoformat()
        initial_progress = {
            "current_generation": 0,
            "max_generations": config.get("generations", 10),
            "evaluated_pipelines": 0,
            "best_score": None,
            "runtime": None,
            "search_space_reduction": 0.0,
            "history": []
        }

        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments (id, dataset_id, dataset_name, target_column, mode, status, config_json, progress_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                dataset_id,
                dataset_name,
                target_column,
                mode,
                "RUNNING",
                json.dumps(config),
                json.dumps(initial_progress),
                created_at
            )
        )
        conn.commit()
        conn.close()

        return {
            "id": experiment_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "mode": mode,
            "status": "RUNNING",
            "config": config,
            "progress": initial_progress,
            "created_at": created_at
        }

    @staticmethod
    def update_experiment_progress(
        experiment_id: str,
        status: str,
        progress: Dict[str, Any],
        error_message: Optional[str] = None
    ) -> None:
        conn = get_db_connection()
        cursor = conn.cursor()
        completed_at = datetime.now(timezone.utc).isoformat() if status in ("COMPLETED", "FAILED", "CANCELLED") else None
        
        cursor.execute(
            """
            UPDATE experiments
            SET status = ?, progress_json = ?, error_message = ?, completed_at = COALESCE(?, completed_at)
            WHERE id = ?
            """,
            (status, json.dumps(progress), error_message, completed_at, experiment_id)
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "dataset_name": row["dataset_name"],
            "target_column": row["target_column"],
            "mode": row["mode"],
            "status": row["status"],
            "config": json.loads(row["config_json"]),
            "progress": json.loads(row["progress_json"]),
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"]
        }

    @staticmethod
    def list_experiments() -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        experiments = []
        for row in rows:
            progress = json.loads(row["progress_json"])
            raw_score = progress.get("best_score", None) if isinstance(progress, dict) else None
            best_score = None
            if raw_score is not None:
                try:
                    val = float(raw_score)
                    if not (math.isnan(val) or math.isinf(val)):
                        best_score = val
                except (ValueError, TypeError):
                    best_score = None

            experiments.append({
                "id": row["id"],
                "dataset_id": row["dataset_id"],
                "dataset_name": row["dataset_name"],
                "target_column": row["target_column"],
                "mode": row["mode"],
                "status": row["status"],
                "best_score": best_score,
                "runtime": progress.get("runtime", 0.0) if isinstance(progress, dict) else (float(progress) if isinstance(progress, (int, float)) else 0.0),
                "error_message": row["error_message"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"]
            })
        return experiments


    @staticmethod
    def save_experiment_results(
        experiment_id: str,
        best_pipeline: Dict[str, Any],
        metrics: Dict[str, Any],
        efficiency: Dict[str, Any]
    ) -> None:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO experiment_results (experiment_id, best_pipeline_json, metrics_json, efficiency_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                experiment_id,
                json.dumps(best_pipeline),
                json.dumps(metrics),
                json.dumps(efficiency)
            )
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get_experiment_results(experiment_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiment_results WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "experiment_id": row["experiment_id"],
            "best_pipeline": json.loads(row["best_pipeline_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "efficiency": json.loads(row["efficiency_json"])
        }

    @staticmethod
    def save_experiment_explanations(
        experiment_id: str,
        shap_data: Dict[str, Any],
        global_importance: Dict[str, Any],
        local_importance: Dict[str, Any],
        evaluation_plots: Dict[str, Any]
    ) -> None:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO experiment_explanations (experiment_id, shap_json, global_importance_json, local_importance_json, evaluation_plots_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                json.dumps(shap_data),
                json.dumps(global_importance),
                json.dumps(local_importance),
                json.dumps(evaluation_plots)
            )
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get_experiment_explanations(experiment_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiment_explanations WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "experiment_id": row["experiment_id"],
            "shap": json.loads(row["shap_json"]),
            "global_importance": json.loads(row["global_importance_json"]),
            "local_importance": json.loads(row["local_importance_json"]),
            "evaluation_plots": json.loads(row["evaluation_plots_json"])
        }

    @staticmethod
    def save_chat(experiment_id: str, chat_id: str, prompt: str, response: Dict[str, Any]) -> None:
        conn = get_db_connection()
        cursor = conn.cursor()
        timestamp = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """
            INSERT INTO chats (id, experiment_id, prompt, response_json, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_id, experiment_id, prompt, json.dumps(response), timestamp)
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get_chats(experiment_id: str) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chats WHERE experiment_id = ? ORDER BY timestamp ASC", (experiment_id,))
        rows = cursor.fetchall()
        conn.close()

        chats = []
        for r in rows:
            chats.append({
                "id": r["id"],
                "experiment_id": r["experiment_id"],
                "prompt": r["prompt"],
                "response": json.loads(r["response_json"]),
                "timestamp": r["timestamp"]
            })
        return chats
