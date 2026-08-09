# System Design

## Overview
GENESIS-AI introduces the Dataset Intelligence Profile (DIP) between data ingestion and AutoML search space generation.

## DIP Workflow
1. Dataset Ingestion (CSV File)
2. CSV Validation (existence, format, non-empty, parseability)
3. Target Column Validation (existence, non-null, unique class count >= 2)
4. Schema & Data Type Extraction (numeric, categorical, boolean, datetime counts/ratios)
5. Quality Metrics Calculation (missingness, duplicates)
6. Statistical Profiling (IQR outliers, skewness, Pearson correlation matrix)
7. Target & Task Analysis (heuristic classification vs regression detection, class imbalance ratio, entropy)
8. Dimensionality Indicators (feature-to-sample ratio)
9. GENESIS DIP Complexity Score Calculation (0-10 normalized weighted score)
10. JSON Serialization & SHA-256 Dataset Hashing
