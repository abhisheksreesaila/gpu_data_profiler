# Modular Data Profiler 2025 🚀

*A Hackathon Project — Honoring the Unsung Heroes of Machine Learning*

## 🎯 Overview

This project shines a light on the essential, yet often underappreciated, work of data profiling and preprocessing in machine learning. While sophisticated models often steal the show, it’s the data engineers and scientists who ensure that clean, well-understood data powers every ML pipeline.

## 💡 Vision

Our goal is to build a dataframe library—drawing inspiration from NVIDIA RAPIDS—that leverages GPU acceleration for high-performance computations.

## 📊 Numerical Analysis

We have implemented GPU-optimized functions for `MAX`, `MIN`, and `MEAN` calculations.

**Input Data Shape:** `(100,)`  
**Sample Input:** `[44. 41. 70. 60.  2. 57. 76. 77. 56. 35.]`

| Operation | Result | Expected | Status                  |
|-----------|--------|----------|-------------------------|
| Max       | 99.0   | 99.0     | ✅ Verification passed! |
| Min       | 0.0    | 0.0      | ✅ Verification passed! |
| Mean      | 49.5   | 49.5     | ✅ Verification passed! |
