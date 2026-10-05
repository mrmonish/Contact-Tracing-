# 🦠 TraceNet-ML: Automated Contact Tracing & Exposure Risk Classifier

## 📌 Project Context & Overview
Developed as a B.Tech CSAIML Semester 1 project, this repository implements a Supervised Machine Learning classification framework to evaluate individual infection exposure risks (categorized into High Risk vs. Low Risk protocols). Engineering predictive proximity engines mirrors backend screening pipelines used by public health clouds, smart city dashboards, and mobile tracing networks (like the Apple-Google Exposure Notification API) to map epidemiological spread dynamics.

## 🛠️ Software Stack
- **Language Stack:** Python 3
- **Algorithmic Library:** Scikit-Learn
- **Data Engineering Blocks:** Pandas

## 📊 Feature Matrix Architecture
The classification engine maps transmission boundary hazards based on these core telemetry vectors:
- `Proximity_Distance`: Measured spatial gap between the subject and an index patient, recorded in meters (Continuous)
- `Contact_Duration`: Total cumulative time metrics spent during the specific contact interaction, in minutes (Continuous)
- `Is_Indoor`: Environmental transmission factor (1 if interaction occurred within an enclosed structure, 0 if outdoors)
- **Target Output Categories:** `Exposure_Risk` (0: Low Risk/Safe, 1: High Risk/Exposure Protocol Trigger)

## 🤖 Algorithmic Baseline
The operational logic runs on a **K-Nearest Neighbors (KNN) Classifier**. KNN represents the ideal geometric baseline for spatial analytics because it maps and evaluates new encounter vectors by calculating mathematical distances against historical interaction matrices to classify final safety thresholds.
