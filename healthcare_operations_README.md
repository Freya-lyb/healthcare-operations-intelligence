# Healthcare Operations Intelligence: Emergency Department Optimization & Patient Flow Redesign

## Overview

A healthcare operations intelligence case study exploring the structural drivers of avoidable emergency department utilization using large-scale encounter, provider, and Social Determinants of Health (SDOH) data.

Rather than beginning with a predefined business question, this project started with broad operational hypotheses around healthcare accessibility, provider utilization, patient flow, and healthcare resource allocation within a large Kansas health system. Through iterative exploratory analysis, feature engineering, provider workload analysis, and systems-level investigation, emergency department overcrowding emerged as the strongest operational signal in the dataset.

Using 947K patients, 7.7M encounters, 1.53M diagnoses, and the NYU ED Algorithm on 202K emergency department visits, this project identified that 55.6% of ED utilization was broadly avoidable. The analysis revealed that avoidable ED burden was not primarily driven by patient behavior, but by structural access failures across both demand-side and supply-side healthcare systems.

---

## Problem Context

Emergency departments across many healthcare systems are increasingly overloaded with non-emergent and preventable visits. While ED overcrowding is often framed as a patient behavior problem, this project approached the issue as a systems coordination and operational access problem.

The core question evolved into:

> What patient and system factors drive avoidable emergency department utilization, and how can operational interventions reduce ED burden without requiring major policy or staffing changes?

---

## Project Evolution & Analytical Approach

One of the defining aspects of this project was that the problem statement was not predefined.

The project began with a broad healthcare operations dataset and an open-ended challenge involving healthcare utilization, provider systems, and patient outcomes. Instead of immediately building predictive models, the team first explored multiple operational hypotheses through iterative exploratory analysis and cross-system investigation.

Through repeated exploratory analysis, systems framing, and feature engineering, emergency department overcrowding consistently emerged as the strongest operational signal across the dataset.

This led to a narrowed project scope centered on:

# Emergency Department Optimization & Patient Flow Redesign

The final analytical workflow evolved into:

```text
Operational hypotheses
→ Data cleaning & integration
→ Feature engineering
→ Exploratory analysis
→ Demand-side analysis
→ Supply-side analysis
→ Intervention design
→ Predictive validation
→ Operational recommendations
→ Visualization & reporting
```

---

## Key Findings

- 55.6% of ED utilization was broadly avoidable
- Weekend ED rates surged 8.7× during clinic closure periods
- Patients without PCP relationships had 2.3× higher ED utilization
- MyChart activation corresponded to 63.6% lower ED utilization
- PCP systems operated near practical capacity ceilings
- Specialist workload distribution was structurally imbalanced

---

## Intervention Design

### Demand-Side Solutions

#### D1 — Dynamic Clinic Hours

- Weekend clinic expansion
- Extended evening hours
- Flexible staffing during high-volume periods

#### D2 — Upgraded Triage Systems

- Licensed clinical staff at intake
- NYU ED Algorithm-informed triage routing
- Immediate outpatient redirection pathways

#### D3 — MyChart + Community Resource Navigation

- MyChart activation support
- Community resource booklet distribution
- Symptom-based healthcare routing guidance

### Supply-Side Solution

#### S1 — Specialist Rebalancing

- Reallocation of underutilized PCP-adjacent specialists
- Expanded outpatient support coverage
- Weekend clinic staffing support

---

## Modeling & Predictive Validation

### Random Forest Modeling

- 30-day readmission prediction
- 5-fold cross-validation
- AUC = 0.953

### Clustering Analysis

KMeans clustering identified three distinct patient populations:

1. High-frequency low-SDOH patients
2. High-risk cycling patients
3. Low-frequency younger populations

---

## Visualization & Operational Storytelling

The project emphasized executive-facing operational storytelling through visualization.

### Visual Analysis Components

- Geographic ED concentration maps
- PCP workload distributions
- Sankey patient flow diagrams
- SDOH correlation heatmaps
- NYU ED classification visuals
- Intervention effect estimation charts

---

## Technical Stack

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- matplotlib
- plotly

---

## Repository Structure

```text
healthcare-operations-intelligence/
│
├── architecture/
├── data_dictionary/
├── data_engineering/
├── analytics/
├── outputs/
├── visualizations/
├── reports/
├── assets/
├── presentation/
└── README.md
```

---

## Project Positioning

This repository is intentionally positioned as:

- a healthcare operations intelligence case study
- an operational analytics & systems redesign project
- a healthcare demand-supply analysis framework
- a patient flow optimization investigation
- a PM + analytics + operational intelligence portfolio project
