# 🗞️ News Recommender System

This project implements and compares multiple **news recommendation models** on the **Microsoft News Dataset (MIND)**.  
It evaluates both **ranking performance** and **carbon emissions**, highlighting the trade-off between accuracy and sustainability.

---

### 🧠 Project Overview

The system predicts which news articles users are most likely to click, using four model families:

- **Baseline:** Popularity-based ranking by recent clicks or CTR  
- **Content-Based:** TF-IDF and Sentence Transformer similarity  
- **Collaborative Filtering:** Matrix factorization (BPR)  
- **Hybrid:** Combines models for improved personalization  

All models are evaluated using **AUC**, **MRR**, **nDCG@5**, and **nDCG@10**, while carbon emissions are tracked via **CodeCarbon**.

---

### ⚙️ Technologies

- Python 3.11, Pandas, NumPy, scikit-learn, TensorFlow  
- Sentence-Transformers, CodeCarbon  
- Typer (CLI), Hydra (configuration)  
- Poetry (dependency management)

---

### 📊 Key Results

| Model | AUC | MRR | nDCG@5 | nDCG@10 | CO₂ Emissions (kg CO₂eq) |
|:------|:---:|:---:|:------:|:-------:|:------------------------:|
| **Baseline (CTR)** | 0.650 | 0.312 | 0.341 | 0.403 | 0.000013 |
| Content-Based | 0.639 | 0.311 | 0.339 | 0.398 | 0.000163 |
| Collaborative | 0.596 | 0.298 | 0.319 | 0.369 | 0.000013 |
| **Hybrid** | **0.766** | **0.404** | **0.448** | **0.507** | **0.000189** |


**Insight:** The hybrid model achieved the best accuracy but consumed the most energy.  
The CTR-based baseline offered the best **accuracy-to-emissions ratio**.

---

### 🧭 Structure
The project builds on the **Cookiecutter Data Science** structure, with a few additional modules for modeling and sustainability tracking.
```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         news_recommender_system and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── news_recommender_system   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes news_recommender_system a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

### 🌱 Sustainability Focus

Each model run logs its **energy consumption**, **runtime**, and **CO₂ emissions** using **CodeCarbon**.  
This enables a data-driven evaluation of both **performance** and **environmental impact**.

---

### 🧩 Key Takeaways

- **Recent-click-based CTR models** are simple, fast, and eco-efficient.  
- **Embedding-based models** improve personalization but increase carbon cost.  
- **Sustainable AI evaluation** should consider both ranking quality and resource efficiency.

