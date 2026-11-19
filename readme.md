# Automated Customer Support Root-Cause Discovery & Ticket Routing

An end-to-end Natural Language Processing (NLP) and Machine Learning pipeline that transforms raw, unlabeled customer support messages into structured operational categories and routes incoming tickets automatically with built-in confidence guardrails.

---

## Business Problem & Economic Value

Customer support teams receive thousands of unstructured messages daily. Manual reading and tagging slow down response times, hide emerging product glitches until refund requests pile up, and waste hours of staff labor.

This pipeline solves both halves of the problem in a single automated workflow:
1. **Unsupervised Root-Cause Discovery:** Automatically reads historical tickets by semantic meaning—not rigid keywords—and groups recurring complaints into clear operational buckets without requiring pre-labeled training data.
2. **Supervised Ticket Routing & Guardrails:** Trains a lightweight classifier on those discovered buckets to route new incoming messages in milliseconds, while automatically flagging low-confidence edge cases (`< 45%` confidence) for human review.

---

## Architecture & Pipeline Flow

```mermaid
flowchart LR
    A[Raw Support CSV<br/>300 Tickets] --> B[Step 1–2: Clean & EDA<br/>Avg 11.0 Words]
    B --> C[Step 3: Sentence Embeddings<br/>all-MiniLM-L6-v2 384-d]
    C --> D[Step 4: Root-Cause Discovery<br/>Agglomerative Clustering]
    D --> E[Step 5: Train & Evaluate<br/>Logistic Regression 80/20 Split]
    E --> F[Step 6: Live Routing<br/>45% Confidence Guardrail]
    F --> G[Step 7–8: Dashboard & CSVs]
```

---

## Technical Stack, Models & Algorithms

| Pipeline Component | Model / Algorithm Used | Purpose & Why It Was Chosen |
| :--- | :--- | :--- |
| **Data Processing & EDA** | `Pandas` | Cleans whitespace/duplicates and computes ticket length distributions (`mean = 11.0 words`, `range = 4–47 words`). |
| **Feature Engineering** | `all-MiniLM-L6-v2` (`Sentence-Transformers`) | Converts unstructured sentences into **384-dimensional normalized dense vectors** that capture semantic meaning rather than exact word matches. |
| **Unsupervised Clustering** | `AgglomerativeClustering` (`metric="cosine"`, `linkage="average"`, `distance_threshold=0.55`) | Discovers problem clusters automatically from the bottom up without requiring a pre-guessed cluster count ($K$). |
| **Cluster Auto-Labeling** | `TfidfVectorizer` | Extracts the top 3 highest-scoring descriptive keywords per cluster to generate human-readable category names. |
| **Supervised Classifier** | `LogisticRegression` (`max_iter=1000`) | Learns decision boundaries over the 384-d embeddings to predict categories for brand-new unseen customer messages. |
| **Visualization & Reporting** | `Matplotlib` | Generates a 2-panel executive dashboard comparing historical issue volume against live routing confidence thresholds. |

---

## Dataset & Experimental Results

### 1. Dataset & Train/Test Split (`PolyAI/banking77`)
* **Total Ingested & Clean Tickets:** `300` real-world customer banking queries (`raw_customer_tickets.csv`).
* **Exploratory Metrics:** Average length of `11.0 words` per ticket (minimum `4 words`, maximum `47 words`).
* **Unsupervised Discovery:** Identified **15 distinct problem clusters** across the 300 tickets.
* **Model Training & Validation Split:** Filtered to recurring themes with $\ge 4$ examples (`288` eligible tickets) and split **80% Training (`230` tickets)** / **20% Validation (`58` tickets)** using stratified sampling.
* **Validation Accuracy on Unseen Test Tickets:** **`100.0%`**

### 2. Top Discovered Root-Cause Clusters
| Discovered Problem Theme | Ticket Count | Share of Top 5 Volume |
| :--- | :---: | :---: |
| `CARD / LINK / NEW` | 89 | 31.9% |
| `EXCHANGE / RATE / RATES` | 77 | 27.6% |
| `PENDING / CASH / WITHDRAWAL` | 40 | 14.3% |
| `CURRENCIES / FIAT / CURRENCY` | 39 | 14.0% |
| `STATEMENT / FEE / EXTRA` | 34 | 12.2% |

### 3. Live Inference & Business Guardrail Test
New unseen messages are scored and routed against a **`45.0%` minimum confidence threshold**:

| New Incoming Customer Ticket | Predicted Route | Confidence | Operational Action |
| :--- | :--- | :---: | :--- |
| *"I tried paying at the grocery store but my card got rejected twice."* | `CARD / LINK / NEW` | `56.4%` | **Auto-Route** |
| *"Why is my bank transfer still pending after three days?"* | `PENDING / CASH / WITHDRAWAL` | `65.1%` | **Auto-Route** |
| *"Someone used my card in another city without my permission!"* | `CARD / LINK / NEW` | `67.0%` | **Auto-Route** |
| *"My cat walked across my keyboard and ordered a toaster."* *(Edge Case)* | `CARD / LINK / NEW` | `39.8%` | **Human Review** |

---

## Executive Dashboard Output

![Executive Dashboard](dashboard.png)

* **Panel A (Discovered Root Causes):** Ranks the highest-volume historical customer issues discovered automatically from raw text.
* **Panel B (Live Prediction Confidence & Guardrail):** Visualizes model confidence for incoming tickets against the dashed red **45% Auto-Route Threshold**, cleanly isolating out-of-domain anomalies in amber for human review.

---

## Repository Deliverables & How to Run

### Generated Files
* `raw_customer_tickets.csv` — Raw ingested dataset (`300` rows) saved prior to preprocessing.
* `clustered_historical_tickets.csv` — Full dataset enriched with `Cluster_ID`, `Problem_Theme`, and `Word_Count`.
* `live_ticket_predictions.csv` — Inference report containing `Predicted_Route`, `Confidence_%`, and `Action` flags.

### Quickstart
1. Install dependencies:
   ```bash
   pip install pandas matplotlib datasets sentence-transformers scikit-learn
   ```
2. Run the end-to-end pipeline script:
   ```bash
   python ticket_root_cause_pipeline.py
   ```
3. To test on your own business dataset, replace `raw_filename = "raw_customer_tickets.csv"` in **Step 1** with your CSV path and point `df["Message"]` to your text column.