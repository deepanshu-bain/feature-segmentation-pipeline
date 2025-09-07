# Disclaimer

* Bain may recommend or share with you code or third party products, including, but not limited to open source software (the “third party products”), as a courtesy only.
* Any use or implementation by you of the code or third party products is at your sole and absolute risk and liability. No responsibility or liability whatsoever is accepted by Bain for such use or implementation.
* You are solely responsible for complying with all terms of third party product licenses.
* The code and/or third party products are provided “as is”, without warranty of any kind.
* Bain makes no representations or warranties of any kind, whether express or implied (either in fact, statutory or by operation of law) with respect to the code or third party products, including, but not limited to, merchantability, fitness for a particular purpose, title, non-infringement, misappropriation of intellectual property rights of a third party and suitability, all of which are expressly disclaimed.
* Bain does not warrant that the code or third party products or services provided by Bain shall be virus free or that the use thereof shall be uninterrupted or error-free.
* Bain does not have any duty to provide maintenance or support, including, but not limited to, updates, supplements or bug fixes.

---

# OC Blitz Feature & Segmentation Pipeline

This repository contains two modular, config-driven pipelines for segmentation:

* **Feature Pipeline:**
  Generates customer and order-level features across OwnApp, Swiggy, and InStore channels.

* **Segmentation Pipeline:**
  Assigns rule-based segments and micro-segments (KMeans) using pre-trained models and the feature outputs.

---

## Quick Start

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Run the full workflow (features + segmentation) with a single command:**

```bash
python main.py --feature_pipeline_config feature_pipeline_config.json --segmentation_config segmentation_config.json
```

* This will sequentially:

  * Generate features for all channels (OwnApp, Swiggy, InStore)
  * Create customer-level feature output
  * Run segmentation (rule-based and micro-segmentation) on those features

---
## Environment & Dependencies

* **pandas + pyarrow** — Parquet IO
* **dask** - To process instore data 
* **s3fs** — enables `pd.read_parquet("s3://...")`
* **boto3** — loads pickled KMeans payloads directly from S3
* **scikit‑learn** — `KMeans`, `StandardScaler` for micro‑segment prediction

> Works in SageMaker / conda / venv environments.

---

## Project Structure

```
SEGMENTATION  
├── features  
│   └── src  
│       └── main  
│           ├── utils  
│           │   ├── __init__.py  
│           │   ├── instore_pipelines_utils.py  
│           │   ├── pipelines_utils.py  
│           │   └── __init__.py  
│           ├── customer_feature_pipeline.py  
│           ├── order_feature_pipeline.py  
│           └── run_pipeline.py  
│           └── __init__.py  
├── segmentation  
│   └── src  
│       └── main  
│           ├── __init__.py  
│           └── segmentation.py  
│       └── __init__.py    
├── main.py  
├── README.md  
├── requirements.txt  
├── feature_pipeline_config.json
└── segmentation_config.json  

```


---

## 1. Feature Pipeline
### **OrderFeaturePipeline**

- **Runs separately for each channel:** OwnApp, Swiggy, InStore
- **Inputs:** Raw channel data (CSV/Parquet from S3)
- **Outputs:** Feature-rich Parquet per channel (S3)

### **CustomerFeaturePipeline**

- **Inputs:** The three output files from above
- **Builds:** Unified, L12M customer-level features for segmentation
- **Output:** Final customer features Parquet (S3)

## 2. SegmentRunner — Rule‑Based Segmentation + Micro‑Segments (KMeans)

A small, config‑driven pipeline for customer segmentation that combines **deterministic, rule‑based segments** with **KMeans micro‑segments** using pre‑trained models stored on S3. The model is trained on one year of data i.e. 01-07-2024 to 30-06-2025. Use this for prediction on one year data.



**SegmentRunner** provides an end‑to‑end segmentation workflow that:

1. Loads features from Parquet (local path or **S3**).
2. Builds **rule‑based segments** (CART‑inspired rules) across three order bands: `1_5`, `6_15`, and `15_plus`.
3. Saves segment assignments to Parquet.
4. Runs **micro‑segmentation** inside selected rule segments using **pre‑trained KMeans models** stored as pickles on S3.
5. Store the final mapping of segments and microsegments with algonomy\_id in S3.

---

## Configuration 
### Feature Pipeline Conifg
All configurations for feature pipelines live in `feature_pipeline_config.json`

Key sections:

- `ownapp`, `swiggy`, `instore`: Input data, columns, output path for each order pipeline
- `customer`: Cutoff date, output path, and will auto-link outputs of other channels
- `common`: Date windows, price bins, mappings, category column lists

**Example **`feature_pipeline_config.json`** extract:**

```json
{
  "ownapp": { "orders_data_path": "...", ... },
  "swiggy": { "orders_data_path": "...", ... },
  "instore": { "orders_file_pattern": "...", ... },
  "customer": {
    "cutoff_date": "2024-07-01",
    "output_path": "s3://.../users_feature.parquet"
  },
  "common": {
    "current_run_date": "2025-07-01",
    "months_offset": 27,
  }
}
```

### Segment Config
The pipeline is **config‑driven**. Below is a compact example (adjust paths and lists to your environment). The JSON below is valid and ready to copy.

#### Config Example

```json
{
  "input_path": "s3://blitz-mcdelivery/features/users/ownapp/20250819_features_for_chaid_tree_L12M/L12M_20250819_features_for_chaid_tree.parquet",
  "output_segments_path": "s3://blitz-mcdelivery/Adhoc/segmentation.par",

  "segment_1_5_1_2_model": "s3://blitz-mcdelivery/Modelling/Artifacts/kmeans_segmentation_1_5_1_2.pkl",
  "segment_1_5_3_model":   "s3://blitz-mcdelivery/Modelling/Artifacts/kmeans_segmentation_1_5_3.pkl",

  "cols_to_keep": [
    "ownapp_order_count",
    "unique_category_base_products_count"
  ],

  "convert_to_flag": [
    "omnichannel_Breakfast_order_count",
    "omnichannel_Lunch_order_count"
  ],

  "rename_dict": {
    "unique_category_base_products_count": "unique_cat_base_count",
    "unique_dri_category_count": "unique_cat_dri_count"
  },

  "rename_dict_microsegments": {
    "categ_base_product_bfast_muffin_count": "base_bfast_muffin_count",
    "categ_base_product_bic_count": "base_bic_count"
  },

  "micro_plan": [
    {
      "name": "Seg1_5_1_and_2",
      "source_segments": ["Seg1_5_1", "Seg1_5_2"],
      "model_key": "segment_1_5_1_2_model",
      "enabled": true
    }
  ]
}
```

#### Config Field Reference

* **`input_path`** — Parquet with the analytical table (local or S3).
* **`output_segments_path`** — where to save rule‑based segment assignments (Parquet).
* **`output_microsegments_path`** — where to save final micro‑segments and segments mapping with `algonomy_id` and features (Parquet).
* **`segment_*_model`** — S3 paths to micro‑segment pickle **payloads** (a dict with keys: `kmeans`, `scaler`, `features`).
* **`cols_to_keep`** — base columns to load.
* **`convert_to_flag`** — columns converted to `*_flag` (`value > 0 → 1`). The source columns are dropped.
* **`rename_dict`** — renames after stripping the `omnichannel_` prefix.
* **`rename_dict_microsegments`** — additional renames before micro‑segment predictions (to match model features).
* **`micro_plan`** — list of micro‑seg tasks:

  * `name` → label for outputs (e.g., `Seg1_5_1_and_2`)
  * `source_segments` → which rule segments feed this micro‑seg
  * `model_key` → key mapping to an S3 pickle path in config
  * `enabled` (optional, default `true`)

---

## SegmentRunner Class

### Key Attributes

* `target_col = "ownapp_order_count"`
* `id_col = "algonomy_id"`
* `lower_q = 0.01`, `upper_q = 0.99` (for outlier capping, when applied)

### Public Methods

* **`load_data() -> DataFrame`**
  Reads Parquet, filters (`ownapp_order_count ≤ 100` and `unique_dri_category_count ≠ 0`), and returns only columns in `cols_to_keep + convert_to_flag`.

* **`create_flags(df) -> DataFrame`**
  For each column in `convert_to_flag` (excluding the ID), create `*_flag = 1{col > 0}` and drop the source column.

* **`rename_cols(df) -> DataFrame`**
  Strips `omnichannel_` prefix and applies `rename_dict`.

* **`run_rule_segmentation(df_in) -> dict`**
  Splits data into order bands (`1_5`, `6_15`, `15_plus`) and applies rule sets. Returns per‑band:

  * `summary` — Segment, Customer Count, Total Orders, Avg Orders per Customer
  * `assignments` — (`algonomy_id`, `Segment`)

* **`save_segments(df)`**
  Saves the combined assignments to `output_segments_path` (Parquet).

---

## Pipelines in `main.py`

### A) Order Pipeline Steps (for ownapp/swiggy/instore)

- **Loads data** (CSV/Parquet, Dask for large S3 files)
- Cleans, filters, and localizes datetimes
- Maps centralized customer/store IDs**
- Maps items to categories (Driver Category and Base Product Category) 
- Removes 
    - promo itmes : the focus is on what a customer selected from the menu (not what they got as freebie)
    - choice : If any of the choice item are appearing as line item (e.g. in a meal if coke is appearing a different line item in order_items) remove that.
    - unmapped items : items for which there is no mapping available
- **Engineers  features:**
  - Price buckets
  - DRI and base categories (one-hot counts)
  - Promo/discount metrics
  - Delivery and time-of-day features
- **Writes Parquet to S3**

### B) Customer Pipeline

1. **Loads all three order outputs**
2. Filters to OwnApp users only (cross-channel user stitching)
3. Standardizes timestamp, categories, price calculations
4. Filters to cutoff window (e.g. last 12 months, from config)
5. Concatenates all orders, builds channel label
6. **Feature engineering:**
   - **Discount metrics** (omnichannel avg)
   - **Unique base/DRI categories** (flatten and dedupe lists)
   - **Order metrics:** count, recency, tenure, frequency, first/last dates per channel
   - **Daypart metrics:** omnichannel and ownapp, all combos (Breakfast/Lunch/...)
   - **Price bucket counts** omnichannel and ownapp
   - **Category order count one-hots** omnichannel and ownapp
   - **OwnApp avg delivery time** (adjusted to IST)
   - **Channel count** (how many distinct channels per user)
7. **Writes customer feature file to S3/local**

### C) Rule Pipeline

**Signature**: `SegmentRunner.run_rule_pipeline() → (DataFrame, DataFrame)`

**Does**:
`load_data → create_flags → rename_cols → run_rule_segmentation → concatenate assignments`

**Returns**:

* `original_df` — prepared feature table used for segmentation
* `combined_segments_df` — two columns: `algonomy_id`, `Segment` (one row per id)

### D) Micro‑Segmentation

**Signature**: `SegmentRunner.run_microsegments_from_config(df_base, df_segments) → DataFrame`

**Process (per micro‑plan item):**

* Subset customers by `source_segments`
* cap outliers (for predefined segments)
* Rename columns via `rename_dict_microsegments`
* Load `{kmeans, scaler, features}` from S3 and predict `microcluster`

**Returns**:

* `microsegments_df` with `algonomy_id`, `microcluster`, `Segment` (plan name)

### E) Persisting Micro‑Segmentation Outputs

```python
# Example: inside main.py after creating `base_ms_df`
base_ms_df.to_parquet(runner.config["output_microsegments_path"], index=False)
```

---

## Rule Sets (Summary)

* **Band `1_5`**
  Segments `Seg1_5_1` … `Seg1_5_8` using:
  `unique_cat_base_count`, `Weekend_flag`, `Weekday_flag`, `channel_count`, `Premium_price_flag`.

* **Band `6_15`**
  Segments `Seg_6_15_01` … `Seg_6_15_10` using:
  `unique_cat_base_count`, `Weekend_flag`, `Lunch_flag`, `Premium_price_flag`, `Late_Night_flag`, `channel_count`.

* **Band `15_plus`**
  Segments `Seg15_plus_1` … `Seg15_plus_9` using:
  `unique_cat_base_count`, `Breakfast_flag`, `Late_Night_flag`, `Evening_flag`.

> Edit `_rules_for_band(band)` to maintain segmentation logic in one place.

---

## Micro Plan Design

Add as many micro‑segment jobs as you need in `micro_plan`. Example:

```json
"micro_plan": [
  {
    "name": "Seg1_5_1_and_2",
    "source_segments": ["Seg1_5_1", "Seg1_5_2"],
    "model_key": "segment_1_5_1_2_model",
    "enabled": true
  },
  {
    "name": "Seg1_5_3",
    "source_segments": ["Seg1_5_3"],
    "model_key": "segment_1_5_3_model"
  }
]
```

**Pickle payload format** (each `segment_*_model` points to a file containing this dict):

```python
{
    "kmeans": <sklearn.cluster.KMeans>,
    "scaler": <sklearn.preprocessing.StandardScaler>,
    "features": ["col_a", "col_b", ...]  # exact training feature order
}
```

---

## Outlier Capping

A helper `_cap_outliers()` (1st/99th percentile clipping) is used **conditionally** within `_prep_micro_base()` for selected segments (default: `{"Seg1_5_6", "Seg_6_15_10", "Seg15_plus_9"}`). Modify that set if you want capping in other micro‑pipelines.

---

## Outputs

* **Rule‑based Segments Parquet** (`output_segments_path`)
  Columns: `algonomy_id`, `Segment` (e.g., `Seg_6_15_04`), one row per id.

* **`microsegments_df`** (`output_microsegments_path`)
  Columns: `algonomy_id`, `microcluster` (string), `Segment` (plan `name`, e.g., `Seg1_5_1_and_2`), and `features`.

---

## Troubleshooting

* **Invalid JSON** — validate `config.json` (commas, braces).
* **S3 access** — ensure IAM allows `s3:GetObject` for model paths; `s3fs` required for `pd.read_parquet("s3://...")`.
* **Missing features** — `_predict_from_model_key` raises with missing column names—check `rename_dict` / `rename_dict_microsegments`.
* **Empty subsets** — a micro‑plan item with no matching users will produce 0 rows.
* **Pickle version warnings** — align scikit‑learn versions where possible.

---

## Notes & Best Practices

* Keep `algonomy_id` intact throughout (never cap/rename/drop).
* Maintain rules centrally in `_rules_for_band()` to avoid drift.
* Ensure micro‑models were trained with the same post‑rename feature names.

---

