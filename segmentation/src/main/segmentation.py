import pandas as pd
import numpy as np
import json
import os
from urllib.parse import urlparse
import boto3, pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class SegmentRunner:
    def __init__(self, config_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        config_path = os.path.join(root_dir, config_path)
        # Load the configuration file
        with open(config_path, "r") as f:
            self.config= json.load(f)
        self.data = None
        self.target_col = "ownapp_order_count"  # default target
        self.id_col = "algonomy_id"
        self.lower_q = 0.01
        self.upper_q = 0.99

    # ---------- Data & flags ----------
    def load_data(self):
        print("[INFO] Loading data...")
        self.data = pd.read_parquet(self.config["input_path"], engine="pyarrow")
        self.data = self.data[
            (self.data[self.target_col] <= 100)
            & (self.data["unique_dri_category_count"] != 0)
        ]

        base_cols = list(self.config.get("cols_to_keep", []))
        flag_src_cols = list(self.config.get("convert_to_flag", []))

        # Safely select existing columns only
        keep = [c for c in (base_cols + flag_src_cols) if c in self.data.columns]
        df_features = self.data.loc[:, keep].copy()
        return df_features

    def create_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create *_flag columns from sources in config["convert_to_flag"].
        Works for names that do or do not end with '_count'.
        """
        print("[Segmentation] converting daypart, weekpart, base categories features to flag")
        flag_cols = [c for c in self.config["convert_to_flag"] if c != self.id_col]  # exclude ID
    
        for col in flag_cols:
            if col not in df.columns:    # optional safety
                continue
            flag_col = col.replace('_count', '_flag')
            df[flag_col] = (df[col] > 0).astype(int)
            df.drop(columns=[col], inplace=True)
    
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capping outliers of the numeric fields to 1st and 99th percentile.
        """
        print("[Segmentation] capping outliers to 1st and 99th percentile")
        numeric_cols_to_cap = [
            col for col in df.select_dtypes(include='number').columns
            if col not in {self.target_col, self.id_col}
        ]
        for col in numeric_cols_to_cap:
            lower = df[col].quantile(self.lower_q)
            upper = df[col].quantile(self.upper_q)
            df[col]=df[col].clip(lower, upper)
        return df

    def rename_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        rename flag columns for better visualization.
        """
        print("[Segmentation] renaming fields for better visualisation")
        df.rename(columns=lambda x: x.replace("omnichannel_", "") if x.startswith("omnichannel_") else x, inplace=True)
        df.rename(columns=self.config["rename_dict"], inplace=True)
        return df

    def _band_filter(self, df: pd.DataFrame, band: str) -> pd.DataFrame:
        """Return the DataFrame filtered to the requested target band."""
        if band == "1_5":
            return df[df[self.target_col] < 6]
        elif band == "6_15":
            return df[(df[self.target_col] > 5) & (df[self.target_col] <= 15)]
        elif band == "15_plus":
            return df[df[self.target_col] > 15]
        else:
            raise ValueError(f"Unknown band: {band}")

    def _rules_for_band(self, band: str):
        """
        Return list of (segment_name, lambda df: boolean mask) for the band.
        Exact rules, expressed once.
        """
        if band == "1_5":
            return [
                ("Seg1_5_1",  lambda d: (d["unique_cat_base_count"] <= 2.5) & (d["Weekend_flag"] == 0)),
                ("Seg1_5_2",  lambda d: (d["unique_cat_base_count"] <= 2.5) & (d["Weekend_flag"] == 1) & (d["Weekday_flag"] == 0)),
                ("Seg1_5_3",  lambda d: (d["unique_cat_base_count"] <= 2.5) & (d["Weekend_flag"] == 1) & (d["Weekday_flag"] == 1)),
                ("Seg1_5_4",  lambda d: (d["unique_cat_base_count"] >  2.5) & (d["Weekday_flag"] == 0) & (d["channel_count"] <= 1.5)),
                ("Seg1_5_5",  lambda d: (d["unique_cat_base_count"] >  2.5) & (d["Weekday_flag"] == 1) & (d["channel_count"] <= 1.5)),
                ("Seg1_5_6",  lambda d: (d["unique_cat_base_count"] >  2.5) & (d["channel_count"] >  1.5) & (d["Premium_price_flag"] == 1)),
                ("Seg1_5_7",  lambda d: (d["unique_cat_base_count"] >  2.5) & 
                 (d["channel_count"] >  1.5) & (d["Premium_price_flag"] == 0) & (d["Weekend_flag"] == 0)),
                ("Seg1_5_8",  lambda d: (d["unique_cat_base_count"] >  2.5) & (d["channel_count"] >  1.5)
                 & (d["Premium_price_flag"] == 0) & (d["Weekend_flag"] == 1)),
            ]

        elif band == "6_15":
            return [
                ("Seg_6_15_01",
                 lambda d: (d["unique_cat_base_count"] <= 6.5) & (d["Weekend_flag"] == 0)),
        
                ("Seg_6_15_02",
                 lambda d: (d["unique_cat_base_count"] <= 6.5) & (d["Lunch_flag"] == 0)
                           & (d["Weekend_flag"] == 1)),
        
                ("Seg_6_15_03",
                 lambda d: (d["unique_cat_base_count"] <= 6.5) & (d["Lunch_flag"] == 1)
                           & (d["Weekend_flag"] == 1)),
        
                ("Seg_6_15_04",
                 lambda d: (d["unique_cat_base_count"] > 6.5) & (d["Lunch_flag"] == 0)
                           & (d["Premium_price_flag"] == 0)),
        
                ("Seg_6_15_05",
                 lambda d: (d["unique_cat_base_count"] > 6.5) & (d["Lunch_flag"] == 1)
                           & (d["Premium_price_flag"] == 0) & (d["Late_Night_flag"] == 0)),
        
                ("Seg_6_15_06",
                 lambda d: (d["unique_cat_base_count"] > 6.5) & (d["Lunch_flag"] == 1)
                           & (d["Premium_price_flag"] == 0) & (d["Late_Night_flag"] == 1)),
        
                ("Seg_6_15_07",
                 lambda d: (d["unique_cat_base_count"] > 6.5) & (d["channel_count"] <= 1.5)
                           & (d["Lunch_flag"] == 0) & (d["Premium_price_flag"] == 1)),
        
                ("Seg_6_15_08",
                 lambda d: (d["unique_cat_base_count"] > 6.5) & (d["channel_count"] <= 1.5)
                           & (d["Lunch_flag"] == 1) & (d["Premium_price_flag"] == 1)),
        
                ("Seg_6_15_09",
                 lambda d: (d["unique_cat_base_count"] > 6.5) & (d["channel_count"] > 1.5)
                           & (d["Late_Night_flag"] == 0) & (d["Premium_price_flag"] == 1)),
        
                ("Seg_6_15_10",
                 lambda d: (d["unique_cat_base_count"] > 6.5) & (d["channel_count"] > 1.5)
                           & (d["Late_Night_flag"] == 1) & (d["Premium_price_flag"] == 1)),
            ]

        elif band == "15_plus":
            return [
                ("Seg15_plus_1", lambda d: (d["unique_cat_base_count"] <= 11.5) & (d["Breakfast_flag"] == 0) & (d["Late_Night_flag"] == 0)),
                ("Seg15_plus_2", lambda d: (d["unique_cat_base_count"] <= 11.5) & (d["Breakfast_flag"] == 0) &
                 (d["Late_Night_flag"] == 1) & (d["Evening_flag"] == 0)),
                ("Seg15_plus_3", lambda d: (d["unique_cat_base_count"] <= 11.5) & (d["unique_cat_base_count"] > 8.5) &
                 (d["Breakfast_flag"] == 0) & (d["Late_Night_flag"] == 1) & (d["Evening_flag"] == 1)),
                ("Seg15_plus_4", lambda d: (d["unique_cat_base_count"] <= 8.5) & (d["Breakfast_flag"] == 0) &
                 (d["Late_Night_flag"] == 1) & (d["Evening_flag"] == 1)),
                ("Seg15_plus_5", lambda d: (d["unique_cat_base_count"] <= 11.5) & (d["Breakfast_flag"] == 1)),
                ("Seg15_plus_6", lambda d: (d["unique_cat_base_count"] >  11.5) & (d["unique_cat_base_count"] <= 16.5) &
                 (d["Breakfast_flag"] == 0) & (d["Late_Night_flag"] == 0)),
                ("Seg15_plus_7", lambda d: (d["unique_cat_base_count"] >  16.5) & (d["Breakfast_flag"] == 0) & (d["Late_Night_flag"] == 0)),
                ("Seg15_plus_8", lambda d: (d["unique_cat_base_count"] >  11.5) & (d["Breakfast_flag"] == 0) & (d["Late_Night_flag"] == 1)),
                ("Seg15_plus_9", lambda d: (d["unique_cat_base_count"] >  11.5) & (d["Breakfast_flag"] == 1)),
            ]
        else:
            raise ValueError(f"Unknown band: {band}")


    def _apply_segments(self, df_band: pd.DataFrame, segments):
        """
        Apply a list of (name, lambda df: mask) to df_band and return:
        - summary_df (Segment, Customer Count, Total Orders, Avg Orders per Customer)
        - assignments_df (algonomy_id, Segment) with unique ids per segment
        """
        results = []
        assignments = []

        for name, rule in segments:
            mask = rule(df_band)
            subset = df_band.loc[mask]

            customer_count = len(subset)
            total_orders = subset[self.target_col].sum()
            aov = (total_orders / customer_count) if customer_count > 0 else np.nan

            results.append({
                "Segment": name,
                "Customer Count": customer_count,
                "Total Orders": total_orders,
                "Avg Orders per Customer": round(aov, 2)
            })

            if self.id_col in subset.columns:
                matched_ids = subset[self.id_col]
                assignments.extend([{"algonomy_id": cid, "Segment": name} for cid in matched_ids])

        summary_df = pd.DataFrame(results).sort_values("Segment")
        assignments_df = pd.DataFrame(assignments).drop_duplicates(subset="algonomy_id")
        return summary_df, assignments_df

    def run_rule_segmentation(self, df_in: pd.DataFrame):
        """
        End-to-end: prepare aliases, make flags (if not present), then run segments for 1_5, 6_15, 15_plus.
        Returns a dict with summaries and assignments per band.
        """
        df = df_in.copy()
        # df = self._ensure_aliases(df)

        out = {}

        for band in ["1_5", "6_15", "15_plus"]:
            df_band = self._band_filter(df, band)
            segs = self._rules_for_band(band)
            summary_df, assignments_df = self._apply_segments(df_band, segs)
            out[band] = {
                "summary": summary_df,
                "assignments": assignments_df
            }
        return out

    def save_segments(self, df):
        """Save segments results in parquet format to S3"""
        df.to_parquet(self.config["output_segments_path"], index=False)

    def _load_pickle_boto(self, s3_path: str):
        """
        Load a pickle object from S3 using boto3.
        """
        u = urlparse(s3_path)
        if u.scheme != "s3":
            raise ValueError("Path must start with s3://")
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))
        body = obj["Body"].read()
        return pickle.loads(body)

    def _prep_micro_base(self, df_base: pd.DataFrame, df_segments: pd.DataFrame, source_segments: list) -> pd.DataFrame:
        """
        Filter customers in any of source_segments, merge back to base, and apply renames once.
        """
        sub = df_segments.loc[df_segments["Segment"].isin(source_segments), ["algonomy_id"]].drop_duplicates()            
        df = sub.merge(df_base, on="algonomy_id", how="left").copy()

        if any(seg in {"Seg1_5_6", "Seg_6_15_10", "Seg15_plus_9"} for seg in source_segments):
            df=self._cap_outliers(df)
    
        # standardize column names for micro-seg models
        df.rename(columns=lambda x: x.replace("omnichannel_", "") if x.startswith("omnichannel_") else x, inplace=True)
        if "rename_dict_microsegments" in self.config:
            df.rename(columns=self.config["rename_dict_microsegments"], inplace=True)
        return df
    
    def _predict_from_model_key(self, df: pd.DataFrame, model_key: str) -> pd.DataFrame:
        """
        Load (kmeans, scaler, features) from S3 via config key, predict, and return algonomy_id + microcluster.
        """
        payload  = self._load_pickle_boto(self.config[model_key])
        kmeans   = payload["kmeans"]
        scaler   = payload["scaler"]
        features = payload["features"]
    
        # ensure required columns & order
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required features for {model_key}: {missing[:8]}{' ...' if len(missing)>8 else ''}")
    
        X = df.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(0)
        preds = kmeans.predict(scaler.transform(X))
    
        out = pd.DataFrame({
            "algonomy_id": df["algonomy_id"].values,
            "microcluster": preds.astype(str)
        })
        return out
    
    def microsegment_one(self, df_base: pd.DataFrame, df_segments: pd.DataFrame, name: str,
                         source_segments: list, model_key: str) -> pd.DataFrame:
        """
        One micro-seg pipeline: subset → rename → predict → label with group name.
        """        
        dfp = self._prep_micro_base(df_base, df_segments, source_segments)
        res = self._predict_from_model_key(dfp, model_key)
        res["Segment"] = name
        # quick read:
        vc = res["microcluster"].value_counts()
        print(f"[Microsegments] {name} size by microcluster: {vc.to_dict()}")
        return res
    
    def microsegments(self, df_base: pd.DataFrame, df_segments: pd.DataFrame, plan: list) -> pd.DataFrame:
        """
        Batch runner. `plan` is a list of dicts:
          {"name": "...", "source_segments": ["Seg1_5_1", "Seg1_5_2"], "model_key": "segment_1_5_1_2_model"}
        Returns a single concatenated DataFrame of all micro-segment assignments.
        """
        frames = []
        for item in plan:
            frames.append(
                self.microsegment_one(
                    df_base=df_base,
                    df_segments=df_segments,
                    name=item["name"],
                    source_segments=item["source_segments"],
                    model_key=item["model_key"],
                )
            )
        return pd.concat(frames, ignore_index=True)

    def _validate_micro_plan(self):
        """
        Check : if all the required fields like name, source_segments, model_key is defined in the config.
        """
        plan = self.config.get("micro_plan", [])
        required = {"name", "source_segments", "model_key"}
        for i, item in enumerate(plan):
            missing = required - set(item)
            if missing:
                raise ValueError(f"micro_plan[{i}] missing keys: {missing}")
            if item["model_key"] not in self.config:
                raise KeyError(f"Model key '{item['model_key']}' not found in config.")
            if not isinstance(item["source_segments"], list) or not item["source_segments"]:
                raise ValueError(f"micro_plan[{i}].source_segments must be a non-empty list")


    def run_microsegments_from_config(self, df_base, df_segments):
        """ Validate and get the keys of the microplan"""
        self._validate_micro_plan()
        plan = [p for p in self.config.get("micro_plan", []) if p.get("enabled", True)]
        return self.microsegments(df_base, df_segments, plan)
        


    def run_rule_pipeline(self):
        """
        Execute the full rule-based segmentation pipeline.
          1. Load raw feature data from the configured Parquet source.
          2. Generate flag variables for selected columns (binary features).
          3. Apply renaming rules to align with expected feature names.
          4. Run the rule-based segmentation logic across predefined order bands
             (`1_5`, `6_15`, `15_plus`).
          5. Extract and deduplicate segment assignments for each band.
          6. Combine all band-level assignments into a single DataFrame.
        """
        original_df = self.load_data()
        df = original_df.copy()
        df = self.create_flags(df)
        df = self.rename_cols(df)
        results = self.run_rule_segmentation(df)
    
        assign_1_5   = results["1_5"]["assignments"].drop_duplicates("algonomy_id")
        assign_6_15  = results["6_15"]["assignments"].drop_duplicates("algonomy_id")
        assign_15p   = results["15_plus"]["assignments"].drop_duplicates("algonomy_id")
    
        combined = pd.concat([assign_1_5, assign_6_15, assign_15p], ignore_index=True)
        return original_df, combined


