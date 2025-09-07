import pandas as pd
import numpy as np
import ast
import os
import json
import logging
from datetime import datetime
from itertools import combinations

logger = logging.getLogger(__name__)

class CustomerFeaturePipeline:
    """
    Pipeline to engineer features at the customer (algonomy_id) level using order features from all channels.
    """

    def __init__(self, config_path:str):
        """
        Initializes the pipeline, loads configuration file.
        
        Args:
            config_path (str): Config path passed as arg.
        """

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        config_path = os.path.join(root_dir, config_path)
        # Load the configuration file
        with open(config_path, "r") as f:
            config_all= json.load(f)

        self.config = dict(config_all["customer"])
        self.config['ownapp_orders'] = config_all['ownapp']['output_path']
        self.config['swiggy_orders'] = config_all['swiggy']['output_path']
        self.config['instore_orders'] = config_all['instore']['orders_output_path']
        self.cutoff_date = self.config.get('cutoff_date', "2024-07-01")


    def run(self):
        """Run the complete pipeline and write customer features parquet."""
        logger.info("Starting customer feature pipeline...")

        self._read_orders()
        self._filter_to_ownapp_users()
        self._add_price_without_taxes()
        self._ensure_category_lists()
        self._standardize_timestamps()
        self._filter_last_12_months()
        self._concat_all_channels()
        self._build_discount_features()
        self._build_category_metrics()
        self._build_dri_category_metrics()
        self._build_order_metrics()
        self._build_daypart_metrics()
        self._build_daypart_metrics_ownapp()
        self._build_price_bucket_metrics()
        self._build_price_bucket_metrics_ownapp()
        self._build_category_order_counts()
        self._build_ownapp_delivery_time()
        self._finalize_and_write()

        logger.info("Customer feature pipeline completed.")
        return self.config['output_path']

    def _read_orders(self):
        """Read all three order files from config."""
        logger.info("Reading orders files for ownapp, swiggy, and instore...")
        self.ownapp_orders = pd.read_parquet(self.config['ownapp_orders'])
        self.swiggy_orders = pd.read_parquet(self.config['swiggy_orders'])
        self.instore_orders = pd.read_parquet(self.config['instore_orders'])

    def _filter_to_ownapp_users(self):
        """Filter Swiggy/Instore to only ownapp users."""
        logger.info("Filtering Swiggy/Instore to only users present in OwnApp...")
        ownapp_users = set(self.ownapp_orders['algonomy_id'])
        # Only keep orders for users who appear in ownapp_orders
        self.swiggy_orders = self.swiggy_orders[self.swiggy_orders['algonomy_id'].isin(ownapp_users)].copy()
        self.instore_orders = self.instore_orders[self.instore_orders['algonomy_id'].isin(ownapp_users)].copy()

    def _add_price_without_taxes(self):
        """Add price_without_taxes column to all three DataFrames."""
        logger.info("Calculating price_without_taxes for all channels...")
        # OwnApp/Swiggy: price_without_taxes = discounted_price + sub_total_price
        self.ownapp_orders['price_without_taxes'] = self.ownapp_orders['discounted_price'] + self.ownapp_orders['sub_total_price']
        self.swiggy_orders['price_without_taxes'] = self.swiggy_orders['discounted_price'] + self.swiggy_orders['sub_total_price']
        # InStore: BaseAmount (pre-tax) + discount
        self.instore_orders['discounted_price'] = self.instore_orders['discounted_price'].fillna(0)
        self.instore_orders['price_without_taxes'] = self.instore_orders['BaseAmount'].fillna(0) + self.instore_orders['discounted_price'].fillna(0)

    def _ensure_category_lists(self):
        """Ensure all list-type category columns are truly lists, not strings."""
        logger.info("Ensuring unique_dri_categories_in_order and unique_base_products_in_order are real lists...")
        for df in [self.ownapp_orders, self.swiggy_orders, self.instore_orders]:
            df['unique_dri_categories_in_order'] = df['unique_dri_categories_in_order'].apply(self._ensure_list)
            df['unique_base_products_in_order'] = df['unique_base_products_in_order'].apply(self._ensure_list)

    def _ensure_list(self, x):
        """Ensures input is a list, else returns an empty list."""
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return []

    def _standardize_timestamps(self):
        """Convert order_placed_datetime columns to pandas.Timestamp (no tz) for all channels."""
        logger.info("Standardizing timestamp columns for cross-channel consistency...")
        self.ownapp_orders['order_placed_datetime'] = pd.to_datetime(self.ownapp_orders['order_placed_datetime'], format='mixed').dt.tz_localize(None)
        self.swiggy_orders['order_placed_datetime'] = pd.to_datetime(self.swiggy_orders['order_placed_datetime'], format='mixed').dt.tz_localize(None)
        self.instore_orders['order_placed_datetime'] = pd.to_datetime(self.instore_orders['order_placed_datetime']).dt.tz_localize(None)

    def _filter_last_12_months(self):
        """Keep only orders in the last 12 months for all channels."""
        logger.info(f"Filtering to cutoff date: {self.cutoff_date}")
        threshold_date = pd.to_datetime(self.cutoff_date).tz_localize(None)
        self.ownapp_orders = self.ownapp_orders[self.ownapp_orders['order_placed_datetime'] >= threshold_date]
        self.swiggy_orders = self.swiggy_orders[self.swiggy_orders['order_placed_datetime'] >= threshold_date]
        self.instore_orders = self.instore_orders[self.instore_orders['order_placed_datetime'] >= threshold_date]
        logger.info(f"Ownapp min/max order date: {self.ownapp_orders['order_placed_datetime'].min()} - {self.ownapp_orders['order_placed_datetime'].max()}")
        logger.info(f"Swiggy min/max order date: {self.swiggy_orders['order_placed_datetime'].min()} - {self.swiggy_orders['order_placed_datetime'].max()}")
        logger.info(f"Instore min/max order date: {self.instore_orders['order_placed_datetime'].min()} - {self.instore_orders['order_placed_datetime'].max()}")

    def _concat_all_channels(self):
        """Concatenate all channels' orders, add channel label, and keep unified columns."""
        logger.info("Concatenating all channel datasets and unifying column set...")
        c = self.config
        cols = c['cols']
        dri_categ_names = c['dri_categ_names']
        swiggy_base_categs = c['swiggy_base_categs']
        ownapp_base_categs = c['ownapp_base_categs']
        instore_base_categs = c['instore_base_categs']

        for df in [self.ownapp_orders, self.swiggy_orders, self.instore_orders]:
            df['algonomy_id'] = df['algonomy_id'].astype('string')  # Uniform type

        # Standardize columns before concat
        self.ownapp_orders = self.ownapp_orders[cols + dri_categ_names + ownapp_base_categs + ['order_delivery_datetime']]
        self.swiggy_orders = self.swiggy_orders[cols + dri_categ_names + swiggy_base_categs]
        self.instore_orders = self.instore_orders[cols + dri_categ_names + instore_base_categs]

        self.ownapp_orders['channel'] = 'ownapp'
        self.swiggy_orders['channel'] = 'swiggy'
        self.instore_orders['channel'] = 'instore'

        self.combined_df = pd.concat(
            [self.ownapp_orders, self.swiggy_orders, self.instore_orders],
            ignore_index=True,
            sort=True
        )

    def _build_discount_features(self):
        """Calculate discount percentage and user-level mean discount."""
        logger.info("Calculating discount percent and aggregating by user...")
        combined_df = self.combined_df
        # If price_without_taxes is 0, result will be inf or NaN
        combined_df['discount_pct'] = (
            combined_df['discounted_price'] / combined_df['price_without_taxes']
        ).replace([np.inf, -np.inf], np.nan)
        user_avg_discount = combined_df.groupby('algonomy_id')['discount_pct'].mean().reset_index()
        user_avg_discount.rename(columns={'discount_pct': 'omnichannel_avg_discount_pct'}, inplace=True)
        user_avg_discount.loc[user_avg_discount['omnichannel_avg_discount_pct'] < 0, 'omnichannel_avg_discount_pct'] = 0
        self.users_feature_df = self.ownapp_orders[["algonomy_id"]].drop_duplicates().reset_index(drop=True)
        self.users_feature_df = self.users_feature_df.merge(user_avg_discount, on='algonomy_id', how='left')

    def _flatten_unique_no_nan(self, lists):
        """Flattens a series of lists and removes any 'nan' elements."""
        return list({
            c
            for sub in lists
            if isinstance(sub, (list, tuple))
            for c in sub
            if c != 'nan'
        })

    def _build_category_metrics(self):
        """Aggregate unique base category products per user."""
        logger.info("Calculating unique base category metrics...")
        def get_user_categs(df, col, count_col):
            out = (
                df.groupby('algonomy_id')[col]
                .agg(self._flatten_unique_no_nan)
                .reset_index()
            )
            out[count_col] = out[col].apply(len)
            return out
        category_metrics = get_user_categs(self.combined_df, 'unique_base_products_in_order', 'unique_category_base_products_count')
        self.users_feature_df = self.users_feature_df.merge(category_metrics, on='algonomy_id', how='left')

    def _build_dri_category_metrics(self):
        """Aggregate unique DRI categories per user."""
        logger.info("Calculating unique DRI category metrics...")
        def get_user_categs(df, col, count_col):
            out = (
                df.groupby('algonomy_id')[col]
                .agg(self._flatten_unique_no_nan)
                .reset_index()
            )
            out[count_col] = out[col].apply(len)
            return out
        dri_category_metrics = get_user_categs(self.combined_df, 'unique_dri_categories_in_order', 'unique_dri_category_count')
        self.users_feature_df = self.users_feature_df.merge(dri_category_metrics, on='algonomy_id', how='left')

    def _build_order_metrics(self):
        """Build user-level order metrics for all channels."""
        logger.info("Building order count, frequency, recency, and channel usage features...")
        combined_df = self.combined_df
        dummies = pd.get_dummies(combined_df['channel'])
        # Combine with core columns to build up the user-level aggregation
        df2 = pd.concat([
            combined_df[['algonomy_id', 'order_id', 'order_placed_datetime', 'total_price', 'channel', 'is_weekend', 'discounted_price']],
            dummies
        ], axis=1)
        # Main groupby aggregation: counts, sums, first/last order time, AOV, etc.
        agg = df2.groupby('algonomy_id').agg(
            omnichannel_order_count=('is_weekend', 'count'),
            swiggy_order_count=('swiggy', 'sum'),
            instore_order_count=('instore', 'sum'),
            ownapp_order_count=('ownapp', 'sum'),
            omnichannel_weekend_order_count=('is_weekend', 'sum'),
            omnichannel_aov=('total_price', 'mean'),
            omnichannel_avg_discount=('discounted_price', 'mean'),
            omnichannel_first_order_at=('order_placed_datetime', 'min'),
            omnichannel_last_order_at=('order_placed_datetime', 'max'),
        )
        agg['omnichannel_weekday_order_count'] = (
            agg['omnichannel_order_count'] - agg['omnichannel_weekend_order_count']
        )
        # OwnApp-specific order timing for recency/frequency
        ownapp_orders2 = df2[df2['channel'] == 'ownapp']
        ownapp_times = ownapp_orders2.groupby('algonomy_id')['order_placed_datetime'].agg(
            ownapp_first_order_at='min',
            ownapp_last_order_at='max'
        )
        # Join back
        agg = agg.merge(ownapp_times, on='algonomy_id', how='left')
        # Get channel label: "OC only", "OC + Swiggy", etc.
        channel_sets = combined_df.groupby('algonomy_id')['channel'].agg(set)
        _label_map = {'ownapp': 'OC', 'instore': 'Dine in', 'swiggy': 'Swiggy'}
        def _make_label(ch_set):
            labs = [_label_map[c] for c in ['ownapp','instore','swiggy'] if c in ch_set]
            return 'OC only' if labs == ['OC'] else ' + '.join(labs)
        agg['channels_ordered_from'] = channel_sets.map(_make_label)
        order_metrics = agg.reset_index()

        # Compute recency/tenure/avg days for omnichannel and ownapp orders
        reference_date = pd.to_datetime("2025-06-30")
        omni_dates = combined_df.groupby('algonomy_id')['order_placed_datetime'].agg(
            omnichannel_first_order_at='min',
            omnichannel_last_order_at='max',
            omnichannel_order_count='count'
        )
        omni_dates['omnichannel_recency'] = (reference_date - omni_dates['omnichannel_last_order_at']).dt.days
        omni_dates['omnichannel_tenure_days'] = (omni_dates['omnichannel_last_order_at'] - omni_dates['omnichannel_first_order_at']).dt.days + 1
        omni_dates['omnichannel_avg_days_between_orders'] = omni_dates['omnichannel_tenure_days'] / (omni_dates['omnichannel_order_count'] - 1)
        omni_dates['omnichannel_avg_days_between_orders'] = omni_dates['omnichannel_avg_days_between_orders'].replace([np.inf, -np.inf], np.nan)
        ownapp_df = combined_df[combined_df['channel'] == 'ownapp']
        ownapp_dates = ownapp_df.groupby('algonomy_id')['order_placed_datetime'].agg(
            ownapp_first_order_at='min',
            ownapp_last_order_at='max',
            ownapp_order_count='count'
        )
        ownapp_dates['ownapp_recency'] = (reference_date - ownapp_dates['ownapp_last_order_at']).dt.days
        ownapp_dates['ownapp_tenure_days'] = (ownapp_dates['ownapp_last_order_at'] - ownapp_dates['ownapp_first_order_at']).dt.days + 1
        ownapp_dates['ownapp_avg_days_between_orders'] = ownapp_dates['ownapp_tenure_days'] / (ownapp_dates['ownapp_order_count'] - 1)
        ownapp_dates['ownapp_avg_days_between_orders'] = ownapp_dates['ownapp_avg_days_between_orders'].replace([np.inf, -np.inf], np.nan)
        # Merge calculated fields into main table
        order_metrics = order_metrics.merge(omni_dates[['omnichannel_recency', 'omnichannel_tenure_days', 'omnichannel_avg_days_between_orders']], on='algonomy_id', how='left')
        order_metrics = order_metrics.merge(ownapp_dates[['ownapp_recency', 'ownapp_tenure_days', 'ownapp_avg_days_between_orders']], on='algonomy_id', how='left')
        self.users_feature_df = self.users_feature_df.merge(order_metrics, on='algonomy_id', how='left')

    def _build_daypart_metrics(self):
        """Build omnichannel daypart order count features (Breakfast, Lunch, etc, and combos)."""
        logger.info("Calculating omnichannel daypart metrics...")
        self.users_feature_df = self.users_feature_df.merge(
            self._daypart_metrics(self.combined_df, 'omnichannel'),
            on='algonomy_id', how='left'
        )

    def _build_daypart_metrics_ownapp(self):
        """Build ownapp daypart order count features."""
        logger.info("Calculating ownapp daypart metrics...")
        ownapp_df = self.combined_df[self.combined_df['channel'] == 'ownapp']
        self.users_feature_df = self.users_feature_df.merge(
            self._daypart_metrics(ownapp_df, 'ownapp'),
            on='algonomy_id', how='left'
        )

    def _daypart_metrics(self, df, channel_prefix="omnichannel"):
        """
        Helper for calculating daypart metrics. Includes combo columns (e.g. Breakfast+Lunch).
        """
        df = df.copy()
        dayparts = ['Breakfast', 'Lunch', 'Evening', 'Dinner', 'Late Night']
        dummies = pd.get_dummies(df['order_time_of_day'])[dayparts]
        df = pd.concat([df, dummies], axis=1)
        agg_dict = {dp: 'sum' for dp in dayparts}
        metrics = df.groupby('algonomy_id').agg(agg_dict)
        # Rename for clarity: omnichannel_Breakfast_order_count, etc.
        metrics = metrics.rename(columns={dp: f'{channel_prefix}_{dp}_order_count' for dp in dayparts})
        # Optionally, add all combos of size 2..5
        for r in range(2, 6):
            for combo in combinations(dayparts, r):
                cols = [f'{channel_prefix}_{dp}_order_count' for dp in combo]
                out_col = f'{channel_prefix}_{"_".join(combo)}_order_count'
                metrics[out_col] = metrics[cols].sum(axis=1)
        return metrics.reset_index()

    def _build_price_bucket_metrics(self):
        """Build omnichannel price bucket metrics (e.g. Add_ons, Economy, ...)."""
        logger.info("Calculating omnichannel price bucket features...")
        price_bucket_columns = self.config['price_bucket_columns']
        self.users_feature_df = self.users_feature_df.merge(
            self._compute_price_bucket_ratios(self.combined_df, "omnichannel", price_bucket_columns),
            on="algonomy_id", how="left"
        )

    def _build_price_bucket_metrics_ownapp(self):
        """Build ownapp price bucket metrics."""
        logger.info("Calculating ownapp price bucket features...")
        price_bucket_columns = self.config['price_bucket_columns']
        ownapp_df = self.combined_df[self.combined_df['channel'] == 'ownapp']
        self.users_feature_df = self.users_feature_df.merge(
            self._compute_price_bucket_ratios(ownapp_df, "ownapp", price_bucket_columns),
            on="algonomy_id", how="left"
        )

    def _compute_price_bucket_ratios(self, df, prefix, price_bucket_columns):
        """Helper for price bucket order count features."""
        grouped = df.groupby('algonomy_id')[price_bucket_columns].sum().astype(int).reset_index()
        # For each price bucket, just use order count as is (these columns are item-count, often 0/1)
        for col in price_bucket_columns:
            grouped[f"{prefix}_order_count_with_{col.replace('_items_count', '_price_range_items')}"] = grouped[col]
        req_cols = ['algonomy_id'] + [f"{prefix}_order_count_with_{col.replace('_items_count', '_price_range_items')}" for col in price_bucket_columns]
        return grouped[req_cols]

    def _build_category_order_counts(self):
        """Feature: count of orders by base product and DRI category."""
        logger.info("Calculating category order count metrics (omnichannel, ownapp)...")
        combined_df = self.combined_df
        # Find all category columns
        category_cols = [col for col in combined_df.columns if col.startswith('categ_base_product_')]
        dri_category_cols = [col for col in combined_df.columns if col.startswith('dri_categ_')]
        # Omnichannel totals
        omni_category_agg = combined_df.groupby('algonomy_id')[category_cols + dri_category_cols].sum().astype(int)
        omni_category_agg.rename(columns=lambda x: f"omnichannel_{x}_count", inplace=True)
        # Ownapp-only totals
        ownapp_df2 = combined_df[combined_df['channel'] == 'ownapp']
        ownapp_category_agg = ownapp_df2.groupby('algonomy_id')[category_cols + dri_category_cols].sum().astype(int)
        ownapp_category_agg.rename(columns=lambda x: f"ownapp_{x}_count", inplace=True)
        # Merge to user features
        self.users_feature_df = self.users_feature_df.merge(omni_category_agg, on='algonomy_id', how='left')
        self.users_feature_df = self.users_feature_df.merge(ownapp_category_agg, on='algonomy_id', how='left')

    def _build_ownapp_delivery_time(self):
        """Feature: average delivery time for ownapp users."""
        logger.info("Calculating ownapp average delivery time (in minutes)...")
        ownapp_df = self.combined_df[self.combined_df['channel'] == 'ownapp'].copy()
        # Adjust for timezone if needed (add 5h30m to delivery time)
        ownapp_df['order_delivery_datetime_adjusted'] = (
            ownapp_df['order_delivery_datetime'] + pd.Timedelta(hours=5, minutes=30)
        )
        # Delivery time = delivered - placed
        ownapp_df['delivery_time_minutes'] = (
            (ownapp_df['order_delivery_datetime_adjusted'] - ownapp_df['order_placed_datetime'])
            .dt.total_seconds() / 60
        )
        ownapp_delivery_metrics = (
            ownapp_df.groupby('algonomy_id')['delivery_time_minutes']
            .mean()
            .reset_index()
            .rename(columns={'delivery_time_minutes': 'ownapp_avg_delivery_time_minutes'})
        )
        self.users_feature_df = self.users_feature_df.merge(ownapp_delivery_metrics, on='algonomy_id', how='left')

    def _finalize_and_write(self):
        """Finalize and write customer features to output Parquet file."""
        logger.info("Finalizing and writing features to Parquet...")
        users_feature_df = self.users_feature_df
        # Standardize datatype for algonomy_id for downstream ML
        users_feature_df['algonomy_id'] = users_feature_df['algonomy_id'].astype('string')
        # Weekend order count is just total minus weekday
        users_feature_df['omnichannel_weekend_order_count'] = users_feature_df['omnichannel_order_count'] - users_feature_df['omnichannel_weekday_order_count']
        # Channel count = how many channels used (binary flags summed)
        users_feature_df["channel_count"] = (
            (users_feature_df["ownapp_order_count"] > 0).astype(int)
            + (users_feature_df["swiggy_order_count"] > 0).astype(int)
            + (users_feature_df["instore_order_count"] > 0).astype(int)
        )
        users_feature_df.to_parquet(self.config['output_path'], index=False)
        logger.info(f"Customer features written to {self.config['output_path']}.")
