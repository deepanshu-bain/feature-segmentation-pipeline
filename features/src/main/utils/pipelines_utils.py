import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

def read_orders(config, storage_options=None):
    """Read orders data from Parquet with selected columns.

    Args:
        config (dict): Configuration with 'orders_data_path' and 'orders_columns'.
        storage_options (dict, optional): Storage options for remote filesystems.

    Returns:
        pd.DataFrame: Orders dataframe.
    """
    logger.info(f"Reading orders from {config['orders_data_path']} ...")
    df = pd.read_parquet(
        config["orders_data_path"],
        columns=config["orders_columns"],
        storage_options=storage_options or {}
    )
    logger.info(f"Orders loaded: {len(df)} rows.")
    return df

def localize_and_filter_dates(df, config, common, datetime_cols):
    """Localize datetime columns to specified timezone and filter by date range.

    Args:
        df (pd.DataFrame): Input orders dataframe.
        config (dict): Pipeline config.
        common (dict): Common config with time info.
        datetime_cols (list of str): List of datetime columns to localize.

    Returns:
        pd.DataFrame: Filtered dataframe within date window.
    """
    run_date = pd.to_datetime(common["current_run_date"]).tz_localize(common["timezone"])
    start_date = run_date - pd.DateOffset(months=common["months_offset"])
    logger.info(f"Filtering orders between {start_date} and {run_date} in {common['timezone']}.")
    for col in datetime_cols:
        logger.info(f"Localizing datetime column: {col}")
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_convert(common["timezone"])
    mask = (df[datetime_cols[0]] >= start_date) & (df[datetime_cols[0]] < run_date)
    filtered = df.loc[mask].copy()
    logger.info(f"Orders after date filtering: {len(filtered)} rows.")
    return filtered

def map_to_centralized_customer_id(df, common, storage_options=None):
    """Map orders to centralized customer (algonomy_id) using a mapping file.

    Args:
        df (pd.DataFrame): Orders dataframe.
        common (dict): Common config with mapping file info.
        storage_options (dict, optional): Storage options for remote filesystems.

    Returns:
        pd.DataFrame: Orders with algonomy_id column added.
    """
    logger.info("Mapping to centralized customer ID (algonomy_id)...")
    mapping = pd.read_csv(common["mapping_file"], dtype={'customer_code_ty': str}, storage_options=storage_options or {})
    mapping = mapping.rename(columns={"invoice_no": "map_invoice_no", "customer_code_ty": "algonomy_id"})
    merged = df.merge(mapping[["map_invoice_no", "algonomy_id"]], left_on="id", right_on="map_invoice_no", how="left")
    logger.info(f"Customer ID mapping done. Null algonomy_id: {merged['algonomy_id'].isna().sum()}")
    #Filter out orders/inovices with no customer mapping
    merged = merged[merged['algonomy_id'].notnull()].copy()
    logger.info(f"Orders after filtering Null algonomy_id: {len(merged)} rows.")
    return merged

def filter_non_oc_users_order(df, all_config):
    """
    Filters the given DataFrame to include only orders from users
    present in the 'algonomy_id' column of the OwnApp orders output.

    Args:
        df (pd.DataFrame): The input DataFrame to filter.
        all_config (dict): The full configuration dictionary. 

    Returns:
        pd.DataFrame: Filtered DataFrame containing only orders from users
            who are present in the OwnApp orders output.
    """
    ownapp_df = pd.read_parquet(all_config['ownapp']['output_path'])
    df_orders = df[df['algonomy_id'].isin(ownapp_df['algonomy_id'])].copy()
    logger.info(f"Orders after filtering non OC algonomy_id: {len(df_orders)} rows.")
    return df_orders

def map_store_details(df, common, storage_options=None):
    """Map orders to enriched store details.

    Args:
        df (pd.DataFrame): Orders dataframe.
        common (dict): Common config with store mapping files.
        storage_options (dict, optional): Storage options for remote filesystems.

    Returns:
        pd.DataFrame: Orders with store details columns.
    """
    logger.info("Mapping store details...")
    df_mds_dsr = pd.read_csv(common["store_mds_dsr_path"], storage_options=storage_options or {})
    df_master = pd.read_csv(common["store_master_path"], storage_options=storage_options or {})
    df_mds_dsr["DSR Code"] = df_mds_dsr["DSR Code"].astype(str).str.strip()
    df_master["Store Code"] = df_master["Store Code"].astype(str).str.strip()
    df_store = df_mds_dsr[["MDS ID", "DSR Code"]].merge(
        df_master, left_on="DSR Code", right_on="Store Code", how="left"
    )
    cols = ["MDS ID", "DSR Code", "Store Name", "Store Type", "Geographical City", "Key City", "Generator New"]
    df_store = df_store[cols]
    merged = df.merge(df_store, left_on="store_id_id", right_on="MDS ID", how="left")
    logger.info(f"Store details mapping done. Null Store Name: {merged['Store Name'].isna().sum()}")
    return merged

def map_order_items_to_categories(df, config, common, storage_options=None):
    """Load order items and map each item to its category details.

    Args:
        df (pd.DataFrame): Orders dataframe (not used here, but could pass for logging).
        config (dict): Pipeline config with item paths/columns.
        common (dict): Common config with categories path.
        storage_options (dict, optional): Storage options for remote filesystems.

    Returns:
        pd.DataFrame: Order items merged with categories.
    """
    logger.info("Loading and mapping order items to categories...")
    df_items = pd.read_parquet(
        config["order_items_data_path"],
        columns=config["order_items_columns"],
        storage_options=storage_options or {}
    )
    df_items['item_id_id'] = df_items['item_id_id'].astype('string')
    df_cat = pd.read_csv(common["categories_path"], dtype={'item_id_id': str}, storage_options=storage_options or {})
    df_cat['item_id_id'] = df_cat['item_id_id'].astype('string')
    df_cat = df_cat[["item_id_id", "ProductCode", "Clubbed base product category", "Final Dri Category"]]
    merged = df_items.merge(df_cat, on="item_id_id", how="left")
    logger.info(f"Order items mapped to categories: {len(merged)} rows.")
    return merged

def remove_choice_and_promo_items(df, common, storage_options=None):
    """Remove choice items and promo items from the items dataframe using clear_master mapping.

    Args:
        df (pd.DataFrame): Items dataframe.
        common (dict): Common config with clear master path.
        storage_options (dict, optional): Storage options for remote filesystems.

    Returns:
        pd.DataFrame: Items with choice/promo items removed.
    """
    logger.info("Removing choice and promo items...")
    clear_master = pd.read_csv(common["clear_master_path"], storage_options=storage_options or {})
    clear_master['header_item_id_id'] = clear_master['header_item_id_id'].astype('str')
    df['item_id_id'] = df['item_id_id'].astype('str')
    header_orders = df[df['promo_item'] == False].merge(
        clear_master[['header_item_id_id']].drop_duplicates(),
        left_on='item_id_id', right_on='header_item_id_id'
    )[['sales_id', 'item_id_id']].rename(columns={'item_id_id': 'header_item_id_id'})
    order_header_choices = header_orders.merge(
        clear_master[['header_item_id_id', 'choice_item_id_id', 'Qty']],
        on='header_item_id_id',
        how='left'
    )
    choice_counts = df[df['promo_item'] == False].groupby(
        ['sales_id', 'item_id_id']
    ).agg(
        choice_count=('item_id_id', 'count'),
        order_quantity=('quantity', 'sum')
    ).reset_index().rename(columns={'item_id_id': 'choice_item_id_id'})
    merged = order_header_choices.merge(
        choice_counts,
        on=['sales_id', 'choice_item_id_id'],
        how='left'
    )
    to_remove = merged[
        (merged['choice_count'] > 0) &
        (merged['Qty'] >= merged['order_quantity']) &
        (~merged['order_quantity'].isna())
    ][['sales_id', 'choice_item_id_id']].drop_duplicates()
    mask_remove = df.set_index(['sales_id', 'item_id_id']).index.isin(
        to_remove.set_index(['sales_id', 'choice_item_id_id']).index
    )
    df_no_choice = df[~mask_remove]
    logger.info(f"Choice and promo item removal done. Remaining items: {len(df_no_choice)}")
    return df_no_choice[df_no_choice['promo_item'] == False].copy()

def remove_unmapped_or_excluded_items(df, exclude_label="Exclude"):
    """Remove items with no mapped category or marked as Exclude.

    Args:
        df (pd.DataFrame): Items dataframe.
        exclude_label (str, optional): Value indicating excluded items.

    Returns:
        pd.DataFrame: Filtered items dataframe.
    """
    logger.info("Removing unmapped and excluded items...")
    before = len(df)
    df = df[~df["Final Dri Category"].isna()]
    df['Clubbed base product category'] = df['Clubbed base product category'].astype(str).str.strip()
    df = df[df['Clubbed base product category'] != exclude_label].copy()
    after = len(df)
    logger.info(f"Removed {before - after} items without mapping or marked as '{exclude_label}'.")
    return df

def flag_orders_with_queries(df, config, storage_options=None):
    """Flag orders that have customer queries or complaints.

    Args:
        df (pd.DataFrame): Orders dataframe.
        config (dict): Pipeline config with query path/flag.
        storage_options (dict, optional): Storage options for remote filesystems.

    Returns:
        pd.DataFrame: Orders with boolean has_query column.
    """
    if not config.get("has_query", False) or not config.get("orderqueries_path"):
        logger.info("No queries mapping needed for this pipeline.")
        df["has_query"] = False
        return df
    logger.info("Flagging orders with customer queries...")
    orderqueries = pd.read_parquet(config["orderqueries_path"], storage_options=storage_options or {})
    query_order_ids = orderqueries.order_id.dropna().astype('Int64').unique()
    df['has_query'] = df['id'].isin(query_order_ids)
    logger.info(f"Orders flagged with queries: {df['has_query'].sum()}")
    return df

def create_base_order_features(df, feature_map):
    """Create a DataFrame of base features using a mapping.

    Args:
        df (pd.DataFrame): Source dataframe.
        feature_map (dict): Mapping {output_column: input_column}.

    Returns:
        pd.DataFrame: DataFrame with only the base features.
    """
    logger.info("Creating base order features DataFrame...")
    features_df = pd.DataFrame({k: df[v] for k, v in feature_map.items() if v in df.columns})
    logger.info(f"Base features DataFrame shape: {features_df.shape}")
    return features_df

def add_price_bucket_features(orders_feature_df, order_items_df, price_col_name, order_col_name, common):
    """Add price bucket features per order.

    Args:
        orders_feature_df (pd.DataFrame): Base order features dataframe.
        order_items_df (pd.DataFrame): Cleaned order items dataframe.
        common (dict): Common config with price_bins and bucket_labels.

    Returns:
        pd.DataFrame: Feature dataframe with price bucket columns added.
    """
    logger.info("Creating price bucket features...")
    price_bins = common["price_config"]["price_bins"]
    bucket_labels = common["price_config"]["bucket_labels"]
    df = order_items_df.copy()
    df['price_bucket'] = pd.cut(
        df[price_col_name],
        bins=price_bins,
        labels=bucket_labels,
        right=False
    )
    bucket_quantity = (
        df.groupby([order_col_name, 'price_bucket'])['quantity']
        .count()
        .clip(upper=1)
        .unstack(fill_value=0)
        .astype(int)
        .reset_index()
    )
    bucket_quantity.columns = ['order_id' if col == order_col_name else f'{col}_items_count' for col in bucket_quantity.columns]
    logger.info(f"Price bucket features created: {bucket_quantity.shape[1]-1} buckets.")
    return orders_feature_df.merge(bucket_quantity, on="order_id", how="left")

def add_category_features(orders_feature_df, order_items_df, order_col_name):
    """Add category and base product category one-hot features per order.

    Args:
        orders_feature_df (pd.DataFrame): Order-level feature dataframe.
        order_items_df (pd.DataFrame): Cleaned order items dataframe.

    Returns:
        pd.DataFrame: Feature dataframe with category columns added.
    """
    logger.info("Creating category and base product category features...")
    def norm(val, prefix):
        val = re.sub(r'[\W\s]+', '_', str(val)).strip('_').lower()
        return prefix + val if val else ""
    df = order_items_df.copy()
    for col, prefix in [("Final Dri Category", "dri_categ_"), ("Clubbed base product category", "categ_base_product_")]:
        df[col] = df[col].astype(str).str.replace(r'[\W\s]+', '_', regex=True).str.strip('_').str.lower()
    main_cat_counts = (
        df.groupby([order_col_name, "Final Dri Category"])['quantity']
        .count().clip(upper=1).unstack(fill_value=0).astype(int).reset_index()
    )
    main_cat_counts.columns = ['order_id' if col == order_col_name else norm(col, "dri_categ_") for col in main_cat_counts.columns]
    base_cat_counts = (
        df.groupby([order_col_name, "Clubbed base product category"])['quantity']
        .count().clip(upper=1).unstack(fill_value=0).astype(int).reset_index()
    )
    base_cat_counts.columns = ['order_id' if col == order_col_name else norm(col, "categ_base_product_") for col in base_cat_counts.columns]
    merged = orders_feature_df.merge(main_cat_counts, on="order_id", how="left")
    merged = merged.merge(base_cat_counts, on="order_id", how="left")
    logger.info(f"Category features added. New shape: {merged.shape}")
    return merged

def add_derived_order_features(orders_feature_df, order_items_df, order_items_deduped_df, channel, order_col_name):
    """Add derived features: customization flag, veg flag, item/promo counts, unique categories per order.

    Args:
        orders_feature_df (pd.DataFrame): Order-level features dataframe.
        order_items_df (pd.DataFrame): All order items (promo/choice included).
        order_items_deduped_df (pd.DataFrame): Deduped items (choice/promo excluded).

    Returns:
        pd.DataFrame: Feature dataframe with derived columns added.
    """
    logger.info("Adding derived features (customization, veg, promo, unique cats)...")
    items_df = order_items_df.copy()
    # instore items data has only quantity information
    if channel == 'instore':
        items_df['Qty'] = items_df['Qty'].astype(int)
        agg_feats = items_df.groupby("InvoiceNo").agg(
            item_count=("Qty", "sum")
        ).reset_index().rename(columns={"InvoiceNo": "order_id"})
        feature_df = orders_feature_df.merge(agg_feats, on="order_id", how="left")

    # this will be executed for swiggy and ownapp
    else:
        items_df['quantity'] = items_df['quantity'].astype(int)
        items_df['is_veg'] = items_df['is_veg'].astype(int)
        agg_feats = items_df.groupby("sales_id").agg(
            customization_flag=("is_customized", "max"),
            veg_indicator=("is_veg", "min"),
            item_count=("quantity", "sum"),
            veg_count=("is_veg", "sum")
        ).reset_index().rename(columns={"sales_id": "order_id"})
        feature_df = orders_feature_df.merge(agg_feats, on="order_id", how="left")
        # Promo items
        promo_items = items_df[items_df['promo_item'] == True]
        promo_summary = (
            promo_items.groupby("sales_id")['quantity']
            .sum()
            .reset_index(name="promo_items")
            .rename(columns={"sales_id": "order_id"})
        )
        feature_df = feature_df.merge(promo_summary, on="order_id", how="left")
        feature_df["promo_items"] = feature_df["promo_items"].fillna(0).astype(int)

    # Get Unique dri categories in an order
    deduped_df = order_items_deduped_df.copy()
    unique_dri_cats = (
        deduped_df.dropna(subset=["Final Dri Category"])
        .groupby(order_col_name, sort=False)["Final Dri Category"]
        .agg(lambda x: list(set(x.tolist())))
        .reset_index()
        .rename(columns={"Final Dri Category": "unique_dri_categories_in_order", order_col_name: "order_id"})
    )
    feature_df = feature_df.merge(unique_dri_cats, on="order_id", how="left")
    
    # Get Unique base products in an order
    unique_base_products = (
        deduped_df.dropna(subset=["Clubbed base product category"])
        .groupby(order_col_name, sort=False)["Clubbed base product category"]
        .agg(lambda x: list(set(x.tolist())))
        .reset_index()
        .rename(columns={"Clubbed base product category": "unique_base_products_in_order", order_col_name: "order_id"})
    )
    feature_df = feature_df.merge(unique_base_products, on="order_id", how="left")

    feature_df["order_time_of_day"] = feature_df["order_placed_datetime"].dt.hour.map(
        lambda h: "Breakfast" if 6 <= h < 11 else
                "Lunch" if 11 <= h < 15 else
                "Evening" if 15 <= h < 19 else
                "Dinner" if 19 <= h <= 23 else
                "Late Night"
    )

    feature_df["is_weekend"] = (feature_df["order_placed_datetime"].dt.weekday >= 5).astype(int)

    logger.info(f"Derived features added. Final shape: {feature_df.shape}")
    return feature_df

def standardize_and_write_parquet(df, output_path):
    """Standardize algonomy_id column and write the DataFrame to Parquet.

    Args:
        df (pd.DataFrame): Final features dataframe.
        output_path (str): Output path for Parquet file.

    Returns:
        None
    """
    logger.info(f"Standardizing algonomy_id and writing output to {output_path} ...")
    df['algonomy_id'] = df['algonomy_id'].astype('string')
    df.to_parquet(output_path)
    logger.info("Write to Parquet complete.")

