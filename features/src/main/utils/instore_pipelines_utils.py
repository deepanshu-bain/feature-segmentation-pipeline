import dask.dataframe as dd
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

def load_and_prepare_instore_orders(all_config, common, storage_options=None):
    """
    Loads and merges in-store order data with mapping, filters by date/store, and filters by users if needed.
    
    Args:
        all_config (dict): Config dictionary containing 'instore' and 'ownapp' keys.
        common (dict): Dictionary of common config values (e.g. date, mapping file).
        storage_options (dict, optional): S3 or other storage options for Dask/Pandas.

    Returns:
        pd.DataFrame: Filtered and merged orders DataFrame (in-memory).
    """
    config = all_config['instore']
    storage_options = storage_options or {}

    run_date = pd.to_datetime(common["current_run_date"])
    start_date = run_date - pd.DateOffset(months=common["months_offset"])
    logger.info(f"Filtering data from {start_date} to {run_date} (IST)")

    orders_dd = dd.read_csv(
        config["orders_file_pattern"],
        storage_options=storage_options,
        assume_missing=True
    )
    logger.info("Loaded in-store order data as Dask DataFrame.")

    orders_dd['BussTime'] = dd.to_datetime(orders_dd['BussTime'])
    orders_dd = orders_dd[
        (orders_dd.StoreCode != 9993) &
        (orders_dd.BussTime >= start_date) &
        (orders_dd.BussTime < run_date)
    ]
    orders_dd['Business_Date'] = orders_dd['BussTime'].dt.date

    logger.info("Reading customer mapping file for algonomy_id mapping...")
    mapping = pd.read_csv(
        common["mapping_file"],
        dtype={'customer_code_ty': 'object'},
        storage_options=storage_options
    )
    mapping["invoice_date"] = pd.to_datetime(mapping["invoice_date"]).dt.date
    mapping["invoice_no"] = mapping["invoice_no"].astype('Int64')
    mapping.rename(columns={"invoice_no": "sahil_map_invoice_no", "customer_code_ty": "algonomy_id"}, inplace=True)

    # Coerce join column types
    orders_dd['InvoiceNo'] = orders_dd['InvoiceNo'].astype('Int64')
    orders_dd['Business_Date'] = orders_dd['Business_Date'].astype('string')
    orders_dd['StoreCode'] = orders_dd['StoreCode'].astype('int64')

    logger.info("Merging order data with customer mapping for algonomy_id...")
    orders_dd = orders_dd.merge(
        mapping,
        left_on=["InvoiceNo", "Business_Date", "StoreCode"],
        right_on=["sahil_map_invoice_no", "invoice_date", "store_code"],
        how="left"
    )

    logger.info("Filtering orders by OwnApp user list...")
    users_df = pd.read_parquet(all_config['ownapp']['output_path'], columns=["algonomy_id"])
    user_ids = users_df["algonomy_id"].dropna().unique().tolist()
    orders_dd = orders_dd[orders_dd['algonomy_id'].isin(user_ids)]
    logger.info(f"Applied user filter using {all_config['ownapp']['output_path']}")

    orders_dd = orders_dd.compute()
    logger.info(f"Final in-store orders loaded: {len(orders_dd)} rows after join/filter.")

    logger.info("Mapping with store master data...")
    df_store_master = pd.read_csv(common['store_master_path'])
    orders_dd = orders_dd.merge(df_store_master, left_on='StoreCode', right_on='Store Code', how='left')
    logger.info(f"Store details mapped. Null Store Name: {orders_dd['Store Name'].isna().sum()}")

    return orders_dd

def load_and_prepare_instore_order_items(config, orders_df, common, storage_options=None):
    """
    Loads and filters in-store order items, keeping only those in the provided orders DataFrame.

    Args:
        config (dict): In-store config with item file pattern.
        orders_df (pd.DataFrame): Orders DataFrame (pandas, with InvoiceNo).
        common (dict): Common config values.
        storage_options (dict, optional): S3 or other storage options.

    Returns:
        pd.DataFrame: Filtered item-level data as pandas DataFrame.
    """
    storage_options = storage_options or {}

    run_date = pd.to_datetime(common["current_run_date"])
    start_date = run_date - pd.DateOffset(months=common["months_offset"])
    logger.info(f"Filtering items from {start_date} to {run_date} and excluding store 9993.")

    df_order_items = dd.read_csv(
        config["order_items_file_pattern"],
        storage_options=storage_options,
        assume_missing=True
    )
    logger.info("Loaded in-store order items as Dask DataFrame.")

    df_order_items['OrderDatetime'] = dd.to_datetime(df_order_items['OrderDatetime'])
    df_order_items = df_order_items[
        (df_order_items['StoreCode'] != 9993) &
        (df_order_items['OrderDatetime'] >= start_date) &
        (df_order_items['OrderDatetime'] < run_date)
    ]
    invoice_nos = orders_df["InvoiceNo"].dropna().unique().tolist()
    df_order_items = df_order_items[df_order_items["InvoiceNo"].isin(invoice_nos)]

    df_order_items = df_order_items.compute()
    logger.info(f"Final order items loaded: {len(df_order_items)} rows after filter.")

    return df_order_items

def map_instore_items_to_categories(order_items_df, common, storage_options=None):
    """
    Maps in-store order items to categories using ProductCode.

    Args:
        order_items_df (pd.DataFrame): Items DataFrame (in-store).
        common (dict): Common config values including category mapping path.
        storage_options (dict, optional): S3 or other storage options.

    Returns:
        pd.DataFrame: Items DataFrame with category columns joined.
    """
    storage_options = storage_options or {}
    logger.info("Loading item category mapping for in-store join...")
    df_items_category = pd.read_csv(
        common["categories_path"],
        dtype={'ProductCode': str},
        storage_options=storage_options
    )
    merged_df = order_items_df.merge(
        df_items_category,
        on="ProductCode",
        how="left"
    )
    logger.info("Mapped in-store items to categories.")
    return merged_df

def remove_instore_choice_and_promo_items(order_items_df, common, storage_options=None):
    """
    Removes choice and promo items from in-store order items using the clear_master mapping.

    Args:
        order_items_df (pd.DataFrame): In-store order items DataFrame.
        common (dict): Common config (clear_master path).
        storage_options (dict, optional): S3 or other storage options.

    Returns:
        pd.DataFrame: Items DataFrame with choice/promo items removed.
    """
    storage_options = storage_options or {}
    logger.info("Removing choice and promo items using clear_master mapping...")
    clear_master = pd.read_csv(
        common["clear_master_path"],
        usecols=["ChoiceItemId", "HeaderItemId", "Qty", "header_item_id_id", "choice_item_id_id"],
        storage_options=storage_options
    )
    header_orders = order_items_df.merge(
        clear_master[["HeaderItemId"]].drop_duplicates(),
        left_on="ProductCode", right_on="HeaderItemId"
    )[["InvoiceNo", "ProductCode"]].rename(columns={"ProductCode": "HeaderItemId"})
    order_header_choices = header_orders.merge(
        clear_master[["HeaderItemId", "ChoiceItemId", "Qty"]],
        on="HeaderItemId",
        how="left"
    )
    choice_counts = order_items_df.groupby(
        ["InvoiceNo", "ProductCode"]
    ).agg(
        choice_count=("ProductCode", "count"),
        order_quantity=("Qty", "sum")
    ).reset_index().rename(columns={"ProductCode": "ChoiceItemId"})
    merged = order_header_choices.merge(
        choice_counts,
        on=["InvoiceNo", "ChoiceItemId"],
        how="left"
    )
    to_remove = merged[
        (merged["choice_count"] > 0) &
        (merged["Qty"] >= merged["order_quantity"]) &
        (~merged["order_quantity"].isna())
    ][["InvoiceNo", "ChoiceItemId"]].drop_duplicates()
    mask_remove = order_items_df.set_index(["InvoiceNo", "ProductCode"]).index.isin(
        to_remove.set_index(["InvoiceNo", "ChoiceItemId"]).index
    )
    df_no_choice = order_items_df[~mask_remove].copy()
    logger.info(f"Choice and promo item removal done. Remaining items: {len(df_no_choice)}")
    return df_no_choice

def create_instore_base_order_features(orders_df):
    """
    Create a DataFrame with base features for in-store orders.

    Args:
        orders_df (pd.DataFrame): Orders DataFrame (already processed/filtered).

    Returns:
        pd.DataFrame: DataFrame with base features.
    """
    logger.info("Creating base features DataFrame for in-store orders.")
    feature_map = {
        "order_id": "InvoiceNo",
        "order_placed_datetime": "BussTime",
        "BussTime": "BussTime",
        "StoreCode": "StoreCode",
        "total_price": "GrossAmount",
        "discounted_price": "Discount",
        "BaseAmount": "BaseAmount",
        "TransType": "TransType",
        "algonomy_id": "algonomy_id",
        "SaleType": "SaleType",
        "store_type": "Store Type",
        "store_generator_new": "Generator New",
        "store_geographical_city": "Geographical City",
        "store_key_city": "Key City"
    }
    features_df = pd.DataFrame({k: orders_df[v] for k, v in feature_map.items() if v in orders_df.columns})
    logger.info(f"Base features DataFrame shape: {features_df.shape}")
    return features_df
