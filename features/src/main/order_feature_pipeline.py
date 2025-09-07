import json
import os
from features.src.main.utils.pipelines_utils import *
from features.src.main.utils.instore_pipelines_utils import *
import logging

logger = logging.getLogger(__name__)


class OrderFeaturePipeline:
    """
    Pipeline to process and engineer features from orders data for different sales channels.
    """

    def __init__(self, config_path:str, channel: str, storage_options=None):
        """
        Initializes the pipeline, loads configuration files and sets channel-specific settings.
        
        Args:
            config_path (str): COnfig path passed as arg.
            channel (str): Name of the sales channel.
            storage_options (dict, optional): Additional options for storage operations.
        """

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        config_path = os.path.join(root_dir, config_path)
        # Load the configuration file
        with open(config_path, "r") as f:
            config_all= json.load(f)

        self.channel = channel
        self.all_config = config_all
        self.channel_config = config_all[channel]
        self.common_config = config_all["common"]
        self.storage_options = storage_options or {}
        
        # Maps raw features to standardized names for output
        self.feature_map = {
            "order_id": "id",
            "user_id": "user_id",
            "device_type_platform": "platform",
            "order_placed_datetime": "order_placed_datetime",
            "order_channel": "channel",
            "store_name": "store_name",
            "store_mds_id": "store_id_id",
            "actual_delivery_time": "order_delivery_datetime",
            "payment_method": "payment_method",
            "pre_discount_order_value": "gross_price",
            "promo_code_applied": "applied_coupon",
            "discounted_price": "discounted_price",
            "total_price": "total_price",
            "gross_price": "gross_price",
            "actual_subtotal": "actual_subtotal",
            "sub_total_price": "sub_total_price",
            "schedule_datetime": "schedule_datetime",
            "status_id": "status_id",
            "applied_coupon": "applied_coupon",
            "platform": "platform",
            "tip": "tip",
            "order_delivery_datetime": "order_delivery_datetime",
            "before_payment_delivery_eta": "before_payment_delivery_eta",
            "before_payment_pickup_eta": "before_payment_pickup_eta",
            "is_distance_fee_applicable": "is_distance_fee_applicable",
            "offer_applied": "offer_applied",
            "business_model_id_id": "business_model_id_id",
            "algonomy_id": "algonomy_id",
            "store_type": "Store Type",
            "store_generator_new": "Generator New",
            "store_geographical_city": "Geographical City",
            "store_key_city": "Key City",
            "has_query": "has_query"
        }

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        config_path = os.path.join(root_dir, config_path)
        # Load the configuration file
        with open(config_path, "r") as f:
            config_all= json.load(f)


    def run(self):
        """
        Main pipeline execution for standard channels.
        Loads orders and order items, processes them, engineers features, and writes output parquet.
        """

        logger.info(f'Pipeline starting for "{self.channel}"')

        # Step 1: Load and process orders data
        df_orders = read_orders(self.channel_config, self.storage_options)
        df_orders = localize_and_filter_dates(
            df_orders, 
            self.channel_config, 
            self.common_config, 
            ["order_placed_datetime", "order_delivery_datetime", "schedule_datetime"]
        )
        df_orders = map_to_centralized_customer_id(
            df_orders, self.common_config, self.storage_options
        )

        # Special filtering for Swiggy channel to consider only OC users' orders
        # Note: 'ownapp' processing must run prior due to dependency
        if self.channel == 'swiggy':
            df_orders = filter_non_oc_users_order(df_orders, self.all_config)

        df_orders = map_store_details(df_orders, self.common_config, self.storage_options)
        df_orders = flag_orders_with_queries(df_orders, self.channel_config, self.storage_options)

        # Step 2: Load and process order items
        df_order_items = map_order_items_to_categories(
            df_orders, self.channel_config, self.common_config, self.storage_options
        )
        df_order_items_v2 = remove_choice_and_promo_items(
            df_order_items, self.common_config, self.storage_options
        )
        df_order_items_v2 = remove_unmapped_or_excluded_items(df_order_items_v2)

        # Step 3: Feature engineering
        orders_feature_df = create_base_order_features(df_orders, self.feature_map)
        orders_feature_df = add_price_bucket_features(
            orders_feature_df, df_order_items_v2, 
            price_col_name='original_price', 
            order_col_name='sales_id', 
            common=self.common_config
        )
        orders_feature_df = add_category_features(
            orders_feature_df, df_order_items_v2, order_col_name='sales_id'
        )
        orders_feature_df = add_derived_order_features(
            orders_feature_df, df_order_items, df_order_items_v2, 
            channel=self.channel, order_col_name='sales_id'
        )

        # Step 4: Write output parquet
        standardize_and_write_parquet(
            orders_feature_df, self.channel_config["output_path"]
        )

        logger.info(f"Feature engineering completed. Output written to {self.channel_config['output_path']}.")

    def run_instore(self):
        """
        Pipeline execution for 'instore' channel.
        Processes instore orders and order items, engineers features, and writes output parquet.
        """

        logger.info(f'Pipeline starting for "{self.channel}"')
        
        # Step 1: Load and process instore orders data
        orders_dd = load_and_prepare_instore_orders(
            self.all_config, self.common_config, storage_options={'anon': False}
        )

        # Step 2: Load and process order items
        df_order_items = load_and_prepare_instore_order_items(
            self.channel_config, orders_dd, self.common_config, storage_options={'anon': False})
        # Map instore order items to categories
        df_order_items = map_instore_items_to_categories(
            df_order_items, self.common_config, storage_options={'anon': False}
        )
        #Remove choice and promo items
        df_order_items_v2 = remove_instore_choice_and_promo_items(
            df_order_items, self.common_config, storage_options={'anon': False}
        )
        #Remove unmapped/excluded items
        df_order_items_v2 = remove_unmapped_or_excluded_items(df_order_items_v2)

        # Step 3: Feature engineering
        orders_feature_df = create_instore_base_order_features(orders_dd)
        orders_feature_df = add_price_bucket_features(
            orders_feature_df, df_order_items_v2, 
            price_col_name='PriceNet', order_col_name='InvoiceNo', 
            common=self.common_config
        )
        orders_feature_df = add_category_features(
            orders_feature_df, df_order_items_v2, order_col_name='InvoiceNo'
        )
        orders_feature_df = add_derived_order_features(
            orders_feature_df, df_order_items, df_order_items_v2, 
            channel=self.channel, order_col_name='InvoiceNo'
        )

        # Step 4: Write output parquet
        standardize_and_write_parquet(
            orders_feature_df, self.channel_config["output_path"]
        )

        logger.info(f"Feature engineering completed. Output written to {self.channel_config['output_path']}.")
