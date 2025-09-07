import argparse
import logging
from pathlib import Path
from segmentation.src.main.segmentation import SegmentRunner
from features.src.main.order_feature_pipeline import OrderFeaturePipeline
from features.src.main.customer_feature_pipeline import CustomerFeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_config", required=True)
    parser.add_argument("--feature_pipeline_config", required=True)
    args = parser.parse_args()

    logger.info("Starting full pipeline: ownapp -> swiggy -> instore ==>> customer")
    # Run orders pipeline
    logger.info("Running OrderFeaturePipeline...")
    ownapp_pipeline = OrderFeaturePipeline(args.feature_pipeline_config, "ownapp")
    ownapp_pipeline.run()

    swiggy_pipeline = OrderFeaturePipeline(args.feature_pipeline_config, "swiggy")
    swiggy_pipeline.run()

    instore_pipeline = OrderFeaturePipeline(args.feature_pipeline_config, "instore")
    instore_pipeline.run_instore()

    # Run customer feature pipeline
    logger.info("Running CustomerFeaturePipeline...")
    customer_pipeline = CustomerFeaturePipeline(args.feature_pipeline_config)
    customer_features_file = customer_pipeline.run()
    logger.info(f"[FEATURES] Customer Feature Output file saved to {customer_features_file}")

    # Start Segmentation
    logger.info(f"[SEGMENTATION] Starting Segmentation Pipeline")
    runner = SegmentRunner(args.segmentation_config)

    original_df, combined_segments_df = runner.run_rule_pipeline()
    runner.save_segments(combined_segments_df)
    logger.info(f"[SEGMENTATION] Output file saved to {runner.config['output_segments_path']}")

    microsegments_df = runner.run_microsegments_from_config(original_df, combined_segments_df)
    base_ms_df = microsegments_df.merge(original_df,on='algonomy_id', how='left')
    base_ms_df.to_parquet(runner.config["output_microsegments_path"], index=False)
    logger.info(f"[MICRO-SEGMENTATION] Output file saved to {runner.config['output_microsegments_path']}")

if __name__ == "__main__":
    main()
