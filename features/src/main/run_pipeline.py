import logging

from src.main.order_feature_pipeline import OrderFeaturePipeline
from src.main.customer_feature_pipeline import CustomerFeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Starting full pipeline: ownapp -> swiggy -> instore -> customer...")

    # Run orders pipeline
    logger.info("Running OrderFeaturePipeline...")
    ownapp_pipeline = OrderFeaturePipeline("ownapp")
    ownapp_pipeline.run()

    swiggy_pipeline = OrderFeaturePipeline("swiggy")
    swiggy_pipeline.run()

    instore_pipeline = OrderFeaturePipeline("instore")
    instore_pipeline.run_instore()

    # Run customer feature pipeline
    logger.info("Running CustomerFeaturePipeline...")
    customer_pipeline = CustomerFeaturePipeline()
    customer_pipeline.run()

    logger.info("All pipelines completed successfully.")
