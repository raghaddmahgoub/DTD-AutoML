from utils.logger import Logger

logger = Logger()

logger.info("Starting AutoML agent initialization...")
logger.warn("Low memory detected, continuing with reduced batch size.")
try:
    1 / 0
except Exception as e:
    logger.error("An unexpected error occurred while running training.", e)
