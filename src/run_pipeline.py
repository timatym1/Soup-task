import subprocess
import sys

from src.common import get_logger, snapshot_gpu


STAGES = [
    "src.data_preparation",
    "src.memory_analysis",
    "src.train",
    "src.merge_model",
    "src.quantization",
    "src.verification",
    "src.model_evaluation",
    "src.silent_failures",
]


def main() -> None:
    logger = get_logger("pipeline", "pipeline.log")
    snapshot_gpu(logger, "pipeline_start")
    for stage in STAGES:
        logger.info("Starting stage %s", stage)
        try:
            subprocess.run([sys.executable, "-m", stage], check=True)
        except subprocess.CalledProcessError:
            logger.exception("Stage %s failed; generating reports from available evidence", stage)
            subprocess.run([sys.executable, "-m", "src.silent_failures"], check=False)
            raise
        logger.info("Completed stage %s", stage)
    snapshot_gpu(logger, "pipeline_complete")


if __name__ == "__main__":
    main()
