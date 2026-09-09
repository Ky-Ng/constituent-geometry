import logging
from pathlib import Path
import sys

from datetime import datetime, timezone

def init_logger(out_dir: str, filename: str | None = None):
    if not filename:
        # 1. Get current time in UTC (best practice for logging)
        now = datetime.now(timezone.utc)

        # 2. Format as YYYY-MM-DD_HH-MM-SS (Windows/Linux safe)
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        # 3. Create the filename
        filename = f"{timestamp}.log"

    # Configure handlers
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(out_dir) / filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)

    # Initialize logging with both handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, stdout_handler]
    )

    # This will write to both app.log and your console terminal
    logging.info("Initializing logger!")
