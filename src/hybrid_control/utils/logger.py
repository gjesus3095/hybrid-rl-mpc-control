import logging

COLOR_RESET = "\033[0m"
COLOR_MAP = {
    "DEBUG": "\033[94m",
    "INFO": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[91m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLOR_MAP.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{COLOR_RESET}"


logger = logging.getLogger("TMR")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    base_format = "[%(levelname)s] %(asctime)s - %(message)s"
    formatter = ColorFormatter(base_format, datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False
