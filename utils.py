import inspect
import logging
import sys

import transformers
from loguru import logger


# transformersのloggerをloguruへ送るための設定
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.S}</green> | "
    "<level>{level}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
)
# loguru に出力フォーマットを設定
logger.remove()  # デフォルトのハンドラを削除
logger.add(sys.stdout, format=log_format, level="INFO")

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

transformers.logging.set_verbosity_info()
transformers.logging.disable_default_handler()
transformers.logging.add_handler(InterceptHandler())
