"""日志模块 - 使用 rich + logging 提供美观的日志记录"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback


# 安装 rich 的美化异常显示
install_rich_traceback(show_locals=False, suppress=[])

# Rich 控制台
console = Console()

# 全局 logger
_logger: Optional[logging.Logger] = None


def setup_logger(log_file: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    初始化日志系统
    
    Args:
        log_file: 日志文件路径
        log_level: 日志级别
    
    Returns:
        配置好的 logger 实例
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # 创建 logger
    _logger = logging.getLogger("FaceSeeker")
    _logger.setLevel(log_level)
    
    # 防止重复添加 handler
    if _logger.handlers:
        return _logger
    
    # 1. 控制台输出 - 使用 RichHandler
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(message)s",
        datefmt="[%X]"
    )
    console_handler.setFormatter(console_formatter)
    _logger.addHandler(console_handler)
    
    # 2. 文件输出 - 使用 RotatingFileHandler（自动轮转，最多保留 5 个备份）
    try:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        _logger.addHandler(file_handler)
    except Exception as e:
        console.print(f"[yellow]警告：无法创建日志文件 {log_file}: {e}[/yellow]")
    
    return _logger


def get_logger() -> logging.Logger:
    """获取全局 logger 实例"""
    if _logger is None:
        # 如果未初始化，创建一个基础的 logger
        return setup_logger("faceseeker.log")
    return _logger


def log_opencv_error(func_name: str, error: Exception, suppress: bool = True):
    """
    记录 OpenCV 相关错误
    
    Args:
        func_name: 函数名称
        error: 异常对象
        suppress: 是否抑制错误（仅记录日志，不影响程序继续运行）
    """
    logger = get_logger()
    error_msg = f"OpenCV 错误 [{func_name}]: {type(error).__name__}: {str(error)}"
    
    if suppress:
        logger.warning(error_msg)
    else:
        logger.error(error_msg, exc_info=True)


def log_exception(message: str, exc: Exception, crash: bool = False):
    """
    记录异常信息
    
    Args:
        message: 描述信息
        exc: 异常对象
        crash: 是否为致命错误（是否需要退出程序）
    """
    logger = get_logger()
    if crash:
        logger.critical(f"{message}: {exc}", exc_info=True)
    else:
        logger.error(f"{message}: {exc}", exc_info=True)
