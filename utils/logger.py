#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志模块 - 提供全局日志功能 - 优化版本支持结构化日志和性能监控
"""

import os
import sys
import logging
import json
import time
import threading
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional
from datetime import datetime

# 日志级别映射
log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# 默认配置
DEFAULT_LOG_LEVEL = "info"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DEFAULT_JSON_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DEFAULT_LOG_FILE = "logs/app.log"
DEFAULT_JSON_LOG_FILE = "logs/app_structured.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

# 确保日志目录存在
os.makedirs(os.path.dirname(DEFAULT_LOG_FILE), exist_ok=True)

# 全局日志级别设置
log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).lower()
global_log_level = log_levels.get(log_level, logging.INFO)
enable_structured_logs = os.environ.get("ENABLE_STRUCTURED_LOGS", "false").lower() == "true"


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def format(self, record):
        # 创建基础日志记录
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "filename": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "message": record.getMessage(),
            "thread": threading.current_thread().name,
            "process": os.getpid()
        }

        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # 添加额外的结构化字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class PerformanceLogger:
    """性能日志记录器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}

    def start_timer(self, operation: str):
        """开始计时"""
        self._timers[operation] = time.time()

    def end_timer(self, operation: str, level: int = logging.INFO, **extra):
        """结束计时并记录"""
        if operation in self._timers:
            duration = (time.time() - self._timers[operation]) * 1000  # 转换为毫秒
            extra_fields = {
                'operation': operation,
                'duration_ms': round(duration, 2),
                'performance_metric': True,
                **extra
            }
            self._log_with_extra(level, f"操作完成: {operation}", **extra_fields)
            del self._timers[operation]
        else:
            self.logger.warning(f"未找到操作计时器: {operation}")

    def _log_with_extra(self, level: int, message: str, **extra):
        """记录带额外字段的日志"""
        if hasattr(self.logger.handlers[0], 'formatter') and isinstance(
            self.logger.handlers[0].formatter, StructuredFormatter
        ):
            # 结构化日志
            record = self.logger.makeRecord(
                self.logger.name, level, "", 0, message, (), None
            )
            record.extra_fields = extra
            self.logger.handle(record)
        else:
            # 普通日志
            extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
            full_message = f"{message} | {extra_str}"
            self.logger.log(level, full_message)


# 创建一个名为 'app' 的父logger
logger = logging.getLogger("app")
logger.setLevel(global_log_level)
logger.propagate = False  # 不向root logger传播

# 性能日志记录器
performance_logger = PerformanceLogger(logger)

# 只给父logger配置handlers
if not logger.handlers:
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    if enable_structured_logs:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    logger.addHandler(console_handler)

    # 普通文件处理器
    try:
        file_handler = RotatingFileHandler(
            DEFAULT_LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"无法创建日志文件处理器: {str(e)}", exc_info=True)

    # 结构化日志文件处理器（如果启用）
    if enable_structured_logs:
        try:
            json_file_handler = RotatingFileHandler(
                DEFAULT_JSON_LOG_FILE,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8"
            )
            json_file_handler.setFormatter(StructuredFormatter())
            logger.addHandler(json_file_handler)
        except Exception as e:
            logger.warning(f"无法创建结构化日志文件处理器: {str(e)}", exc_info=True)

def get_logger(name=None, with_performance: bool = False):
    """
    获取一个 'app' logger的子logger - 优化版本支持性能日志

    日志消息会通过这个子logger传播到父logger 'app',
    然后由 'app' logger的处理器来处理.

    Args:
        name: logger名称, 如果为None则使用调用模块的名称
        with_performance: 是否包含性能日志功能

    Returns:
        logging.Logger 或 EnhancedLogger: 配置好的logger实例
    """
    if name is None:
        # 获取调用者的模块名
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')

        # 如果是__main__, 使用文件名
        if name == '__main__':
            filename = frame.f_globals.get('__file__', 'main')
            name = os.path.splitext(os.path.basename(filename))[0]

    # 所有获取的logger都是'app'的子logger
    child_logger = logging.getLogger(f"app.{name}")

    if with_performance:
        return EnhancedLogger(child_logger, performance_logger)
    else:
        return child_logger


class EnhancedLogger:
    """增强的日志记录器，支持性能监控和结构化日志"""

    def __init__(self, logger: logging.Logger, perf_logger: PerformanceLogger):
        self.logger = logger
        self.perf_logger = perf_logger

    def debug(self, message, **kwargs):
        """记录DEBUG级别日志"""
        self._log_with_extra(logging.DEBUG, message, **kwargs)

    def info(self, message, **kwargs):
        """记录INFO级别日志"""
        self._log_with_extra(logging.INFO, message, **kwargs)

    def warning(self, message, **kwargs):
        """记录WARNING级别日志"""
        self._log_with_extra(logging.WARNING, message, **kwargs)

    def error(self, message, **kwargs):
        """记录ERROR级别日志"""
        self._log_with_extra(logging.ERROR, message, **kwargs)

    def critical(self, message, **kwargs):
        """记录CRITICAL级别日志"""
        self._log_with_extra(logging.CRITICAL, message, **kwargs)

    def exception(self, message, **kwargs):
        """记录异常日志"""
        kwargs['exc_info'] = True
        self._log_with_extra(logging.ERROR, message, **kwargs)

    def performance(self, operation: str, duration_ms: float, **kwargs):
        """记录性能指标"""
        extra_fields = {
            'operation': operation,
            'duration_ms': round(duration_ms, 2),
            'performance_metric': True,
            **kwargs
        }
        self._log_with_extra(logging.INFO, f"性能指标: {operation}", **extra_fields)

    def start_timer(self, operation: str):
        """开始性能计时"""
        self.perf_logger.start_timer(operation)

    def end_timer(self, operation: str, **kwargs):
        """结束性能计时"""
        self.perf_logger.end_timer(operation, **kwargs)

    def _log_with_extra(self, level: int, message: str, **kwargs):
        """记录带额外字段的日志"""
        # 分离日志参数和额外字段
        log_params = {}
        extra_fields = {}

        for key, value in kwargs.items():
            if key in ['exc_info', 'stack_info', 'extra']:
                log_params[key] = value
            else:
                extra_fields[key] = value

        if extra_fields:
            # 检查是否使用结构化日志
            structured_handler = None
            for handler in self.logger.handlers:
                if (hasattr(handler, 'formatter') and
                    isinstance(handler.formatter, StructuredFormatter)):
                    structured_handler = handler
                    break

            if structured_handler:
                # 结构化日志
                record = self.logger.makeRecord(
                    self.logger.name, level, "", 0, message, (), None
                )
                record.extra_fields = extra_fields
                self.logger.handle(record)
            else:
                # 普通日志，将额外字段添加到消息中
                extra_str = " | ".join(f"{k}={v}" for k, v in extra_fields.items())
                full_message = f"{message} | {extra_str}"
                self.logger.log(level, full_message, **log_params)
        else:
            # 普通日志
            self.logger.log(level, message, **log_params)

    def __getattr__(self, name):
        """代理其他logger方法"""
        return getattr(self.logger, name)


# 导出全局日志对象和获取logger的函数
__all__ = [
    "logger",
    "get_logger",
    "performance_logger",
    "StructuredFormatter",
    "EnhancedLogger"
]