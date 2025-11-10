"""
性能监控器模块
提供系统性能指标收集、记录和分析功能
"""

import time
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import json
from datetime import datetime, timedelta
from utils.logger import get_logger


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: float
    metric_type: str
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """性能统计信息"""
    metric_type: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    sum_value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """性能监控器 - 收集和分析系统性能指标"""

    def __init__(self, max_history: int = 10000, cleanup_interval: int = 300):
        """
        初始化性能监控器

        Args:
            max_history: 最大历史记录数量
            cleanup_interval: 清理间隔（秒）
        """
        self.max_history = max_history
        self.cleanup_interval = cleanup_interval
        self.logger = get_logger(__name__)

        # 性能指标存储
        self._metrics: deque = deque(maxlen=max_history)
        self._stats_cache: Dict[str, PerformanceStats] = {}
        self._lock = threading.Lock()

        # 启动清理任务
        self._cleanup_task = None
        self._running = False
        self.start()

    def start(self):
        """启动性能监控器"""
        if not self._running:
            self._running = True
            self._cleanup_task = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="PerformanceMonitor-Cleanup"
            )
            self._cleanup_task.start()
            self.logger.info("性能监控器已启动")

    def stop(self):
        """停止性能监控器"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.join(timeout=5)
        self.logger.info("性能监控器已停止")

    def record_metric(
        self,
        metric_type: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        记录性能指标

        Args:
            metric_type: 指标类型（如 'response_time', 'queue_size'）
            value: 指标值
            unit: 单位（如 'ms', 'count'）
            tags: 标签字典
            metadata: 额外元数据
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_type=metric_type,
            value=value,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )

        with self._lock:
            self._metrics.append(metric)
            # 清除相关统计缓存
            cache_key = f"{metric_type}:{json.dumps(tags or {}, sort_keys=True)}"
            self._stats_cache.pop(cache_key, None)

        self.logger.debug(f"记录性能指标: {metric_type}={value}{unit}")

    def record_function_execution(
        self,
        metric_type: str,
        func: Callable,
        *args,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        记录函数执行性能

        Args:
            metric_type: 指标类型
            func: 要执行的函数
            tags: 标签字典
            *args, **kwargs: 函数参数

        Returns:
            函数执行结果
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            raise
        finally:
            execution_time = (time.time() - start_time) * 1000  # 转换为毫秒

            # 记录执行时间
            self.record_metric(
                metric_type=metric_type,
                value=execution_time,
                unit="ms",
                tags=tags,
                metadata={
                    "success": success,
                    "error": error
                }
            )

        return result

    async def record_async_function_execution(
        self,
        metric_type: str,
        func: Callable,
        *args,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        记录异步函数执行性能

        Args:
            metric_type: 指标类型
            func: 要执行的异步函数
            tags: 标签字典
            *args, **kwargs: 函数参数

        Returns:
            函数执行结果
        """
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            raise
        finally:
            execution_time = (time.time() - start_time) * 1000  # 转换为毫秒

            # 记录执行时间
            self.record_metric(
                metric_type=metric_type,
                value=execution_time,
                unit="ms",
                tags=tags,
                metadata={
                    "success": success,
                    "error": error,
                    "async": True
                }
            )

        return result

    def get_stats(
        self,
        metric_type: str,
        tags: Optional[Dict[str, str]] = None,
        time_window: Optional[int] = None
    ) -> Optional[PerformanceStats]:
        """
        获取性能统计信息

        Args:
            metric_type: 指标类型
            tags: 标签过滤条件
            time_window: 时间窗口（秒），None表示全部

        Returns:
            统计信息或None
        """
        cache_key = f"{metric_type}:{json.dumps(tags or {}, sort_keys=True)}:{time_window or 'all'}"

        with self._lock:
            # 检查缓存
            if cache_key in self._stats_cache:
                cached_time = self._stats_cache[cache_key].metadata.get('cache_time', 0)
                if time.time() - cached_time < 60:  # 缓存1分钟
                    return self._stats_cache[cache_key]

            # 过滤指标
            current_time = time.time()
            filtered_metrics = []

            for metric in self._metrics:
                if metric.metric_type != metric_type:
                    continue

                if tags and not all(metric.tags.get(k) == v for k, v in tags.items()):
                    continue

                if time_window and current_time - metric.timestamp > time_window:
                    continue

                filtered_metrics.append(metric)

            if not filtered_metrics:
                return None

            # 计算统计信息
            values = [m.value for m in filtered_metrics]
            stats = PerformanceStats(
                metric_type=metric_type,
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=sum(values) / len(values),
                sum_value=sum(values),
                unit=filtered_metrics[0].unit,
                tags=tags or {},
                metadata={'cache_time': time.time()}
            )

            # 缓存结果
            self._stats_cache[cache_key] = stats

            return stats

    def get_recent_metrics(
        self,
        metric_type: Optional[str] = None,
        limit: int = 100,
        time_window: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """
        获取最近的性能指标

        Args:
            metric_type: 指标类型过滤
            limit: 返回数量限制
            time_window: 时间窗口（秒）

        Returns:
            性能指标列表
        """
        with self._lock:
            current_time = time.time()
            metrics = []

            # 从最新开始遍历
            for metric in reversed(self._metrics):
                if metric_type and metric.metric_type != metric_type:
                    continue

                if time_window and current_time - metric.timestamp > time_window:
                    continue

                metrics.append(metric)
                if len(metrics) >= limit:
                    break

            return metrics

    def get_all_metric_types(self) -> List[str]:
        """获取所有指标类型"""
        with self._lock:
            return list(set(m.metric_type for m in self._metrics))

    def get_metrics_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        获取指标摘要

        Args:
            time_window: 时间窗口（秒）

        Returns:
            摘要信息字典
        """
        metric_types = self.get_all_metric_types()
        summary = {
            "time_window": time_window,
            "total_metrics": len(self._metrics),
            "metric_types": len(metric_types),
            "metrics": {}
        }

        for metric_type in metric_types:
            stats = self.get_stats(metric_type, time_window=time_window)
            if stats:
                summary["metrics"][metric_type] = {
                    "count": stats.count,
                    "avg": round(stats.avg_value, 2),
                    "min": stats.min_value,
                    "max": stats.max_value,
                    "unit": stats.unit
                }

        return summary

    def clear_metrics(self, metric_type: Optional[str] = None, older_than: Optional[int] = None):
        """
        清理性能指标

        Args:
            metric_type: 指标类型，None表示全部
            older_than: 清理早于该时间（秒）的指标
        """
        with self._lock:
            if metric_type is None and older_than is None:
                # 清理所有指标
                self._metrics.clear()
                self._stats_cache.clear()
                self.logger.info("已清理所有性能指标")
                return

            current_time = time.time()
            original_count = len(self._metrics)

            # 过滤保留的指标
            filtered_metrics = []
            for metric in self._metrics:
                should_keep = True

                if metric_type and metric.metric_type == metric_type:
                    should_keep = False

                if older_than and current_time - metric.timestamp > older_than:
                    should_keep = False

                if should_keep:
                    filtered_metrics.append(metric)

            # 更新指标列表
            self._metrics = deque(filtered_metrics, maxlen=self.max_history)

            # 清理相关缓存
            cache_keys_to_remove = []
            for cache_key in self._stats_cache.keys():
                if metric_type and cache_key.startswith(f"{metric_type}:"):
                    cache_keys_to_remove.append(cache_key)

            for cache_key in cache_keys_to_remove:
                self._stats_cache.pop(cache_key, None)

            cleaned_count = original_count - len(self._metrics)
            self.logger.info(f"已清理 {cleaned_count} 条性能指标")

    def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                # 清理1小时前的指标
                self.clear_metrics(older_than=3600)
                time.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"清理性能指标失败: {e}")
                time.sleep(60)  # 出错时等待1分钟再重试

    def export_metrics(self, filepath: str, time_window: Optional[int] = None):
        """
        导出性能指标到文件

        Args:
            filepath: 文件路径
            time_window: 时间窗口（秒）
        """
        try:
            metrics = self.get_recent_metrics(time_window=time_window, limit=10000)
            data = {
                "export_time": datetime.now().isoformat(),
                "time_window": time_window,
                "count": len(metrics),
                "metrics": [
                    {
                        "timestamp": m.timestamp,
                        "datetime": datetime.fromtimestamp(m.timestamp).isoformat(),
                        "type": m.metric_type,
                        "value": m.value,
                        "unit": m.unit,
                        "tags": m.tags,
                        "metadata": m.metadata
                    }
                    for m in metrics
                ]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"性能指标已导出到: {filepath}")

        except Exception as e:
            self.logger.error(f"导出性能指标失败: {e}")


# 全局性能监控器实例
_global_monitor = None


def get_global_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def record_metric(metric_type: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
    """便捷函数：记录性能指标"""
    monitor = get_global_monitor()
    monitor.record_metric(metric_type, value, unit, tags)


def monitor_function(metric_type: str, tags: Optional[Dict[str, str]] = None):
    """装饰器：监控函数执行性能"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            return monitor.record_function_execution(metric_type, func, *args, tags=tags, **kwargs)
        return wrapper
    return decorator


def monitor_async_function(metric_type: str, tags: Optional[Dict[str, str]] = None):
    """装饰器：监控异步函数执行性能"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            return await monitor.record_async_function_execution(metric_type, func, *args, tags=tags, **kwargs)
        return wrapper
    return decorator