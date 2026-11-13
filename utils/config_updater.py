"""
配置更新工具

需求：账号认证成功后，自动将完整 UID 追加到 config.json：
- transfer.main_account_user_ids：所有账号的完整 UID
  （主账号为纯数字如 '661962391'，子账号为 'cs_{shop_id}_{user_id}'）
- transfer.sub_account_uids：子账号 UID 列表（以 cs_ 开头），
  供 _is_main_account 判断子账号（子账号不调用转人工 API，仅静默标记 + 通知）

- 若字段不存在则创建，UID 已存在则跳过（去重）
- 保留 config.json 中其他所有字段（llm、notification、ai_reply 等）

说明：这里使用项目已有的 Config 单例（config.set）而非直接写文件，
确保内存缓存与磁盘同步——move_conversation._is_main_account 通过
get_config 读取该列表，只有缓存更新后规则才能立即生效。
"""
import threading

from config import config, get_config
from utils.logger_loguru import get_logger

logger = get_logger("ConfigUpdater")

# 写入锁，防止多个登录并发写入造成数据竞争
_write_lock = threading.Lock()


def update_config_with_uid(uid: str) -> bool:
    """将账号完整 UID 写入 config.json（去重，保留其他配置字段）

    完整 UID 格式：主账号为纯数字（如 '661962391'），
    子账号为 'cs_{shop_id}_{user_id}'（如 'cs_661962391_189109418'）。
    以 'cs_' 开头的 UID 同时追加到 transfer.sub_account_uids，
    供子账号判断（不调用转人工 API，仅静默标记 + 通知）。

    Args:
        uid: 账号完整 UID，如 '661962391' 或 'cs_661962391_189109418'

    Returns:
        bool: 写入成功返回 True；参数为空或写入异常返回 False
    """
    if not uid:
        logger.warning("UID 为空，跳过写入 config.json")
        return False

    with _write_lock:
        try:
            uid_str = str(uid)
            changed = False

            # 1. 追加到所有账号列表（去重）
            main_uids = list(get_config("transfer.main_account_user_ids", []) or [])
            if uid_str not in main_uids:
                main_uids.append(uid_str)
                config.set("transfer.main_account_user_ids", main_uids, save=False)
                changed = True

            # 2. 子账号（cs_ 开头）同时追加到子账号列表（去重）
            if uid_str.startswith("cs_"):
                sub_uids = list(get_config("transfer.sub_account_uids", []) or [])
                if uid_str not in sub_uids:
                    sub_uids.append(uid_str)
                    config.set("transfer.sub_account_uids", sub_uids, save=False)
                    changed = True

            if changed:
                # 一次原子写入，保留其他所有配置字段
                config.save()
                logger.info(
                    f"UID {uid_str} 已写入 config.json"
                    f"（sub_account={uid_str.startswith('cs_')}）"
                )
            else:
                logger.debug(f"UID {uid_str} 已存在，跳过写入")
            return True
        except Exception as e:
            logger.error(f"写入 UID {uid} 到 config.json 失败: {e}")
            return False
