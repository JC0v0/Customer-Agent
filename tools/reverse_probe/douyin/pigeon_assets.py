"""协议骨架资源：随包分发、不含凭据的 pigeon 模板。

为什么需要这个模块
------------------
pigeon 通道发送帧必须复用真实抓包的结构，但原始抓包里带 token、pigeon_sign、
device_id、店铺号，既不能入库也不能随程序分发。``strip_capture.py`` 把这些值
清空、只留字段骨架，产物放在 ``Channel/douyin/assets/`` 下一并打包。

运行时有两份来源，优先级从高到低：

1. ``temp/dy_send_capture*.json``：登录时由 Playwright 监听 WebSocket 自动录下来
   的真实帧序列。含当前账号的注册帧（店铺归属靠它），是正常路径。
2. 包内骨架：登录录帧失败、或老版本升级上来还没重新登录时的兜底。
   发送序列（10016 开会话、10018 正文）都在骨架里——登录页面不会发消息，
   这两个帧永远录不到，只能靠骨架。握手和注册帧必须来自真实抓取，
   凭空构造的店铺标识服务端不认。

HTTP 请求体（``get_user_message``）不含店铺信息，骨架里只有字段结构和
token/device/sign 的空位，全店通用，直接走骨架即可。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.runtime_path import get_resource_path

# 相对项目根目录的资源位置，打包时按同样路径放进 _MEIPASS
SEND_SKELETON_REL = "Channel/douyin/assets/pigeon_send_frame.json"
START_SKELETON_REL = "Channel/douyin/assets/pigeon_start_frame.json"
ENTER_SKELETON_REL = "Channel/douyin/assets/pigeon_enter_frame.json"
USERMSG_SKELETON_REL = "Channel/douyin/assets/pigeon_usermsg_body.json"

# 注册帧命令号。pigeon 协议里 10011 之前是握手/认证，之后（10013 起）才是
# 业务帧，所以「命令号 < 10011」就是握手帧。
#
# 不要写死握手帧号——客户端版本一变就换号，2026 年三次实测各不相同：
#   9 月旧协议 10008→10009、headless 新连接 10010→10011、
#   11-13 真机 GUI 登录 10005→10006。写死白名单的结果就是每次改版都
#   把用户挡在「缺少抓包文件」门外。
REGISTER_CMD = 10011

# pigeon 连接地址特征。工作台页面同时会连 frontier.snssdk.com，那条连接
# 的命令号也落在 10000 段，录帧不按 URL 过滤就会把它的帧混进来。
PIGEON_WS_HOST_MARK = "ws.fxg.jinritemai.com"


def is_handshake_cmd(cmd) -> bool:
    """命令号小于注册帧的即为握手/认证帧（10005/10006、10008/10009、10010）。"""
    try:
        return int(cmd) < REGISTER_CMD
    except (TypeError, ValueError):
        return False


class PigeonTemplateError(RuntimeError):
    """通道模板缺失或不完整。

    单独定义而不是抛 KeyError/FileNotFoundError，是为了让上层能把
    「需要重新登录采集模板」这个可操作结论直接告诉用户，而不是退避重试。
    """


def send_skeleton_path() -> Path:
    return get_resource_path(SEND_SKELETON_REL)


def start_skeleton_path() -> Path:
    return get_resource_path(START_SKELETON_REL)


def enter_skeleton_path() -> Path:
    return get_resource_path(ENTER_SKELETON_REL)


def usermsg_skeleton_path() -> Path:
    return get_resource_path(USERMSG_SKELETON_REL)


def _load_frame_skeleton(path: Path) -> Optional[bytes]:
    """读取单帧骨架文件（``{desc, cmd, b64}``），缺失或损坏返回 None。"""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        b64 = data.get("b64")
        if not b64:
            return None
        return base64.b64decode(b64)
    except (OSError, ValueError):
        return None


def load_send_skeleton_frame() -> Optional[bytes]:
    """返回脱敏后的 10018 发送帧；骨架缺失或损坏时返回 None。"""
    return _load_frame_skeleton(send_skeleton_path())


def load_start_skeleton_frame() -> Optional[bytes]:
    """返回脱敏后的 10016 开会话帧；骨架缺失或损坏时返回 None。

    登录录帧只录握手/注册（登录页面不会发消息），发送序列里的 10013~10021
    永远录不到。10018 早有骨架兜底，10016 之前没有，新客户登录后第一次
    回复就报 ``missing_start_template``——文件能连、能收消息、就是发不出去。
    """
    return _load_frame_skeleton(start_skeleton_path())


def load_enter_skeleton_frame() -> Optional[bytes]:
    """返回脱敏后的 10014 进会话帧；缺失或损坏时返回 None。

    进会话帧的回包带「目标会话的票据 + 会话短 ID」，文本发送必须先拿到它。
    登录录帧只会录握手和注册，进会话帧永远录不到，只能靠这份骨架。
    """
    return _load_frame_skeleton(enter_skeleton_path())


def load_usermsg_skeleton() -> Optional[Dict[str, Any]]:
    """返回脱敏后的 HTTP 请求体模板字典（url_base + body_b64）。"""
    path = usermsg_skeleton_path()
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not data.get("body_b64"):
        return None
    return data


def read_capture_frames(template_path: Optional[str]) -> List[bytes]:
    """读取运行时抓包文件里的 ws_sent 二进制帧。

    文件缺失、格式不对、单帧损坏都只跳过该帧，不抛异常：抓包文件是
    登录流程写的，可能被截断（用户中途关窗口），不该让整个通道起不来。
    """
    if not template_path:
        return []
    path = Path(template_path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []

    frames: List[bytes] = []
    for item in data.get("ws_sent", []):
        b64 = item.get("b64")
        if not b64:
            continue
        try:
            frames.append(base64.b64decode(b64))
        except (ValueError, TypeError):
            continue
    return frames


__all__ = [
    "PIGEON_WS_HOST_MARK",
    "PigeonTemplateError",
    "REGISTER_CMD",
    "SEND_SKELETON_REL",
    "START_SKELETON_REL",
    "USERMSG_SKELETON_REL",
    "is_handshake_cmd",
    "load_send_skeleton_frame",
    "load_start_skeleton_frame",
    "load_usermsg_skeleton",
    "read_capture_frames",
    "send_skeleton_path",
    "start_skeleton_path",
    "usermsg_skeleton_path",
]
