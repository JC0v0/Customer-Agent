"""Douyin runtime credential paths.

Keep the pigeon session file scoped to one account.  The old global
``dy_session_live.json`` made a second Douyin account overwrite the first
account's token/sign as soon as it logged in.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Optional

from utils.runtime_path import get_temp_path


def pigeon_session_path(
    profile_scope: Optional[str] = None,
    name: Optional[str] = None,
) -> Path:
    """Return the account-scoped pigeon session path."""
    scope = str(profile_scope or name or "account")
    digest = sha256(scope.encode("utf-8")).hexdigest()[:16]
    return get_temp_path(f"dy_session_{digest}.json")


def legacy_pigeon_session_path() -> Path:
    """Old single-account path kept for a one-time migration fallback."""
    return get_temp_path("dy_session_live.json")


def pigeon_capture_path(
    profile_scope: Optional[str] = None,
    name: Optional[str] = None,
) -> Path:
    """Return the account-scoped WS frame-capture path.

    登录流程把浏览器里发出的 pigeon WS 帧录下来存到这里，作为发送/注册
    帧模板。注册帧（10011）里带着店铺归属，和会话凭据一样是按账号隔离
    的——两个抖音账号先后登录时，各自的录帧文件互不覆盖。
    """
    scope = str(profile_scope or name or "account")
    digest = sha256(scope.encode("utf-8")).hexdigest()[:16]
    return get_temp_path(f"dy_send_capture_{digest}.json")


def legacy_pigeon_capture_path() -> Path:
    """Old global capture path kept as a read-only migration fallback."""
    return get_temp_path("dy_send_capture.json")


def existing_pigeon_capture_path(
    shop_id: Optional[str] = None,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Path:
    """Prefer the account-scoped capture, then the username one, then legacy."""
    candidates = []
    if shop_id and user_id:
        candidates.append(pigeon_capture_path(f"douyin:{shop_id}:{user_id}"))
    if name:
        candidates.append(pigeon_capture_path(name=name))
    candidates.append(legacy_pigeon_capture_path())
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def existing_pigeon_session_path(
    shop_id: Optional[str] = None,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Path:
    """Prefer the account-scoped file, then the username file, then legacy."""
    candidates = []
    if shop_id and user_id:
        candidates.append(pigeon_session_path(f"douyin:{shop_id}:{user_id}"))
    if name:
        candidates.append(pigeon_session_path(name=name))
    candidates.append(legacy_pigeon_session_path())
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


__all__ = [
    "existing_pigeon_capture_path",
    "existing_pigeon_session_path",
    "legacy_pigeon_capture_path",
    "legacy_pigeon_session_path",
    "pigeon_capture_path",
    "pigeon_session_path",
]
