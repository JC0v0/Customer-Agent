"""Robust shop-logo loading shared by the account UI and auto-reply UI.

The original code raised ``ValueError`` whenever ``QPixmap.loadFromData`` could
not decode the bytes, and the catch block logged only the exception *type*
(``error_type=ValueError``) — dropping the message — so the real cause was
impossible to see.  It also emitted an empty pixmap, which the UI rendered as
the text "加载失败".

This module centralises decoding + circular cropping + a neutral placeholder so
that:

* every failure path records the *full* reason at DEBUG level (no WARNING/ERROR
  noise by default, but the cause is still discoverable when verbosity is raised);
* a benign, non-null placeholder is returned instead of an empty pixmap, so the
  UI never shows "加载失败" and never logs a warning for an unavailable logo.
"""

from __future__ import annotations

import io
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QImageReader, QPainter, QPainterPath, QPixmap

from utils.logger_loguru import get_logger
from utils.safe_image_fetch import fetch_image

_logger = get_logger("LogoHelper")

_DEFAULT_BG = "#e8eaed"  # neutral gray, same tone as the card "加载中" box


def decode_image_bytes(data: bytes) -> Optional[QPixmap]:
    """Decode raw image bytes into a QPixmap using several fallbacks.

    Tries, in order:
      1. ``QPixmap.loadFromData`` (auto format detection).
      2. ``QImageReader`` with no explicit format (broader auto detection).
      3. Pillow: reopen as a PIL Image, normalise to RGBA, re-export as PNG
         bytes and decode those (covers odd BMP/WebP variants Qt may skip).

    Returns ``None`` only when every path fails.
    """
    if not data:
        return None

    pixmap = QPixmap()
    if pixmap.loadFromData(data):
        return pixmap

    reader = QImageReader()
    reader.setDevice(__bytes_io(data))
    image = reader.read()
    if not image.isNull():
        return QPixmap.fromImage(image)

    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as pil_image:
            pil_image = pil_image.convert("RGBA")
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            fallback = QPixmap()
            if fallback.loadFromData(buf.getvalue()):
                return fallback
    except Exception:  # noqa: BLE001 - Pillow is best-effort only
        pass

    return None


def make_circular(pixmap: QPixmap, size: int = 60) -> QPixmap:
    """Return a circular crop of ``pixmap`` fitted into ``size`` x ``size``."""
    circular = QPixmap(size, size)
    circular.fill(Qt.GlobalColor.transparent)

    painter = QPainter(circular)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)
        scaled = pixmap.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(0, 0, scaled)
    finally:
        painter.end()
    return circular


def default_placeholder(size: int = 60, letter: str = "") -> QPixmap:
    """A neutral, never-null circular placeholder (gray, optional initial)."""
    placeholder = QPixmap(size, size)
    placeholder.fill(Qt.GlobalColor.transparent)

    painter = QPainter(placeholder)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)
        painter.fillPath(path, _gray_brush(size))
        if letter:
            painter.setPen(Qt.GlobalColor.white)
            font = painter.font()
            font.setBold(True)
            font.setPointSize(max(14, size // 3))
            painter.setFont(font)
            painter.drawText(placeholder.rect(), Qt.AlignmentFlag.AlignCenter, letter[0].upper())
    finally:
        painter.end()
    return placeholder


def _gray_brush(size: int):
    from PyQt6.QtGui import QBrush, QColor

    return QBrush(QColor(_DEFAULT_BG))


def __bytes_io(data: bytes):
    return io.BytesIO(data)


def load_logo(url: str, size: int = 60, letter: str = "") -> QPixmap:
    """Fetch + decode + crop a shop logo, always returning a non-null pixmap.

    Any failure (network, SSRF gate, decode) is recorded at DEBUG level with the
    real exception message and the URL, then a neutral placeholder is returned.
    This guarantees the UI never shows "加载失败" and the app never emits a
    WARNING/ERROR for a merely unavailable logo.
    """
    if not url:
        return default_placeholder(size, letter)

    try:
        image_data = fetch_image(url)
        pixmap = decode_image_bytes(image_data)
        if pixmap is None:
            _logger.debug(
                f"Logo 解码失败（已用占位图）: url={url} len={len(image_data)}"
            )
            return default_placeholder(size, letter)
        return make_circular(pixmap, size)
    except Exception as exc:  # noqa: BLE001 - defensive: logo is non-critical
        _logger.debug(
            f"Logo 加载失败（已用占位图）: url={url} "
            f"error_type={type(exc).__name__} detail={exc}"
        )
        return default_placeholder(size, letter)
