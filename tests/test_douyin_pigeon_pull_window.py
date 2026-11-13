# -*- coding: utf-8 -*-
"""v3 get_user_message 请求窗口回归。

真机实测（2026-11-13 23:46）：买家 23:45:58 / 23:46:02 发的消息在会话列表里
能查到，时间线接口却一条都拿不到。原因是请求体固定回溯 24h，而服务端按时间
升序返回且有条数上限——返回的永远是窗口里最早的那批，最新消息被截掉。基线
因此定在 23:33:25，之后每轮拉到的还是同一批旧消息。

窗口贴着客户端游标走之后，基线落在真正的最新一条上；一旦单页仍被截断，
就用本页最新时间戳做下一页起点，追到窗口末端为止。
"""
from __future__ import annotations

import time
import unittest

from tools.reverse_probe.douyin.pigeon_http import PigeonHttp

NOW_US = int(time.time() * 1e6)


def _msg(ts):
    return {"ts": ts, "text": "x", "sender_biz_role": "Buyer"}


def _make_http(pages):
    """按页返回预置消息，同时记录每次请求的时间窗口。"""
    http = PigeonHttp.__new__(PigeonHttp)
    http.session = None
    http.last_status = 200
    http.last_error = ""
    http.windows = []
    queue = list(pages)

    def build_body(start_us, end_us):
        http.windows.append((start_us, end_us))
        return b"body"

    def decode(_resp):
        if queue:
            return queue.pop(0)
        return []

    http._build_body = build_body
    http._post = lambda _body: b"resp"
    http._decode_response = decode
    return http


class TestPullWindow(unittest.TestCase):
    def test_window_starts_one_minute_before_cursor(self):
        since = NOW_US - 60 * 10 ** 6
        http = _make_http([[_msg(since + 10 ** 6)]])

        http.pull_messages(since)

        start, end = http.windows[0]
        self.assertEqual(start, since - 60 * 10 ** 6)
        self.assertGreater(end, since)

    def test_baseline_window_covers_only_recent_minutes(self):
        http = _make_http([[]])

        http.pull_messages(0)

        start, _end = http.windows[0]
        self.assertLessEqual(start, NOW_US - 500 * 10 ** 6)
        self.assertGreater(start, NOW_US - 700 * 10 ** 6)

    def test_truncated_page_continues_from_its_newest(self):
        since = NOW_US - 30 * 60 * 10 ** 6
        old_page = [_msg(since + (i + 1) * 10 ** 6) for i in range(50)]
        new_page = [_msg(NOW_US - 10 ** 6)]
        http = _make_http([old_page, new_page])

        messages = http.pull_messages(since, end_us=NOW_US)

        self.assertEqual(len(messages), 51)
        self.assertEqual(len(http.windows), 2)
        self.assertEqual(http.windows[1][0], old_page[-1]["ts"])

    def test_short_page_is_not_paged_again(self):
        since = NOW_US - 5 * 60 * 10 ** 6
        http = _make_http([[_msg(NOW_US - 10 ** 6)]])

        messages = http.pull_messages(since, end_us=NOW_US)

        self.assertEqual(len(messages), 1)
        self.assertEqual(len(http.windows), 1)


if __name__ == "__main__":
    unittest.main()
