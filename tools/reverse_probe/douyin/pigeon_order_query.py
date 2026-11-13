# -*- coding: utf-8 -*-
"""IM 页面订单接口探针：确认 `/backstage/cmpoent/order/query` 的请求体形态。

背景
----
订单查询已改为「IM 接口优先、抖店后台兜底」。IM 接口是飞鸽工作台的内部接口，
没有对外文档，抓包只留下响应（`{"check_spam_resp": ..., "order_info": {...}}`），
请求体字段只能按响应结构反推。代码里内置了四种候选形态（见
`Channel/douyin/utils/API/order_query.py` 的 `IM_BODY_STYLES`），
本脚本用真实登录态逐个试，告诉你哪一种被服务端接受。

用法
----
    # 用会话里出现过的订单卡片自动取单号
    python tools/reverse_probe/douyin/pigeon_order_query.py

    # 指定单号 / 指定会话
    python tools/reverse_probe/douyin/pigeon_order_query.py --order-id 1234567890123456789
    python tools/reverse_probe/douyin/pigeon_order_query.py --conv-id <parentConShortId>

    # 连后台接口一起对比（确认兜底路径是否可用）
    python tools/reverse_probe/douyin/pigeon_order_query.py --order-id xxx --with-backend

输出每一步的 HTTP 状态、返回 code/msg、是否含 order_info；命中后把
order_info 的字段名与关键值打印出来（手机号/地址做脱敏）。
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Channel.douyin.utils.API.order_query import (  # noqa: E402
    IM_BODY_STYLES,
    PIGEON_ORDER_QUERY_URL,
    _IM_HEADERS,
    OrderQuery,
    _cookie_header,
    collect_session_order_ids,
    load_session_cookies,
)


def _mask(value, keep=4):
    text = str(value or "")
    if len(text) <= keep:
        return text
    return text[:keep] + "*" * max(0, len(text) - keep - 2) + text[-2:]


def _post(style_name, build, order_id, conv_id, cookies, timeout):
    import requests

    body = build(order_id, conv_id)
    headers = dict(_IM_HEADERS)
    headers["cookie"] = _cookie_header(cookies)
    try:
        resp = requests.post(
            PIGEON_ORDER_QUERY_URL, json=body, headers=headers, timeout=timeout
        )
    except Exception as exc:
        return {"style": style_name, "body": body, "error": type(exc).__name__}

    result = {"style": style_name, "body": body, "status": resp.status_code}
    try:
        payload = resp.json()
    except ValueError:
        result["error"] = "not_json"
        result["text"] = resp.text[:200]
        return result

    result["code"] = payload.get("code")
    result["msg"] = str(payload.get("msg") or payload.get("message") or "")[:120]
    info = payload.get("order_info")
    if not isinstance(info, dict) or not info:
        data = payload.get("data")
        if isinstance(data, dict):
            info = data.get("order_info")
    if isinstance(info, dict) and info:
        result["order_info"] = info
    return result


def main():
    parser = argparse.ArgumentParser(description="IM 订单接口请求体探针")
    parser.add_argument("--order-id", default="", help="订单号；缺省从会话卡片里取")
    parser.add_argument("--conv-id", default="", help="会话 ID（parentConShortId）")
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--shop-id", default="", help="账号的 shop_id（定位登录态文件用）")
    parser.add_argument("--user-id", default="", help="账号的 user_id（定位登录态文件用）")
    parser.add_argument("--name", default="", help="登录名（定位登录态文件用）")
    parser.add_argument("--with-backend", action="store_true", help="同时试抖店后台接口")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON（含 order_info）")
    args = parser.parse_args()

    cookies = load_session_cookies()
    if not cookies:
        print("没有读到飞鸽登录态（dy_session_*.json），请先在界面里登录抖音账号。")
        return 2

    order_id = args.order_id.strip()
    conv_id = args.conv_id.strip()
    if not order_id:
        if not conv_id:
            print("未指定 --order-id / --conv-id，尝试从会话列表里找一个订单卡片…")
            try:
                from Channel.douyin.session_paths import existing_pigeon_session_path
                from tools.reverse_probe.douyin.pigeon_conv_list import PigeonConvList

                session_path = existing_pigeon_session_path(
                    args.shop_id, args.user_id, args.name
                )
                for conv in PigeonConvList(str(session_path)).fetch_conversations():
                    cid = str(conv.get("parentConShortId") or "")
                    ids = collect_session_order_ids(cid)
                    if ids:
                        conv_id, order_id = cid, ids[0]
                        break
            except Exception as exc:
                print(f"会话列表读取失败: {type(exc).__name__}: {exc}")
        else:
            ids = collect_session_order_ids(conv_id)
            if ids:
                order_id = ids[0]
        if not order_id:
            print("会话里没有找到订单号，请用 --order-id 指定。")
            return 2

    print(f"接口: {PIGEON_ORDER_QUERY_URL}")
    print(f"订单号: {order_id}  会话: {conv_id or '(未指定)'}")
    print("-" * 78)

    hit = None
    for name, build in IM_BODY_STYLES:
        result = _post(name, build, order_id, conv_id, cookies, args.timeout)
        head = f"[{name:<13}] body={json.dumps(result['body'], ensure_ascii=False)}"
        if "error" in result and "status" not in result:
            print(f"{head} -> 请求失败: {result['error']}")
            continue
        info = result.pop("order_info", None)
        print(
            f"{head}\n{'':16}-> http={result.get('status')} code={result.get('code')!r} "
            f"msg={result.get('msg')!r} order_info={'有' if info else '无'}"
        )
        if info:
            hit = (name, info)
            break

    if not hit:
        print("-" * 78)
        print("四种请求体都没有拿到 order_info。")
        print("请在 IM 页面按 F12 → Network，点开任意一条 cmpoent/order/query 请求，")
        print("把 Payload 贴出来，按它替换 IM_BODY_STYLES 即可（只需保留命中那一种）。")
        return 1

    style_name, info = hit
    print("-" * 78)
    print(f"命中请求体形态: {style_name}")
    print("order_info 字段:")
    for key in sorted(info):
        value = info[key]
        if key in ("mobile", "post_receiver", "receiver"):
            value = _mask(value)
        if key in ("full_address", "post_addr"):
            value = "（已省略）" if key == "post_addr" else str(value)[:28] + "…"
        print(f"  {key:<24} = {value!r}")

    order = OrderQuery._normalize_im_order(info)
    print("-" * 78)
    print("归一化结果（上层工具实际拿到的字段）:")
    for key in (
        "order_sn", "order_status_desc", "goods_name", "quantity",
        "pay_amount", "create_time", "logistics_company", "tracking_no",
    ):
        print(f"  {key:<20} = {order.get(key)!r}")

    if args.with_backend:
        print("-" * 78)
        query = OrderQuery()
        row = query._search_orders({"order_id": order_id}, limit=1)
        if row is None:
            print(f"后台接口不可用: {query.last_search_error}")
        elif not row:
            print("后台接口可用，但没有查到该订单")
        else:
            print("后台接口可用，字段数:", len(row[0]))

    if args.json:
        print("-" * 78)
        print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
