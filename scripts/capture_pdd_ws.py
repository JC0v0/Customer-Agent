"""
拼多多客服 WebSocket 实时抓包脚本
================================

用途
----
复用项目已有的 Playwright 持久化登录态（user_data/），在你自己的浏览器会话里
挂 CDP（Chrome DevTools Protocol）监听器，把 wss://m-ws.pinduoduo.com/ 的
所有帧原样落盘成 JSONL，用于回答两个问题：

  1. 服务端到底会推送哪些类型的消息（下行 response / message.type 全集）
  2. 官方前端有没有往 WS 里发业务报文（上行 frameSent 是否只有 ping）

第 2 点是当前 Channel/pinduoduo 实现没有覆盖的空白：现有代码只做接收，
发消息全部走 HTTP POST。抓一次真实会话就能确认 WS 上行能不能用。

用法
----
    # 列出数据库里可用的账号
    python scripts/capture_pdd_ws.py --list

    # 抓 5 分钟（默认账号 = 数据库里第一个）
    python scripts/capture_pdd_ws.py --duration 300

    # 指定账号（shop_id:user_id，来自 --list 的输出）
    python scripts/capture_pdd_ws.py --account 123456:789 --duration 600

    # 概览分析（方法分布、消息类型统计）
    python scripts/capture_pdd_ws.py --analyze temp/pdd_ws_20260920_153000.jsonl

    # 解码成可读 JSON（逐帧 + 消息列表），见 scripts/decode_pdd_ws.py
    python scripts/decode_pdd_ws.py temp/pdd_ws_20260920_153000.jsonl

注意
----
- 浏览器窗口会打开，需要你保持登录态；如果 user_data 里的会话过期，
  手动在弹出的窗口里登录一次即可，之后会被持久化。
- 默认对 access_token / Cookie 做脱敏（--no-redact 可关闭）。
  抓包文件里仍可能含买家昵称等信息，分享前请自行检查。
- Ctrl+C 可随时停止并正常落盘。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 让脚本能从 scripts/ 目录直接运行：把项目根目录加进 sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 初始化顺序必须和 app.py 一致：config -> DI 容器 -> 其余业务模块。
# 少了 configure_standard_services()，db_manager 代理会抛「服务未注册」。
from config import config as _app_config  # noqa: E402
from core.di_container import configure_standard_services  # noqa: E402

configure_standard_services(_app_config)

# 注意：pdd_login 在 import 阶段就会设置 PLAYWRIGHT_BROWSERS_PATH，
# 所以必须在它之后再 import playwright 相关内容。
from Channel.pinduoduo.pdd_login import PDDLogin  # noqa: E402
from database import db_manager  # noqa: E402

# 商家客服工作台——这个页面才会建立 wss://m-ws.pinduoduo.com 连接
CHAT_URL = "https://mms.pinduoduo.com/chat-merchant/index.html"

# WebSocket 帧 opcode（RFC 6455）。区分业务帧和协议控制帧的关键。
OPCODE_NAMES = {0: "continuation", 1: "text", 2: "binary", 8: "close", 9: "ping", 10: "pong"}

# 需要监听的 CDP 事件 -> 我们自己的简短事件名
CDP_EVENTS = {
    "Network.webSocketCreated": "created",
    "Network.webSocketWillSendHandshakeRequest": "handshakeRequest",
    "Network.webSocketHandshakeResponseReceived": "handshakeResponse",
    "Network.webSocketFrameSent": "frameSent",
    "Network.webSocketFrameReceived": "frameReceived",
    "Network.webSocketFrameError": "frameError",
    "Network.webSocketClosed": "closed",
}


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------
def redact(text: str) -> str:
    """把 access_token 和 Cookie 值替换掉，避免抓包文件泄露凭证。"""
    if not isinstance(text, str):
        return text
    text = re.sub(r"(access_token=)[^&\s\"']+", r"\1<REDACTED>", text)
    text = re.sub(r"(\"[Cc]ookie\"\s*:\s*\")[^\"]*", r"\1<REDACTED>", text)
    return text


# ---------------------------------------------------------------------------
# 账号解析
# ---------------------------------------------------------------------------
def list_accounts() -> list[dict]:
    """从数据库读出所有账号（含店铺信息）。"""
    try:
        return db_manager.get_all_accounts_with_details() or []
    except Exception as exc:
        print(f"[!] 读取账号失败: {type(exc).__name__}: {exc}")
        return []


def pick_account(selector: str | None) -> dict | None:
    """按 'shop_id:user_id' 选账号；不传则取第一个。"""
    accounts = [a for a in list_accounts() if a.get("channel_name") == "pinduoduo"]
    if not accounts:
        print("[!] 数据库里没有拼多多账号，请先在应用里添加并登录一次。")
        return None

    if not selector:
        return accounts[0]

    for acc in accounts:
        if f"{acc.get('shop_id')}:{acc.get('user_id')}" == selector:
            return acc
    print(f"[!] 未找到账号 {selector}，可用列表：")
    for acc in accounts:
        print(f"    {acc.get('shop_id')}:{acc.get('user_id')}  {acc.get('shop_name')}")
    return None


# ---------------------------------------------------------------------------
# 抓包主流程
# ---------------------------------------------------------------------------
class WSRecorder:
    """把 CDP 的 WebSocket 事件写成 JSONL，并维护实时统计。"""

    def __init__(self, out_path: Path, max_payload: int, do_redact: bool):
        self.out_path = out_path
        self.max_payload = max_payload
        self.do_redact = do_redact
        self.fp = out_path.open("w", encoding="utf-8")
        self.counter: Counter = Counter()
        self.urls: dict[str, str] = {}   # requestId -> url
        self.total = 0
        # WS 是否已真正建立。会话过期时页面会停在登录页，永远不会置 True，
        # 主流程据此决定何时开始倒计时，避免把抓包时长浪费在人工登录上。
        self.ws_established = False

    def record(self, event_name: str, params: dict) -> None:
        req_id = params.get("requestId", "")

        # webSocketCreated 时记住 URL，后续帧事件只有 requestId
        if event_name == "created":
            self.urls[req_id] = params.get("url", "")

        url = self.urls.get(req_id, "")

        # 只关心拼多多这条连接，其它 WS（埋点等）跳过
        if url and "pinduoduo" not in url:
            return

        if event_name in ("created", "handshakeResponse") and "pinduoduo" in url:
            if not self.ws_established:
                print(f"  [✓] WebSocket 已建立: {url[:110]}")
            self.ws_established = True

        row: dict = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event_name,
            "requestId": req_id,
            "url": url,
        }

        # 帧事件：取出 opcode 和 payload
        frame = params.get("response") or {}
        if event_name in ("frameSent", "frameReceived"):
            opcode = frame.get("opcode")
            payload = frame.get("payloadData", "")
            row["opcode"] = opcode
            row["opcodeName"] = OPCODE_NAMES.get(opcode, str(opcode))
            row["len"] = len(payload)
            if len(payload) > self.max_payload:
                payload = payload[: self.max_payload] + f"...<truncated {len(payload)}>"
            row["payload"] = payload
            self.counter[f"{event_name}:{row['opcodeName']}"] += 1
        elif event_name == "handshakeRequest":
            row["headers"] = params.get("request", {}).get("headers", {})
            self.counter[event_name] += 1
        elif event_name == "handshakeResponse":
            row["status"] = frame.get("status")
            row["headers"] = frame.get("headers", {})
            self.counter[event_name] += 1
        else:
            if event_name == "frameError":
                row["error"] = params.get("errorMessage")
            self.counter[event_name] += 1

        line = json.dumps(row, ensure_ascii=False)
        if self.do_redact:
            line = redact(line)
        self.fp.write(line + "\n")
        self.fp.flush()   # 立即落盘，Ctrl+C 也不丢数据
        self.total += 1

        # 上行业务帧是本次调查的重点，实时打出来
        if event_name == "frameSent" and row.get("opcode") == 1:
            print(f"  [↑ 上行业务帧] {row['payload'][:200]}")

    def close(self) -> None:
        self.fp.close()

    def summary(self) -> str:
        lines = [f"共记录 {self.total} 条事件 -> {self.out_path}"]
        for key, count in sorted(self.counter.items()):
            lines.append(f"  {key:<34} {count}")
        return "\n".join(lines)


async def attach_cdp(context, page, recorder: WSRecorder) -> None:
    """给单个 page 挂 CDP 监听器。"""
    try:
        client = await context.new_cdp_session(page)
        await client.send("Network.enable")
        for cdp_event, short_name in CDP_EVENTS.items():
            # 默认参数绑定 short_name，避免闭包late-binding 拿到最后一个值
            client.on(
                cdp_event,
                lambda params, _n=short_name: recorder.record(_n, params),
            )
    except Exception as exc:
        print(f"[!] CDP 挂载失败（该页跳过）: {type(exc).__name__}: {exc}")


async def capture(args) -> int:
    account = pick_account(args.account)
    if not account:
        return 1

    shop_id = account.get("shop_id")
    user_id = account.get("user_id")
    username = account.get("username") or ""
    # 与 refresh_pdd_cookies 使用同一套 profile 作用域，直接复用已登录状态
    profile_scope = f"pinduoduo:{shop_id}:{user_id}"

    print(f"[*] 账号: {account.get('shop_name')} ({shop_id}:{user_id})")

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "temp" / f"pdd_ws_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recorder = WSRecorder(out_path, args.max_payload, not args.no_redact)

    login = PDDLogin(name=username, password="", profile_scope=profile_scope)
    user_data_dir = str(login._profile_dir())
    print(f"[*] 浏览器 profile: {user_data_dir}")
    if not Path(user_data_dir).exists():
        print("[!] 该账号还没有持久化登录数据，窗口打开后请手动登录一次。")

    from playwright.async_api import async_playwright

    playwright = None
    context = None
    try:
        playwright = await async_playwright().start()
        # 复用项目自己的启动逻辑：Chrome/Edge 探测 + CDP 回退 + profile 锁清理
        context = await login._launch_context(playwright, user_data_dir, headless=False)

        # 新开的页面也要挂上监听（客服工作台可能弹窗口）
        async def on_page(new_page):
            await attach_cdp(context, new_page, recorder)

        context.on("page", lambda p: asyncio.create_task(on_page(p)))

        # 持久化上下文启动时可能已有一个空白页，复用它
        pages = context.pages
        page = pages[0] if pages else await context.new_page()
        await attach_cdp(context, page, recorder)

        print(f"[*] 打开客服工作台: {args.url}")
        try:
            await page.goto(args.url, timeout=60000)
        except Exception as exc:
            print(f"[!] 页面加载异常（可继续抓包）: {type(exc).__name__}")

        # ---- 阶段一：等待 WS 建立（会话过期时需要人工在窗口里登录）----
        print(f"[*] 等待 WebSocket 建立，最长 {args.login_timeout}s")
        print("    若窗口停在登录页，请手动完成登录（滑块/短信验证也在此完成）")
        waited = 0
        while not recorder.ws_established and waited < args.login_timeout:
            await asyncio.sleep(2)
            waited += 2
            if waited % 30 == 0:
                print(f"    ...已等待 {waited}s，仍未建立 WS")

        if not recorder.ws_established:
            print("[!] 超时仍未建立 WebSocket —— 多半是没登录成功，本次没有可用数据")
        else:
            # ---- 阶段二：正式抓包计时 ----
            print(f"\n[*] 开始抓包，持续 {args.duration}s（Ctrl+C 可提前结束）")
            print("[*] 请用买家小号发文本/图片/点商品咨询/问售后，覆盖更多消息类型\n")
            try:
                await asyncio.sleep(args.duration)
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl+C，正在收尾...")
    except Exception as exc:
        print(f"[!] 抓包过程出错: {type(exc).__name__}: {exc}")
    finally:
        recorder.close()
        if context:
            try:
                await context.close()
            except Exception:
                pass
        if playwright:
            try:
                await playwright.stop()
            except Exception:
                pass

    print("\n" + recorder.summary())
    print(f"\n[*] 分析: python scripts/capture_pdd_ws.py --analyze {out_path}")
    return 0


# ---------------------------------------------------------------------------
# 离线分析
# ---------------------------------------------------------------------------
def _decode_binary(payload: str):
    """CDP 的 binary 帧 payloadData 是 base64 编码的。

    这份样本的 titan-ws 二进制帧疑似使用 protobuf，并含 gzip 特征，尚未验证 schema。其
    部分字符串值会以明文留在字节流里，提取可打印片段可辅助判断这一帧
    在做什么（例如 titan.notifyDataLite.ack）。

    Returns: (原始字节, 可读标识符列表, 是否包含 gzip 段)
    """
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return b"", [], False
    idents = [m.decode("ascii", "ignore") for m in re.findall(rb"[ -~]{6,}", raw)]
    return raw, idents, (b"\x1f\x8b\x08" in raw)


def _conn_label(url: str) -> str:
    """把 URL 归类到短名，便于按连接分组统计。"""
    if "m-ws" in url:
        return "m-ws (聊天)"
    if "titan" in url:
        return "titan-ws (通知)"
    return url.split("/")[2] if "//" in url else (url or "<未知>")


def analyze(path: Path) -> int:
    """统计抓包文件。

    关键：WebSocket opcode 1=text / 2=binary 都是**数据帧**，只有
    8/9/10 (close/ping/pong) 才是协议控制帧。早期版本把 binary 误判成
    控制帧，会得出「上行没有业务报文」的错误结论。
    """
    if not path.exists():
        print(f"[!] 文件不存在: {path}")
        return 1

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # requestId -> url：帧事件本身只带 requestId，URL 只在 created 时出现
    url_of: dict[str, str] = {}
    for r in rows:
        if r.get("url"):
            url_of[r.get("requestId", "")] = r["url"]

    def conn(r) -> str:
        return _conn_label(url_of.get(r.get("requestId", ""), ""))

    print(f"=== {path.name} 共 {len(rows)} 条事件 ===\n")

    # ---- 1. 连接清单 ----
    print("-- 连接 --")
    seen = set()
    for r in rows:
        if r["event"] == "created" and r.get("url") and r["url"] not in seen:
            seen.add(r["url"])
            print(f"  {_conn_label(r['url']):<18} {r['url'][:120]}")
        elif r["event"] == "handshakeResponse":
            print(f"  {'':<18} └─ HTTP {r.get('status')}")

    # ---- 2. 按连接 × 方向 × 帧类型 ----
    CONTROL = {"close", "ping", "pong"}
    grid: Counter = Counter()
    for r in rows:
        if r["event"] in ("frameSent", "frameReceived"):
            direction = "上行" if r["event"] == "frameSent" else "下行"
            grid[(conn(r), direction, r.get("opcodeName", "?"))] += 1

    print("\n-- 帧分布（连接 × 方向 × 类型）--")
    for (c, d, op), n in sorted(grid.items()):
        kind = "控制帧" if op in CONTROL else "数据帧"
        print(f"  {c:<18} {d}  {op:<8} {n:<5} {kind}")

    # ---- 3. 上行数据帧：判断 WS 能否双向承载业务 ----
    sent_data = [
        r for r in rows
        if r["event"] == "frameSent" and r.get("opcodeName") not in CONTROL
    ]
    print(f"\n-- 上行数据帧 {len(sent_data)} 条 --")
    if not sent_data:
        print("  无：本次抓包中前端没有通过 WS 发送任何业务报文")
    else:
        per_conn: defaultdict = defaultdict(Counter)
        for r in sent_data:
            if r.get("opcodeName") == "text":
                try:
                    o = json.loads(r.get("payload", ""))
                    cmd = (o.get("cmd") or o.get("data", {}).get("cmd")) if isinstance(o, dict) else None
                except json.JSONDecodeError:
                    cmd = "<非JSON>"
                per_conn[conn(r)][f"text cmd={cmd or '<无>'}"] += 1
            else:
                _raw, idents, gz = _decode_binary(r.get("payload", ""))
                tag = next((s for s in idents if "." in s and " " not in s), "<无可读标识>")
                per_conn[conn(r)][f"binary {tag}" + (" +gzip" if gz else "")] += 1
        for c, counter in per_conn.items():
            print(f"  [{c}]")
            for k, n in counter.most_common():
                print(f"    {k:<44} {n}")

    # ---- 4. 下行文本帧（m-ws 聊天协议）----
    recv_text = [
        r for r in rows
        if r["event"] == "frameReceived" and r.get("opcodeName") == "text"
    ]
    print(f"\n-- 下行文本帧 {len(recv_text)} 条 --")
    resp_types: Counter = Counter()
    msg_types: defaultdict = defaultdict(Counter)
    for r in recv_text:
        try:
            obj = json.loads(r.get("payload", ""))
        except json.JSONDecodeError:
            resp_types["<非JSON>"] += 1
            continue
        if not isinstance(obj, dict):
            continue
        resp = obj.get("response", "<无response>")
        resp_types[resp] += 1
        if resp == "push":
            msg = obj.get("message", {})
            if isinstance(msg, dict):
                key = f"type={msg.get('type')}"
                if msg.get("sub_type") is not None:
                    key += f" sub_type={msg.get('sub_type')}"
                msg_types[resp][key] += 1

    if resp_types:
        print("  response 字段分布：")
        for k, v in resp_types.most_common():
            print(f"    {k:<24} {v}")
    else:
        print("  无")

    if msg_types.get("push"):
        print("\n  push 下的 message.type 分布：")
        # 对照 Channel/pinduoduo/pdd_message.py 的 PDDMsgType / PDDSubType
        known = {
            "0": "TEXT", "1": "IMAGE", "5": "EMOTION", "14": "VIDEO",
            "24": "TRANSFER", "64": "GOODS_SPEC", "1002": "WITHDRAW",
        }
        sub_known = {"0": "GOODS_INQUIRY", "1": "ORDER_INFO"}
        for k, v in msg_types["push"].most_common():
            parts = dict(p.split("=", 1) for p in k.split())
            t = parts.get("type", "")
            sub = parts.get("sub_type")
            if t == "0" and sub is not None:
                tag = sub_known.get(sub, "TEXT (未知 sub_type)")
            else:
                tag = known.get(t, "!! 未在 PDDMsgType 中定义")
            print(f"    {k:<28} {v:<5} {tag}")
    else:
        print("\n  [!] 本样本没有观察到文本 push，不能据此判定没有买家消息，")
        print("      还需对照操作时间及二进制通知，确认聊天消息的实际传输路径。")

    # ---- 5. 版本号漂移检查 ----
    for r in rows:
        u = r.get("url", "")
        if "m-ws" in u and "version=" in u:
            live = u.split("version=")[1].split("&")[0]
            print("\n-- 版本号 --")
            print(f"  线上前端 version = {live}")
            print("  代码 API_VERSION  = 202506091557  (Channel/pinduoduo/pdd_channel.py:63)")
            if live != "202506091557":
                print("  [!] 与样本前端版本不一致；尚未验证旧值是否失效，不应仅据此修改。")
            break

    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="拼多多客服 WebSocket 抓包 / 分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="列出数据库里的拼多多账号")
    parser.add_argument("--account", help="指定账号，格式 shop_id:user_id")
    parser.add_argument("--out", help="输出 JSONL 路径（默认 temp/pdd_ws_<时间>.jsonl）")
    parser.add_argument("--duration", type=int, default=300, help="抓包时长（秒），默认 300")
    parser.add_argument("--login-timeout", type=int, default=600,
                        help="等待 WS 建立/人工登录的最长秒数，默认 600")
    parser.add_argument("--url", default=CHAT_URL, help="要打开的页面")
    parser.add_argument("--max-payload", type=int, default=4000, help="单帧 payload 截断长度")
    parser.add_argument("--no-redact", action="store_true", help="不脱敏 access_token/Cookie")
    parser.add_argument("--analyze", help="分析已有的抓包文件，不启动浏览器")
    args = parser.parse_args()

    if args.analyze:
        return analyze(Path(args.analyze))

    if args.list:
        accounts = [a for a in list_accounts() if a.get("channel_name") == "pinduoduo"]
        if not accounts:
            print("（无）")
            return 0
        print(f"{'shop_id:user_id':<28} {'店铺':<24} 登录名")
        for a in accounts:
            key = f"{a.get('shop_id')}:{a.get('user_id')}"
            print(f"{key:<28} {str(a.get('shop_name')):<24} {a.get('username')}")
        return 0

    # Windows 上 Playwright 需要 Proactor 事件循环（Python 3.8+ 默认已是）
    try:
        return asyncio.run(capture(args))
    except KeyboardInterrupt:
        print("\n[*] 已中断")
        return 0


if __name__ == "__main__":
    sys.exit(main())
