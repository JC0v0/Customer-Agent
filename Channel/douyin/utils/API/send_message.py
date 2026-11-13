from ..base_request import BaseRequest
from typing import Dict, Any, Optional
from utils.logger_loguru import get_logger
from Channel.douyin.utils.log_throttle import throttled_warning, throttled_info

logger = get_logger("SendMessage")


class SendMessage(BaseRequest):
    """抖音消息发送器（pigeon 协议优先，页面通道兜底）

    【传输层选择】
    1. pigeon 纯协议（首选）：先发一帧「进会话」拿回本会话票据（服务端下发，
       约 11 分钟有效），再把正文帧发出去，几百毫秒投递完成，不需要浏览器
       常驻。2026-11-14 真机 + HTTP 时间线双重验证通过。
    2. 浏览器页面（兜底）：协议未连接、票据拿不到、或服务端静默丢弃时，
       交给飞鸽页面（点会话 → 填输入框 → 点发送），页面自己带票据。
       见 Channel/douyin/feige_bridge.py。

    - send_image / send_mallGoodsCard / move_conversation：三级降级
      （pigeon→bridge→degraded），始终返回 dict 绝不返回 None，协议待逆向。
    - getAssignCsList：已真实化，调 backstage/onlineservice 获取在线客服列表，
      失败时回退 currentuser 取当前客服，返回非空 dict（绝不返回 None）。
    """

    # 飞鸽后台：在线客服列表（抓包实证返回 data=list[{id,shop_id,toutiao_id,service_name,online_status}]）
    DOUYIN_ONLINE_SERVICE_URL = (
        "https://pigeon.jinritemai.com/backstage/onlineservice?busy=1&biz_type=4&_pms=1"
    )
    # 飞鸽后台：当前登录用户/店铺信息（与 get_shop_info/get_user_info 共用端点）
    DOUYIN_CURRENT_USER_URL = (
        "https://pigeon.jinritemai.com/backstage/currentuser?biz_type=4&_pms=1"
    )

    def __init__(self, shop_id: str, user_id: str, channel_name: str = "douyin",
                 cookies=None):
        super().__init__(shop_id, user_id, channel_name)
        if cookies:
            self.update_cookies(cookies)

        # 检查账户信息是否正确加载
        if not hasattr(self, 'account_name'):
            self.logger.error(f"无法在数据库中找到账户: shop_id={shop_id}, user_id={user_id}")
            raise ValueError("找不到指定的账户信息")

    def _transport_key(self) -> str:
        from bridge.context import make_account_key
        return make_account_key(self.channel_name, self.shop_id, self.user_id)

    def _buyer_nickname(self, conv_id: str) -> str:
        """会话号 → 买家昵称（页面按昵称定位会话）。"""
        try:
            from tools.reverse_probe.douyin.pigeon_conv_list import PigeonConvList
            from Channel.douyin.session_paths import existing_pigeon_session_path
            from Channel.douyin.utils.API.order_query import extract_buyer_nick_from_ext

            # 会话文件按登录用户名落盘，必须带上它才能命中（见 session_paths）。
            session_path = existing_pigeon_session_path(
                self.shop_id, self.user_id, getattr(self, "account_name", "") or ""
            )
            convs = PigeonConvList(str(session_path)).fetch_conversations() or []
        except Exception as exc:
            self.logger.warning(
                f"会话列表拉取失败，无法定位买家昵称: conv={conv_id}, "
                f"error_type={type(exc).__name__}"
            )
            return ""
        for conv in convs:
            cid = str(conv.get("parentConShortId") or conv.get("subConversationId") or "")
            if cid != str(conv_id):
                continue
            for item in conv.get("msgList") or []:
                body = item.get("messageBody") or {}
                nick = extract_buyer_nick_from_ext(body.get("ext"))
                if nick:
                    return nick
        return ""

    def _send_by_page(self, conv_id: str, text: str, wait_seconds: float = 40) -> bool:
        """页面发送：点会话 → 填输入框 → 点发送，页面自己带票据。

        刚连上账号时浏览器还在加载，这里等它就绪而不是直接放弃：页面没起来
        时退回 pigeon 一定失败（票据拿不到），等一会儿反而能把消息发出去。
        """
        import time

        from Channel.douyin.feige_bridge import FeigeBridge
        from Channel.douyin.message_rules import record_self_sent

        bridge = FeigeBridge.instance(self._transport_key())
        if not bridge.is_send_ready:
            deadline = time.time() + wait_seconds
            while not bridge.is_send_ready and time.time() < deadline:
                if not bridge.is_running:
                    break
                time.sleep(0.5)
        if not bridge.is_send_ready:
            self.logger.warning(
                f"页面通道未就绪，发送跳过: conv={conv_id}, "
                f"running={getattr(bridge, 'is_running', None)}"
            )
            return False
        nick = self._buyer_nickname(conv_id)
        if not nick:
            self.logger.error(f"发送文本消息失败: 未取到买家昵称, conv={conv_id}")
            return False
        if not bridge.send_text(nick, text):
            self.logger.error(
                f"发送文本消息失败: recipient={conv_id}, nick={nick}, transport=page"
            )
            return False
        record_self_sent(self.shop_id, conv_id, text)
        return True

    def _degraded_not_implemented(self, method_name: str) -> None:
        self.logger.warning(
            f"{method_name} 暂不支持: 降级返回空结果"
        )

    def _send_by_protocol(self, recipient_uid: str, message_content: str):
        """pigeon 纯协议发送：进会话取票 → 发正文帧 → 等商家侧回声。

        2026-11-14 真机实证：进会话帧（10014/f610）只换会话号和买家键，
        服务端就回本会话的票据；正文帧带齐会话号/票据/会话短 ID/买家复合键
        即可投递，HTTP 时间线能查到原文。成功后无需浏览器。
        返回 dict 表示成功，None 表示这次没发出去（交给页面通道兜底）。
        """
        from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport
        from Channel.douyin.message_rules import record_self_sent, clear_self_sent

        pt = PigeonTransport.instance(self._transport_key())
        if pt is None:
            self.logger.info("协议通道未连接，转页面通道")
            return None
        try:
            # 预登记回声指纹（发送前，不是发送成功后）。
            #
            # 真机 2026-11-11 01:18:14 复现：服务端推送的商家侧回声比
            # send_text() 返回更早到达，拉取循环先读到了它，此时指纹还没
            # 登记，于是把自己刚发的第 2 条回复判成「人工客服介入」，
            # 整条会话静默转人工。发送前登记即可关掉这个竞态；
            # 发送失败时立刻撤销，避免同文本的真人消息被吞。
            record_self_sent(self.shop_id, recipient_uid, message_content)
            ok = pt.send_text(recipient_uid, message_content)
            if ok:
                self.logger.info(
                    f"发送文本消息成功: recipient={recipient_uid}, transport=pigeon"
                )
                return {
                    "success": True,
                    "recipient_uid": recipient_uid,
                    "transport": "pigeon",
                }
            clear_self_sent(self.shop_id, recipient_uid, message_content)
            detail = getattr(pt, "last_send_result", {}) or {}
            reason = detail.get("reason", "unknown")
            # 把服务端返回码和实际使用的会话短 ID 一并打出来：text_no_echo
            # 这类空洞结论看不出到底是「登录过期」「会话已关闭」还是
            # 「发到了旧子会话」。
            extra = ""
            if detail.get("code") is not None:
                extra = ", code=%s, msg=%s" % (detail.get("code"), detail.get("msg") or "")
            elif detail.get("detail"):
                extra = ", detail=%s" % detail.get("detail")
            if detail.get("talk_id"):
                extra += ", talk_id=%s" % detail.get("talk_id")
            if detail.get("ack_code") is not None:
                extra += ", ack_code=%s, ack_msg=%s" % (
                    detail.get("ack_code"), detail.get("ack_msg") or ""
                )
            self.logger.warning(
                f"协议发送未成功，转页面通道: recipient={recipient_uid}, "
                f"reason={reason}{extra}"
            )
            return None
        except Exception as exc:
            self.logger.error(
                f"协议发送异常，转页面通道: error_type={type(exc).__name__}"
            )
            return None

    def send_text(self, recipient_uid, message_content):
        """
        发送文本消息。

        优先 pigeon 纯协议：进会话帧换回本会话票据后直接发正文帧，几百毫秒
        就能投递，不依赖浏览器常驻。协议不通（未连接、票据拿不到、服务端静默
        丢弃）时退回页面通道，由页面自己带票据把消息发出去。
        阻塞到发送完成/超时，成功返回 {"success": True}。
        """
        recipient_uid = str(recipient_uid)
        message_content = str(message_content)

        # 1) pigeon 纯协议
        result = self._send_by_protocol(recipient_uid, message_content)
        if result:
            return result

        # 2) 页面通道（兜底）
        if self._send_by_page(recipient_uid, message_content):
            self.logger.info(
                f"发送文本消息成功: recipient={recipient_uid}, transport=page"
            )
            return {
                "success": True,
                "recipient_uid": recipient_uid,
                "transport": "page",
            }

        self.logger.error(
            f"发送文本消息失败: recipient={recipient_uid}, transport=all_failed"
        )
        return None

    # 图片上传端点（待逆向：飞鸽图片上传接口，通常返回 image_url/resource_id）
    IMAGE_UPLOAD_ENDPOINT = None
    # pigeon IM 图片消息帧类型（待逆向：非文本帧 10018，图片帧可能为 10020+）
    PIGEON_IMAGE_MSG_TYPE = None

    def send_image(self, recipient_uid, image_url, width=0, height=0):
        """发送图片消息。

        三级降级策略：
        1. PigeonTransport 纯协议路径（未来逆向后自动启用，hasattr 检测）
        2. FeigeBridge 浏览器桥路径（未来自动启用，hasattr 检测）
        3. 完善降级：返回明确 dict，绝不返回 None

        Args:
            recipient_uid: 接收者 uid
            image_url: 图片 URL（未来上传后返回的 URL）
            width: 图片宽度（预留，与拼多多签名对齐）
            height: 图片高度（预留，与拼多多签名对齐）

        Returns:
            dict: 始终返回 dict，含 success/transport/reason/detail 等字段
        """
        # 路径1：pigeon 纯协议（未来逆向后 PigeonTransport 添加 send_image 方法即自动启用）
        try:
            from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport
            pt = PigeonTransport.instance(self._transport_key())
            if pt is not None and hasattr(pt, "send_image") and callable(pt.send_image):
                result = pt.send_image(recipient_uid, image_url, width=width, height=height)
                if result:
                    return {"success": True, "transport": "pigeon", "result": result}
                if result is False:
                    detail = getattr(pt, "last_send_result", {}) or {}
                    return {
                        "success": False,
                        "transport": "pigeon",
                        "reason": detail.get("reason", "image_send_failed"),
                    }
        except Exception as e:
            self.logger.debug(
                f"图片消息发送跳过: error_type={type(e).__name__}, transport=pigeon"
            )

        # 路径2：浏览器桥（未来 FeigeBridge 添加 send_image 方法即自动启用）
        try:
            from Channel.douyin.feige_bridge import FeigeBridge
            bridge = FeigeBridge.instance()
            if bridge is not None and hasattr(bridge, "send_image") and callable(bridge.send_image):
                result = bridge.send_image(recipient_uid, image_url, width=width, height=height)
                if result is not None:
                    return {"success": True, "transport": "bridge", "result": result}
        except Exception as e:
            self.logger.debug(
                f"图片消息发送跳过: error_type={type(e).__name__}, transport=bridge"
            )

        # 路径3：完善降级（当前实际路径）
        self.logger.warning(
            f"图片消息发送失败: 暂不支持, recipient={recipient_uid}, "
            f"image_url={image_url}"
        )
        return {
            "success": False,
            "transport": "degraded",
            "reason": "image_protocol_not_reverse_engineered",
            "detail": "抖音图片发送协议尚未逆向（图片上传端点+pigeon图片消息帧类型），"
                      "当前无法发送图片。请在飞鸽工作台手动发送图片，或重新抓包获取图片上传接口后补全。",
            "recipient_uid": recipient_uid,
            "image_url": image_url,
        }

    # 商品卡发送端点（待逆向：飞鸽商品卡消息协议）
    MALL_GOODS_CARD_ENDPOINT = None
    # pigeon IM 商品卡消息帧类型（待逆向）
    PIGEON_GOODS_CARD_MSG_TYPE = None

    def send_mallGoodsCard(self, recipient_uid, goods_id, biz_type: int = 2):
        """发送商城商品卡片消息。

        三级降级策略（与 send_image 同构）：
        1. PigeonTransport 纯协议路径（未来逆向后自动启用，hasattr 检测）
        2. FeigeBridge 浏览器桥路径（未来自动启用，hasattr 检测）
        3. 完善降级：返回明确 dict，绝不返回 None

        Args:
            recipient_uid: 接收消息的用户UID
            goods_id: 商品ID
            biz_type: 业务类型

        Returns:
            dict: 始终返回 dict，含 success/transport/reason/detail 等字段
        """
        # 路径1：pigeon 纯协议
        try:
            from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport
            pt = PigeonTransport.instance(self._transport_key())
            if pt is not None and hasattr(pt, "send_mallGoodsCard") and callable(pt.send_mallGoodsCard):
                result = pt.send_mallGoodsCard(recipient_uid, goods_id, biz_type=biz_type)
                if result is not None:
                    return {"success": True, "transport": "pigeon", "result": result}
        except Exception as e:
            self.logger.debug(
                f"商品卡片发送跳过: error_type={type(e).__name__}, transport=pigeon"
            )

        # 路径2：浏览器桥
        try:
            from Channel.douyin.feige_bridge import FeigeBridge
            bridge = FeigeBridge.instance()
            if bridge is not None and hasattr(bridge, "send_mallGoodsCard") and callable(bridge.send_mallGoodsCard):
                result = bridge.send_mallGoodsCard(recipient_uid, goods_id, biz_type=biz_type)
                if result is not None:
                    return {"success": True, "transport": "bridge", "result": result}
        except Exception as e:
            self.logger.debug(
                f"商品卡片发送跳过: error_type={type(e).__name__}, transport=bridge"
            )

        # 路径3：完善降级
        self.logger.warning(
            f"商品卡片发送失败: 暂不支持, recipient={recipient_uid}, "
            f"goods_id={goods_id}"
        )
        return {
            "success": False,
            "transport": "degraded",
            "reason": "mall_goods_card_protocol_not_reverse_engineered",
            "detail": "抖音商品卡发送协议尚未逆向（商品卡消息帧类型+发送端点），"
                      "当前无法发送商品卡片。send_goods_link 工具已通过 ProductManager 校验商品有效性，"
                      "可引导买家在店铺商品页查看，或重新抓包获取商品卡协议后补全。",
            "recipient_uid": recipient_uid,
            "goods_id": goods_id,
            "biz_type": biz_type,
        }

    def getAssignCsList(self) -> Optional[Dict[str, Any]]:
        """
        获取分配的客服列表（真实化：backstage/onlineservice + 降级兜底）。

        返回格式与拼多多对齐：dict，key 为 CS UID（cs_{shop_id}_{user_id}），
        value 为含 username 的 dict。move_conversation 据此筛选可转接客服。

        降级策略（绝不返回 None）：
          1. 主路径：调 backstage/onlineservice?busy=1 获取在线客服列表
          2. 回退1：调 backstage/currentuser 取当前客服信息，构造单元素 dict
          3. 回退2：用构造入参构造单元素 dict（cs_{shop_id}_{user_id}）
        """
        # 1) 主路径：backstage/onlineservice 获取在线客服列表
        try:
            headers = {
                "user-agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
                ),
                "referer": "https://im.jinritemai.com/",
            }
            result = self.get(
                self.DOUYIN_ONLINE_SERVICE_URL,
                params=self._backstage_params(),
                headers=headers,
                timeout=10,
            )
            if (
                isinstance(result, dict)
                and result.get("code") == 0
                and isinstance(result.get("data"), list)
                and len(result["data"]) > 0
            ):
                cs_dict = {}
                for cs in result["data"]:
                    if not isinstance(cs, dict):
                        continue
                    toutiao_id = str(
                        cs.get("toutiao_id_str")
                        or cs.get("toutiao_id")
                        or cs.get("service_toutiao_id_str")
                        or cs.get("service_toutiao_id")
                        or ""
                    )
                    shop_id = str(cs.get("shop_id") or self.shop_id or "")
                    cs_name = cs.get("service_name") or cs.get("nickname") or toutiao_id or "未知客服"
                    if toutiao_id and shop_id:
                        cs_uid = f"cs_{shop_id}_{toutiao_id}"
                    elif toutiao_id:
                        cs_uid = f"cs_{toutiao_id}"
                    else:
                        continue
                    cs_dict[cs_uid] = {
                        "username": cs_name,
                        "toutiao_id": toutiao_id,
                        "shop_id": shop_id,
                        "online_status": cs.get("online_status", 1),
                    }
                if cs_dict:
                    logger.debug(
                        f"客服列表拉取成功: count={len(cs_dict)}, "
                        f"cs_uids={list(cs_dict.keys())}"
                    )
                    return cs_dict
                throttled_warning(logger, "cs.empty", "客服列表拉取为空，回退 currentuser")
        except Exception as e:
            throttled_warning(logger, "cs.exception",
                f"客服列表拉取异常: error_type={type(e).__name__}"
            )

        # 2) 回退1：backstage/currentuser 取当前客服信息
        try:
            headers = {
                "user-agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
                ),
                "referer": "https://im.jinritemai.com/",
            }
            result = self.get(
                self.DOUYIN_CURRENT_USER_URL,
                params=self._backstage_params(),
                headers=headers,
                timeout=10,
            )
            if (
                isinstance(result, dict)
                and result.get("code") == 0
                and isinstance(result.get("data"), dict)
            ):
                data = result["data"]
                csi = data.get("CustomerServiceInfo") or {}
                toutiao_id = str(
                    csi.get("id")
                    or data.get("ToutiaoId")
                    or data.get("SubToutiaoId")
                    or ""
                )
                shop_id = str(data.get("ShopId") or self.shop_id or "")
                cs_name = csi.get("screen_name") or data.get("ShopName") or "当前客服"
                if toutiao_id:
                    cs_uid = f"cs_{shop_id}_{toutiao_id}" if shop_id else f"cs_{toutiao_id}"
                    cs_dict = {cs_uid: {"username": cs_name, "toutiao_id": toutiao_id, "shop_id": shop_id}}
                    logger.debug(
                        f"客服列表降级成功: cs_uid={cs_uid}, cs_name={cs_name}"
                    )
                    return cs_dict
        except Exception as e:
            throttled_warning(logger, "cs.fallback_exception",
                f"客服列表降级异常: error_type={type(e).__name__}"
            )

        # 3) 回退2：用构造入参构造单元素 dict（绝不返回 None）
        shop_id = str(self.shop_id or "unknown")
        user_id = str(self.user_id or "unknown")
        cs_uid = f"cs_{shop_id}_{user_id}"
        cs_dict = {cs_uid: {"username": self.account_name or "当前客服"}}
        throttled_warning(logger, "cs.fallback",
            f"客服列表降级兜底: cs_uid={cs_uid}, "
            f"onlineservice/currentuser 均不可用"
        )
        return cs_dict

    # 会话转接端点（待逆向：飞鸽客服转接接口）
    MOVE_CONVERSATION_ENDPOINT = None

    def move_conversation(self, recipient_uid, cs_uid):
        """转移会话。

        三级降级策略（与 send_image/send_mallGoodsCard 同构）：
        1. PigeonTransport 纯协议路径（未来逆向后自动启用）
        2. FeigeBridge 浏览器桥路径（未来自动启用）
        3. 完善降级：返回明确 dict，绝不返回 None

        注意：工具层 transfer_conversation 对抖音已走「标记 handoff + 企业微信通知」
        降级路径，不调用此方法。此方法为 API 层对齐预留。

        Args:
            recipient_uid: 买家 uid
            cs_uid: 目标客服 uid

        Returns:
            dict: 始终返回 dict
        """
        # 路径1：pigeon 纯协议
        try:
            from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport
            pt = PigeonTransport.instance(self._transport_key())
            if pt is not None and hasattr(pt, "move_conversation") and callable(pt.move_conversation):
                result = pt.move_conversation(recipient_uid, cs_uid)
                if result is not None:
                    return {"success": True, "transport": "pigeon", "result": result}
        except Exception as e:
            self.logger.debug(
                f"转移会话跳过: error_type={type(e).__name__}, transport=pigeon"
            )

        # 路径2：浏览器桥
        try:
            from Channel.douyin.feige_bridge import FeigeBridge
            bridge = FeigeBridge.instance()
            if bridge is not None and hasattr(bridge, "move_conversation") and callable(bridge.move_conversation):
                result = bridge.move_conversation(recipient_uid, cs_uid)
                if result is not None:
                    return {"success": True, "transport": "bridge", "result": result}
        except Exception as e:
            self.logger.debug(
                f"转移会话跳过: error_type={type(e).__name__}, transport=bridge"
            )

        # 路径3：完善降级
        self.logger.warning(
            f"转移会话失败: 暂不支持, recipient={recipient_uid}, cs_uid={cs_uid}"
        )
        return {
            "success": False,
            "transport": "degraded",
            "reason": "move_conversation_protocol_not_reverse_engineered",
            "detail": "抖音会话转接协议尚未逆向。工具层 transfer_conversation 对抖音已走"
                      "「标记 handoff + 企业微信通知」降级路径，AI 不再抢答，人工客服可通过"
                      "企业微信通知及时介入。重新抓包获取转接接口后可补全真实转接。",
            "recipient_uid": recipient_uid,
            "cs_uid": cs_uid,
        }


__all__ = ['SendMessage']
