from ..base_request import BaseRequest
from typing import Dict, Any, Optional
from utils.logger_loguru import get_logger
from Channel.douyin.utils.log_throttle import throttled_warning, throttled_info

logger = get_logger("SendMessage")


class SendMessage(BaseRequest):
    """抖音消息发送器（优先 pigeon 纯协议，回退浏览器桥）

    【传输层选择】
    1. pigeon 纯协议（推荐，与拼多多同构）：由
       tools/reverse_probe/douyin/pigeon_transport.PigeonTransport 维护持久 WS，
       会话凭据(token+pigeon_sign)从浏览器抓取一次复用（等同拼多多 cookie）。
       send_text 走 WS 帧 type=10018，已实证服务端 ACK code=0 且消息真实投递。
    2. 浏览器桥（feige_bridge.py）：当 pigeon 传输未初始化/未连接时自动回退，
       走 DOM（输入框+发送按钮），签名由页面自身完成。

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

    def _degraded_not_implemented(self, method_name: str) -> None:
        self.logger.warning(
            f"{method_name} 暂不支持: 降级返回空结果"
        )

    def send_text(self, recipient_uid, message_content):
        """
        发送文本消息。

        优先 pigeon 纯协议：recipient_uid 为会话 conversation_id（数值字符串），
        由 PigeonTransport 依据 conv_state 缓存回填 security_receiver_id/shop_id。
        若 pigeon 传输未就绪则回退浏览器桥（recipient_uid 为买家显示名）。
        阻塞到发送完成/超时，成功返回 {"success": True}。
        """
        recipient_uid = str(recipient_uid)
        message_content = str(message_content)

        # 1) pigeon 协议（首选）
        #    先走 HTTP 单次请求（与拼多多 send_message 同构），失败再走 WS 协议流。
        #    因此这里不再要求 WS 已连接：HTTP 不需要长连接。
        try:
            from tools.reverse_probe.douyin.pigeon_transport import PigeonTransport
            from Channel.douyin.message_rules import record_self_sent, clear_self_sent
            pt = PigeonTransport.instance(self._transport_key())
            if pt is not None:
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
                        f"发送文本消息成功: recipient={recipient_uid}"
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
                # 「发到了旧子会话」。short_id 与 talk_id 对不上时，基本就是
                # 会话被平台关闭后又重开、子会话 ID 已经换了一轮。
                extra = ""
                if detail.get("code") is not None:
                    extra = ", code=%s, msg=%s" % (detail.get("code"), detail.get("msg") or "")
                elif detail.get("detail"):
                    extra = ", detail=%s" % detail.get("detail")
                if detail.get("short_id"):
                    extra += ", short_id=%s, talk_id=%s" % (
                        detail.get("short_id"), detail.get("talk_id") or ""
                    )
                if detail.get("ack_code") is not None:
                    extra += ", ack_code=%s, ack_msg=%s" % (
                        detail.get("ack_code"), detail.get("ack_msg") or ""
                    )
                self.logger.error(
                    f"发送文本消息失败: recipient={recipient_uid}, reason={reason}{extra}"
                )
                return None
        except Exception as exc:
            self.logger.error(
                f"发送文本消息异常: error_type={type(exc).__name__}, "
                f"transport=pigeon"
            )

        # 2) 浏览器桥（回退）
        try:
            from Channel.douyin.feige_bridge import FeigeBridge
            bridge = FeigeBridge.instance()
            if not bridge.is_ready:
                self.logger.error(
                    f"发送文本消息失败: 浏览器桥未就绪, recipient={recipient_uid}"
                )
                return None
            ok = bridge.send_text(recipient_uid, message_content)
            if ok:
                try:
                    from Channel.douyin.message_rules import record_self_sent
                    record_self_sent(self.shop_id, recipient_uid, message_content)
                except Exception:
                    pass
                return {"success": True, "recipient_uid": recipient_uid, "transport": "bridge"}
            self.logger.error(
                f"发送文本消息失败: recipient={recipient_uid}, transport=bridge"
            )
            return None
        except Exception as exc:
            self.logger.error(
                f"发送文本消息异常: error_type={type(exc).__name__}, "
                f"transport=bridge"
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
