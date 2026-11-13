# 抖店订单接口说明

本文记录抖店订单查询用到的接口契约。**所有标注「实测」的结论均来自
2026-11-12 真实订单验证**（用真实账号下单后走浏览器 cookie 直调）。
`Channel/douyin/utils/API/order_query.py` 按本文实现。

## 1. 接口总览

| 用途 | 方法 | 端点 | 状态 |
|---|---|---|---|
| 订单查询（IM 主路径） | POST | `https://pigeon.jinritemai.com/backstage/cmpoent/order/query` | **2026-11-13 抓包确认页面在用** |
| 订单查询（后台兜底） | GET | `https://fxg.jinritemai.com/api/order/searchlist` | **2026-11-12 实测可用** |
| 订单详情（飞鸽域旧接口） | GET | `https://pigeon.jinritemai.com/backstage/getOrder` | 实测被风控，弃用 |
| 按订单号反查买家 | GET | `https://pigeon.jinritemai.com/backstage/getuserbyorder` | 实测缺版本头，弃用 |

查询顺序：**IM 接口 → 后台 searchlist → 会话里的订单号骨架**。

之所以把 IM 接口放第一位：它和消息通道共用飞鸽 cookie（按账号隔离的
`dy_session_<hash>.json` 里的 `cookies_full`，早期是全局 `dy_session_live.json`），
买家一进线就有凭据可用；而 searchlist 要的是
`fxg.jinritemai.com` 域的会话，登录时没打开过抖店后台就拿不到。

## 2. IM 订单接口（主路径，2026-11-13 抓包）

```
POST https://pigeon.jinritemai.com/backstage/cmpoent/order/query
```

- 路径里的 `cmpoent` 是**服务端真实拼写**（不是 component），改了会 404。
- 请求头要带 IM 页面的 referer：`https://im.jinritemai.com/pc_seller_v2/main/workspace`。
  用 fxg 的 referer 会被按跨站请求拒掉。
- 鉴权：飞鸽域 cookie，与消息接收/发送同一份登录态。

响应（真实抓包，金额单位分）：

```json
{"check_spam_resp": {...},
 "order_info": {
   "order_id": "69194...", "order_status": 3, "order_status_desc": "已发货",
   "order_type_desc": "普通订单", "create_time": 1794303361, "pay_time": 1794303400,
   "actual_pay_amount": 12980, "seller_price": 12980, "pay_type_desc": "支付宝",
   "full_address": "贵州省 贵阳市 …", "mobile": "138****5678",
   "buyer_words": "", "order_remark": "",
   "logistics_info": {"tracking_no": "...", "express_company": "申通快递"},
   "sku_order_list": [{"product_id": "...", "product_name": "...", "count": 2}]}}
```

请求体：该接口是页面内部接口，没有对外文档，抓包只留下了响应。
实现里按候选形态依次试探（`IM_BODY_STYLES`）：`{"order_id"}` →
`{"order_id","biz_type":4}` → `{"data":{"order_id"}}` → `{"order_id","cid"}`，
命中后记住形态，不再重复试探；四种都失败则冷却 10 分钟并退回后台接口。
若确认了服务端真实请求体，把 `IM_BODY_STYLES` 收敛成那一种即可。

## 3. searchlist（后台兜底，实测通过）

```
GET https://fxg.jinritemai.com/api/order/searchlist
```

请求头：`referer: https://fxg.jinritemai.com/ffa/morder/order/list`。
鉴权靠 fxg 域 cookie。

### 实测通过的参数

| 参数 | 说明 |
|---|---|
| `page` | 页码，从 `0` 开始 |
| `pageSize` | 每页数量（实测 20 可用） |
| `order_by` / `order` | `create_time` / `desc` |
| `tab` | `all` |
| `order_id` | **按订单号精确查询**（实测返回匹配的 1 条） |
| `buyer` | **按买家真实昵称精确筛选**（语义见 §3.1，掩码昵称查不到） |
| `source` | 固定 `shop_order_view_upgrade` |
| `appid` / `_bid` / `aid` | 固定 `1` / `ffa_order` / `4272` |
| `verifyFp` / `fp` / `msToken` / `a_bogus` | 页面请求会带，但实测可省略 |

### 响应外壳（实测）

```json
{"st": 0, "msg": "", "code": 0, "data": [订单行...], "total": 1, "page": 0, "size": 20}
```

注意 `st` 是**数字 0**（飞鸽 pigeon 域接口是字符串 `"0"`，两者不同）。

### 订单行字段（真实订单实测）

| 字段 | 含义 |
|---|---|
| `shop_order_id` | 订单号。**`order_id` 字段实测为空串，勿用** |
| `order_status` | 状态码 int（实测 2=待发货） |
| `order_status_info.order_status_text` | 状态文案（"待发货"） |
| `order_status_info.dead_line_time` | 发货截止时间戳（秒） |
| `pay_amount` / `actual_pay_amount` / `post_amount` / `total_price` | 金额，**单位分** |
| `create_time_str` | 已格式化的下单时间 |
| `create_time` / `pay_time` / `exp_ship_time` | 秒级时间戳 |
| `user_nickname` | 买家昵称（平台脱敏后） |
| `buyer_words` / `remark` | 买家备注 / 商家备注 |
| `pay_type_desc` | 支付方式（"抖音支付"） |
| `receiver_info.post_receiver` / `post_tel` | 收件人 / 电话（脱敏） |
| `receiver_info.post_addr` | `{province, city, town, street}.name` + `detail` |
| `latest_logistic_info.tracking_no` / `company_code` | 物流单号 / 承运商（未发货为空） |
| `product_item[]` | 商品子单数组，见下 |

`product_item[]` 每项（实测）：

| 字段 | 含义 |
|---|---|
| `item_order_id` | 子单号 |
| `product_id` / `product_name` / `img` | 商品 |
| `sku_spec` | `[{name, value}]` 规格，如 颜色分类/尺码大小 |
| `combo_num` / `combo_amount` / `pay_amount` | 数量 / 金额（分） |
| `sku_order_status_info.order_status_text` | 子单状态文案 |

## 4. 飞鸽域其它订单接口（均不可用，勿再尝试）

| 接口 | 实测响应 | 结论 |
|---|---|---|
| `/backstage/getOrder` | `{"code":"10001010A","msg":"当前环境存在风险，请稍后重试"}` | 字节风控签名体系（a_bogus），有头/无头浏览器、context.request、页内 fetch 三种方式全部被拦。**注意它与 §2 的 `cmpoent/order/query` 不是同一个接口**，后者是 IM 页面自己在用的，不受此限制 |
| `/backstage/getuserbyorder` | `{"code":11001,"msg":"飞鸽版本过低，浏览器用户请刷新页面"}` | 服务端校验飞鸽客户端版本标识，纯 HTTP 无法绕过 |

客服场景的「按买家查订单」改用 searchlist 的 `buyer` 参数实现（见 §3.1）。

### 4.1 会话买家 ↔ 订单的关联链路（2026-11-12 深挖实测）

结论：**只有「真实昵称」这一条关联键是通的**，其余全部实测否定。

| 尝试 | 实测结果 | 结论 |
|---|---|---|
| `buyer=<真实昵称>`（如「不喜欢说话」） | total=1，精确返回该买家订单 | **可用**。链路：会话侧真实昵称 → `buyer=` → 订单 |
| `buyer=<掩码昵称>`（如「不*」） | total=0 | 订单行的 `user_nickname` 是展示层掩码，不能反查 |
| `buyer=<昵称前缀>`（如「不」） | total=0 | 服务端**精确匹配**，不支持模糊 |
| `user_id=` / `buyer_id=` / `uid=` | `st=10000` 无效参数 | searchlist 无按 ID 查询入口 |
| 订单行 `user_id` 做关联键 | 两次请求值不同（仅前 24 位盐值稳定） | **逐请求加密，不是稳定 ID**，不能做关联键 |
| `/backstage/getuserbyorder?oid=<订单号>` | 仍 `11001` 飞鸽版本过低 | 参数名已从 JS 确认为 `oid`，但服务端还校验版本标识，不可用 |
| `/backstage/getuserinfo`（POST `{data:{uids}}`） | `st=0`，纯 cookie 可调 | 飞鸽域买家资料接口可用，但 uid 传飞鸽数字 uid 返回的是卖家自身资料，买家 uid 的正确格式待真实会话验证 |

**会话侧真实昵称字段已实测确认**（2026-11-12 真实会话，买家发「转人工」）：

- 买家发言消息 `messageBody.ext`：
  - `ext/nickname` = 真实昵称（如「不喜欢说话」，仅买家消息有此字段）
  - `ext/uname` = 真实昵称（买家消息）；机器人/系统消息为占位值「用户」
- **补充关联键（卡片消息）**：机器人「订单确认信息」卡片
  `messageBody.ext/bury_order_id` = 买家真实订单号
  （实测 6919514772943875116），`ext/req_id` 里也带同一订单号
  （`..._ACK_ORDER_ADDRESS_<订单号>`）。即：**会话里出现过的订单卡片
  可直接拿订单号精确查单**，比昵称反查更可靠。
- 加密 UID 交叉验证：会话 `security_sender_id` 与订单 `user_id`
  前 24 位盐值一致（`AQDP453ixAhMC5VYiBNwrlgo`），证实是同一买家；
  但后半段逐请求不同，仍不可做关联键。

实现提供两个辅助函数（`Channel/douyin/utils/API/order_query.py`）：
`extract_buyer_nick_from_ext(ext)` 从消息 ext 提取真实昵称（排除占位值），
`extract_card_order_ids(message_body)` 从卡片消息提取订单号。

### 4.2 物流字段（schema 实测，内容待发货订单验证）

订单行的物流字段（待发货订单实测，全部为空）：

| 字段 | 含义 |
|---|---|
| `logistics_status` / `logistics_time` / `ship_time` | 发货状态 / 时间 |
| `shipped_count` / `wait_ship_count` / `logistics_package_count` | 包裹计数 |
| `latest_logistic_info.tracking_no` / `company_code` | 运单号 / 承运商 |
| `latest_logistic_info.track` / `lo_status_text` / `track_info` | 轨迹 / 状态文案 |
| `latest_logistic_info.deliver_name` / `deliver_mobile` | 派件员信息 |

字段 schema 已确认存在，但**有真实物流数据时的内容结构未验证**
（验证用的测试订单未发货）。轨迹明细如果 `track` 字段不够用，
可能还有独立的物流详情接口，未探查。

## 5. 鉴权与 cookie 要求

- IM 接口（§2）用飞鸽域 cookie，即登录态文件里的 `cookies_full`，与消息通道同一份。
  登录态文件自 2026-11 起**按账号隔离**（`temp/dy_session_<hash>.json`，见
  `Channel/douyin/session_paths.py`），订单模块统一用 `existing_pigeon_session_path()`
  解析，不要写死文件名。
- searchlist（§3）需要 `fxg.jinritemai.com` 域的登录 cookie（抖店商家后台）。
- 登录流程（`douyin_login.py`）抓取的是**全浏览器 cookie**（`page.context.cookies()`），
  若登录时抖店后台也在同一浏览器登录过，cookies_full 天然包含 fxg 域会话。
- 若抖店后台未登录，searchlist 返回未登录（`st != 0`），实现走消息历史骨架兜底。

## 6. 已知限制

1. **金额单位是分**，实现里统一除以 100 转成元。
2. `buyer` 参数按**真实昵称精确匹配**：昵称改过、或买家改用新昵称后查不到；
   不同买家昵称撞名时**不能**直接返回订单详情。当前实现只把昵称反查当作
   候选集合，必须再由买家提供订单号并在候选集合内精确命中；没有订单号时
   只提示买家提供订单号，避免串单泄漏。
3. ~~会话列表响应中「买家真实昵称」的 API 字段名未确认~~
   **已确认**（2026-11-12 真实会话实测）：买家消息 `ext/nickname` /
   `ext/uname` 为真实昵称，机器人消息 `uname` 是占位值「用户」，需过滤。
4. `order_status` 状态码枚举只实测了 2=待发货，其余按通用枚举推断，
   文案缺失时兜底映射可能不准。
5. 物流字段 schema 已确认，但有货在途数据时的内容结构未验证。
6. 售后状态、优惠明细字段（`promotion_detail` 等）未接入。

## 7. 回退策略

真实接口失败（未登录 / 风控 / 网络 / 订单不存在）时，回退到会话消息历史里
正则提取订单号（15-25 位数字），返回只含订单号的骨架，`source` 标为
`message_history`，其余字段标「待核实」。这样买家问订单时至少能拿到订单号，
不会完全无响应。

补充：`extract_card_order_ids()` 已预留卡片订单号解析，但当前接收管线只
持久化买家文本消息；订单卡片若要作为主路径，需要先把卡片原文写入会话历史。
现阶段用买家消息里出现过的订单号作为安全候选来源。
