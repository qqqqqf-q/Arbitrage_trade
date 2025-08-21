import ccxt
import ccxt.pro as ccxtpro
import time
import logging
from datetime import datetime
import os
from decimal import (
    Decimal,
    getcontext,
    ROUND_DOWN,
    ROUND_UP,
    InvalidOperation as DecimalInvalidOperation,
)  # 显式导入异常
import math
import collections
import asyncio
import traceback  # 用于打印详细错误
import ctypes  # <--- 导入 ctypes
import json  # <--- 导入 json
import sys
import random
import collections  # 确保导入 collections
import concurrent.futures  # <--- 添加这一行

# --- 导入 CCXT 错误 ---
from ccxt.base.errors import (
    InsufficientFunds,
    InvalidOrder,
    OrderNotFound,
    NetworkError as CCXTNetworkError,
    ExchangeError as CCXTExchangeError,
    ArgumentsRequired,
    RateLimitExceeded,
    RequestTimeout,
    ExchangeNotAvailable,
)  # 导入更多特定错误
from ccxt import (
    AuthenticationError,
    NetworkError,
    RequestTimeout,
    ExchangeNotAvailable,
    RateLimitExceeded,
)  # <--- 确保这些基础错误已从 ccxt 导入

# Telegram Bot Library
from telegram import LinkPreviewOptions  # <--- 添加这一行 (用于解决 DeprecationWarning)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ApplicationBuilder,
    AIORateLimiter,
    Defaults,
)
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError as TelegramNetworkError
from ccxt.base.errors import BadSymbol  # 导入 BadSymbol

# --- 配置区域 ---

# Binance API Keys (从环境变量读取) - 确保有 **交易** 权限！
API_KEY = ""
API_SECRET = ""

# xt api
API_KEY = ""
API_SECRET = ""

# Telegram Bot Token (从环境变量读取)
TELEGRAM_BOT_TOKEN = ""
# 你的 Telegram 用户 ID (可选, 用于限制谁能控制机器人, 需要先知道自己的 ID)
AUTHORIZED_USER_ID = 0  # <--- 请务必修改为你的真实 TG 用户 ID

# --- DLL 配置 ---
if sys.platform == "win32":
    BF_DLL_PATH = "./arbitrage_cpp.dll"  # <--- Bellman-Ford DLL 路径 !!
    GRAPH_DLL_PATH = "./build_arbitrage_graph.dll"  # <--- 图构建 DLL 路径 !!
    OPS_DLL_PATH = "./arbitrage_operations.dll"  # <--- 风险评估/模拟 DLL 路径 !!
else:
    BF_DLL_PATH = "./arbitrage_cpp.so"  # <--- Bellman-Ford SO 路径 !!
    GRAPH_DLL_PATH = "./build_arbitrage_graph.so"  # <--- 图构建 SO 路径 !!
    OPS_DLL_PATH = "./arbitrage_operations.so"  # <--- 风险评估/模拟 SO 路径 !!

# 套利和验证配置 (设为全局变量，以便 TG Bot 修改)
config = {
    "min_profit_percent_verify": Decimal(
        "0.10"
    ),  # 模拟验证的最低利润率 (原 verify_profit_simulation 使用)
    "min_profit_full_sim_percent": Decimal(
        "0.05"
    ),  # 全路径模拟 (含闪兑) 的最低利润率 (用于 main_arbitrage_loop)
    "simulation_start_amount": Decimal("100.0"),  # 模拟起始金额 (USDT)
    "taker_fee_rate": Decimal(
        "0.00075"
    ),  # **重要**: 吃单手续费率 (Binance VIP 0 现货是 0.1%, 使用BNB抵扣是 0.075%) - 根据你的实际费率修改！
    "run_interval_seconds": 0.0005,  # 主循环计算间隔（不是获取ticker间隔）
    "max_arbitrage_depth": 5,  # 最大套利路径深度
    "running": True,  # 控制主循环是否运行
    "auto_trade_enabled": False,  # <--- 控制是否自动触发 **真实** 交易 (默认为 False, 非常重要!)
    "use_cpp_bf": True,  # 控制是否使用 C++ DLL 进行 Bellman-Ford
    "use_cpp_graph_build": True,  # <--- 新增: 控制是否使用 C++ DLL 构建图
    "use_cpp_risk_assessment": True,  # <--- 新增: 控制是否使用 C++ 进行风险评估
    "use_cpp_full_simulation": True,  # <--- 新增: 控制是否使用 C++ 进行全路径模拟
    "min_24h_quote_volume": Decimal("100"),  # 市场流动性过滤阈值 (计价货币)
    "websocket_chunk_size": 180,  # WebSocket 监听块大小
    "balance_update_interval_seconds": 5,  # <--- 余额更新间隔 (秒)
    "max_trade_retries": 2,  # <--- 交易失败时的最大重试次数
    "trade_retry_delay_sec": 1.5,  # <--- 交易重试间隔 (秒)
    "stablecoin_preference": [
        "USDT",
        "USDC",
    ],  # <--- 稳定币偏好（用于潜在的起始资金闪兑）
    "min_trade_amount_usd_equivalent": Decimal(
        "6.0"
    ),  # <--- 估计的最低交易起始金额 (略高于币安5U限制以防万一)
    "use_quote_order_qty_for_buy": True,  # <--- 买入时是否优先使用 quoteOrderQty (按花费金额购买)
    "risk_assessment_enabled": True,  # 是否启用风险评估
    "max_allowed_slippage_percent_total": Decimal(
        "0.15"
    ),  # 整个路径允许的总预估滑点百分比
    "max_bid_ask_spread_percent_per_step": Decimal(
        "0.50"
    ),  # 单步允许的最大买卖价差百分比
    "min_depth_required_usd": Decimal(
        "100.0"
    ),  # 订单簿上靠近顶部的最小流动性要求 (以USD计价)
    "min_profit_after_slippage_percent": Decimal(
        "0.05"
    ),  # 考虑滑点后的最低可接受利润率
    "order_book_depth": 10,  # 获取订单簿的深度 (例如 top 10 bids/asks)
    "ticker_batch_size": 500,  # 首次获取 Ticker 进行流动性过滤的批次大小
    "orderbook_fetch_max_workers": 4,  # 用于获取订单簿的最大线程数
}

# Decimal 精度设置
getcontext().prec = 15  # 设置足够高的精度

# --- 全局状态 (用于 Bot 和主循环通信) ---
user_chat_id = None  # 存储授权用户的 chat_id
last_verified_opportunities = []  # 存储最近发现的可验证机会摘要 (不是完整 cycle_info)
pending_confirmation = (
    {}
)  # 存储待用户确认的机会 {opp_id: cycle_info} - (目前未使用，因为自动交易直接执行)
global_tickers = {}  # 全局存储最新 Ticker 数据 (由 WebSocket 更新)
websocket_symbols = []  # 存储需要监听的交易对列表
websocket_connected = False  # 标记 WebSocket 是否连接正常
last_ticker_update_time = 0  # 记录上次收到 ticker 更新的时间
websocket_connection_status = []  # 列表，每个元素对应一个 chunk 的状态 (True/False)
ticker_watch_tasks = []  # 存储所有 watcher 任务对象，方便管理
last_cpp_relaxation_count = None
last_cycle_duration = None
last_end_time = None
global_balances = (
    {}
)  # <--- 新增: 全局存储最新余额信息 {'free': {'USDT': Decimal('...'), ...}}
balance_update_task = None  # <--- 新增: 余额更新任务句柄
arbitrage_execution_task = None  # <--- 新增: 自动套利任务句柄 (同一时间只允许一个执行)
is_trading_active = asyncio.Lock()  # <--- 新增: 锁，防止并发执行自动交易
current_execution_task = None  # <--- 新增: 当前活动的套利执行任务
snap_copy_duration_g = None
graph_build_duration_g = None
bf_call_duration_g = None
verification_duration_g = None
other_duration_g = None  # 新增，用于存储其他耗时
ops_lib = None  # 用于风险评估/模拟 (arbitrage_operations.dll)
cpp_ops_lib_loaded = False

# --- 新增：统计相关全局变量 ---
cycle_count_total = 0  # 总计算循环次数 (用于计算 CPS)
stats_reporting_start_time = 0  # 统计开始时间戳 (用于计算 CPS)
last_execution_duration_g = None  # 上次套利执行任务的耗时 (秒)


# --- 日志设置 ---
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ArbitrageBot")
logger.setLevel(logging.INFO)  # 设置主日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)
logging.getLogger("ccxtpro").setLevel(logging.INFO)  # ccxtpro 日志级别
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
if not logger.handlers:
    logger.addHandler(stream_handler)


def fetch_order_book_in_thread(pair: str, exchange_config: dict, limit: int):
    """
    在一个新的事件循环中同步地获取指定交易对的订单簿。
    设计为在 ThreadPoolExecutor 中运行。

    Args:
        pair (str): 交易对符号。
        exchange_config (dict): 用于重新创建 ccxtpro 实例的配置。
                                (注意: 传递实例本身跨线程可能不安全，传递配置更稳妥)
        limit (int): 订单簿深度。

    Returns:
        tuple: (pair, book_data) 或 (pair, None) 如果失败。
               book_data 是包含 "bids" 和 "asks" 列表的字典，元素已转为字符串。
    """
    thread_loop = None
    exchange_thread = None
    try:
        # 1. 在当前线程创建并设置新的事件循环
        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)

        # 2. 在新循环中创建 ccxtpro 交易所实例
        # 注意: 这里不传递 API Key/Secret，假设 fetch_l2_order_book 是公开端点
        # 如果需要认证的端点，则需要传递认证信息并处理可能的线程安全问题
        # 对于公开订单簿，通常不需要认证
        exchange_thread = ccxtpro.binance(
            {
                "enableRateLimit": False,  # 仍然启用内部速率限制
                "options": exchange_config.get("options", {}),  # 传递原始选项
            }
        )

        # 3. 运行异步获取函数直到完成
        # logger.debug(f"[Thread-{threading.get_ident()}] Fetching order book for {pair}...")
        book = thread_loop.run_until_complete(
            exchange_thread.fetch_l2_order_book(pair, limit=limit)
        )
        # logger.debug(f"[Thread-{threading.get_ident()}] Fetched order book for {pair}.")

        # 4. 处理结果（同原始 async 版本）
        if not isinstance(book, dict) or "bids" not in book or "asks" not in book:
            logger.warning(
                f"[fetch_order_book_in_thread] 获取的订单簿 {pair} 结构无效: {book}"
            )
            return pair, None
        bids_cpp = [
            [str(p), str(a)]
            for p, a in book.get("bids", [])
            if isinstance(p, (int, float, str, Decimal))
            and isinstance(a, (int, float, str, Decimal))
        ]
        asks_cpp = [
            [str(p), str(a)]
            for p, a in book.get("asks", [])
            if isinstance(p, (int, float, str, Decimal))
            and isinstance(a, (int, float, str, Decimal))
        ]
        return pair, {"bids": bids_cpp, "asks": asks_cpp}

    except (
        CCXTNetworkError,
        RequestTimeout,
        ExchangeNotAvailable,
        BadSymbol,
        RateLimitExceeded,
        asyncio.TimeoutError,
    ) as e:
        logger.warning(
            f"[fetch_order_book_in_thread] 获取订单簿 {pair} 失败: {type(e).__name__} - {e}"
        )
        return pair, None
    except Exception as e:
        logger.error(
            f"[fetch_order_book_in_thread] 获取订单簿 {pair} 时发生意外错误: {e}",
            exc_info=True,
        )
        return pair, None
    finally:
        # 5. 关闭交易所连接和事件循环
        if exchange_thread:
            try:
                # 需要在事件循环中关闭
                if thread_loop and thread_loop.is_running():
                    thread_loop.run_until_complete(exchange_thread.close())
                # logger.debug(f"[Thread-{threading.get_ident()}] Closed exchange for {pair}.")
            except Exception as close_exc:
                logger.warning(
                    f"[fetch_order_book_in_thread] 关闭线程内交易所实例时出错 ({pair}): {close_exc}"
                )
        if thread_loop:
            try:
                thread_loop.close()
                # logger.debug(f"[Thread-{threading.get_ident()}] Closed event loop for {pair}.")
            except Exception as loop_close_exc:
                logger.warning(
                    f"[fetch_order_book_in_thread] 关闭线程内事件循环时出错 ({pair}): {loop_close_exc}"
                )


# --- C++ 风险评估包装器 ---
async def assess_risk_cpp_wrapper(
    cycle_info: dict,
    start_amount: Decimal,
    exchange: ccxtpro.Exchange,  # 仍然需要 exchange 获取订单簿
    markets: dict,
    current_tickers: dict,
    config: dict,
) -> dict:
    """
    (包装器) 调用 C++ DLL 中的 assess_risk_cpp_buffered 函数。
    由 Python 分配缓冲区，C++ 写入结果。
    """
    global cpp_ops_lib_loaded, ops_lib, logger
    if not cpp_ops_lib_loaded or not ops_lib:
        logger.error("操作 C++ 库未加载，无法调用 assess_risk_cpp_buffered。")
        return {
            "is_viable": False,
            "reasons": ["C++ 库未加载"],
            "error": "C++ library not loaded",
        }

    # --- 1. 获取订单簿 ---
    order_books_data = {}
    path_pairs = set(
        trade.get("pair") for trade in cycle_info.get("trades", []) if trade.get("pair")
    )
    if not path_pairs:
        logger.warning("[assess_risk_cpp_wrapper] cycle_info 中没有有效的交易对。")
        return {
            "is_viable": False,
            "reasons": ["路径中无交易对"],
            "error": "No pairs in cycle",
        }

    logger.debug(f"[assess_risk_cpp_wrapper] 开始获取 {len(path_pairs)} 个订单簿...")
    order_book_depth = config.get("order_book_depth", 10)
    fetch_start_time = time.time()  # 整体获取开始时间

    # --- !! 修改点: 根据配置选择获取方式 !! ---
    if config.get("use_threaded_orderbook_fetch", False):
        logger.debug("[assess_risk_cpp_wrapper] 使用线程池获取订单簿...")
        max_workers = config.get("orderbook_fetch_max_workers")  # 获取配置的最大线程数
        results_list = []
        # 提取 exchange 的配置用于在线程中重建
        exchange_config_for_thread = {
            "apiKey": None,  # 公开数据不需要 key
            "secret": None,
            "enableRateLimit": False,  # 线程内实例也需要限速
            "options": exchange.options,  # 传递原始选项
        }
        try:
            # 使用 functools.partial 预设 exchange_config 和 limit 参数
            fetch_func_partial = functools.partial(
                fetch_order_book_in_thread,
                exchange_config=exchange_config_for_thread,
                limit=order_book_depth,
            )
            # 使用线程池执行
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # map 会阻塞直到所有结果返回
                results_list = list(executor.map(fetch_func_partial, path_pairs))

            # 处理线程池返回的结果
            for result in results_list:
                if isinstance(result, tuple):
                    pair, book_data = result
                    if book_data is not None:
                        order_books_data[pair] = book_data
                    # else: 失败已在线程函数内记录
                else:
                    # executor.map 理论上会直接返回结果或抛出线程中的异常
                    # 但以防万一，记录非元组的结果
                    logger.error(
                        f"[assess_risk_cpp_wrapper] ThreadPoolExecutor 返回了非预期的结果类型: {type(result)} - {result}"
                    )

        except Exception as thread_e:
            logger.error(
                f"[assess_risk_cpp_wrapper] 使用线程池获取订单簿时发生错误: {thread_e}",
                exc_info=True,
            )
            # 出错时，order_books_data 可能不完整，后续检查会处理

    else:
        # --- 保持原始的 asyncio.gather 方式 ---
        logger.debug("[assess_risk_cpp_wrapper] 使用 asyncio.gather 获取订单簿...")

        async def fetch_book_async(pair):  # 重命名以区分
            try:
                book = await exchange.fetch_l2_order_book(pair, limit=order_book_depth)
                if (
                    not isinstance(book, dict)
                    or "bids" not in book
                    or "asks" not in book
                ):
                    logger.warning(
                        f"[assess_risk_cpp_wrapper/async] 获取的订单簿 {pair} 结构无效: {book}"
                    )
                    return pair, None
                bids_cpp = [
                    [str(p), str(a)]
                    for p, a in book.get("bids", [])
                    if isinstance(p, (int, float, str, Decimal))
                    and isinstance(a, (int, float, str, Decimal))
                ]
                asks_cpp = [
                    [str(p), str(a)]
                    for p, a in book.get("asks", [])
                    if isinstance(p, (int, float, str, Decimal))
                    and isinstance(a, (int, float, str, Decimal))
                ]
                return pair, {"bids": bids_cpp, "asks": asks_cpp}
            except (
                CCXTNetworkError,
                RequestTimeout,
                ExchangeNotAvailable,
                BadSymbol,
                RateLimitExceeded,
                asyncio.TimeoutError,
            ) as e:
                logger.warning(
                    f"[assess_risk_cpp_wrapper/async] 获取订单簿 {pair} 失败: {type(e).__name__} - {e}"
                )
                return pair, None
            except Exception as e:
                logger.error(
                    f"[assess_risk_cpp_wrapper/async] 获取订单簿 {pair} 意外错误: {e}",
                    exc_info=True,
                )
                return pair, None

        tasks = [fetch_book_async(pair) for pair in path_pairs]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, tuple) and result[1] is not None:
                order_books_data[result[0]] = result[1]
            elif isinstance(result, tuple) and result[1] is None:
                pass  # 失败已记录
            elif isinstance(result, Exception):
                logger.error(
                    f"[assess_risk_cpp_wrapper/async] 获取订单簿任务异常: {result}",
                    exc_info=result,
                )
    # --- !! 结束修改点 !! ---

    fetch_end_time = time.time()
    logger.info(
        f"[assess_risk_cpp_wrapper] 订单簿获取耗时: {(fetch_end_time - fetch_start_time) * 1000:.2f} ms (方式: {'线程池' if config.get('use_threaded_orderbook_fetch', False) else 'asyncio.gather'})"
    )  # <--- 记录耗时
    # 检查是否所有订单簿都获取成功
    if len(order_books_data) != len(path_pairs):
        missing_pairs = path_pairs - set(order_books_data.keys())
        reason = f"未能获取所有必需的订单簿数据 (缺少: {', '.join(missing_pairs)})"
        logger.error(f"[assess_risk_cpp_wrapper] {reason}")
        return {"is_viable": False, "reasons": [reason], "error": reason}
    logger.debug(f"[assess_risk_cpp_wrapper] 订单簿获取完成.")
    prepare_start_time = time.time()
    # 2. 准备输入数据
    cycle_info_json_str = ""
    start_amount_json_str = ""
    order_books_json_str = ""
    markets_json_str = ""
    tickers_json_str = ""
    config_json_str = ""
    try:
        # 清理和准备 JSON 字符串
        cleaned_nodes = []
        for node in cycle_info.get("nodes", []):
            try:
                cleaned_node = "".join(c for c in str(node) if c.isalnum())
                if not cleaned_node:
                    raise ValueError(f"节点清理后为空: '{node}'")
                cleaned_nodes.append(cleaned_node)
            except Exception as e:
                raise ValueError(f"处理节点 '{node}' 时出错: {e}") from e

        cleaned_trades = []
        for trade in cycle_info.get("trades", []):
            try:
                original_pair = trade.get("pair")
                cleaned_pair = "".join(
                    c for c in str(original_pair) if c.isalnum() or c == "/"
                )
                if not cleaned_pair:
                    raise ValueError(f"交易对清理后为空: '{original_pair}'")
                from_node_clean = "".join(
                    c for c in str(trade.get("from")) if c.isalnum()
                )
                to_node_clean = "".join(c for c in str(trade.get("to")) if c.isalnum())
                if (
                    from_node_clean not in cleaned_nodes
                    or to_node_clean not in cleaned_nodes
                ):
                    continue  # 跳过不匹配的
                trade_type = trade.get("type")
                assert trade_type in ("BUY", "SELL")
                cleaned_trades.append(
                    {
                        "from": from_node_clean,
                        "to": to_node_clean,
                        "pair": cleaned_pair,
                        "type": trade_type,
                    }
                )
            except Exception as e:
                raise ValueError(f"处理交易 '{trade}' 时出错: {e}") from e

        cycle_info_to_cpp = {"nodes": cleaned_nodes, "trades": cleaned_trades}
        cycle_info_json_str = json.dumps(cycle_info_to_cpp)
        start_amount_json_str = str(start_amount)
        order_books_json_str = json.dumps(order_books_data)
        # 准备 markets_to_cpp
        markets_to_cpp = {}
        involved_symbols = set(t["pair"] for t in cleaned_trades)  # 使用清理后的交易对
        for symbol in involved_symbols:
            m = markets.get(symbol)
            if m:
                base = m.get("base", "")
                quote = m.get("quote", "")
                # 提供默认值 '0'
                min_amt = str(m.get("limits", {}).get("amount", {}).get("min", "0"))
                min_cost = str(m.get("limits", {}).get("cost", {}).get("min", "0"))
                markets_to_cpp[symbol] = {
                    "base": base,
                    "quote": quote,
                    "limits": {"amount": {"min": min_amt}, "cost": {"min": min_cost}},
                }
        markets_json_str = json.dumps(markets_to_cpp)
        # 准备 tickers_to_cpp
        tickers_to_cpp = {}
        involved_currencies_for_usd_est = set(cleaned_nodes)  # 使用清理后的节点
        all_symbols_needed = involved_symbols.copy()
        for curr in involved_currencies_for_usd_est:
            if curr != "USDT":
                all_symbols_needed.add(f"{curr}/USDT")
                all_symbols_needed.add(f"USDT/{curr}")
        for symbol in all_symbols_needed:
            t = current_tickers.get(symbol)
            bid_val = t.get("bid") if t else None
            ask_val = t.get("ask") if t else None
            if bid_val is not None and ask_val is not None:
                try:
                    bid_str = str(bid_val)
                    ask_str = str(ask_val)
                    bid_str.encode("utf-8")
                    ask_str.encode("utf-8")
                    tickers_to_cpp[symbol] = {"bid": bid_str, "ask": ask_str}
                except:
                    pass
        tickers_json_str = json.dumps(tickers_to_cpp)
        # 准备 config_to_cpp
        config_to_cpp = {
            k: str(v) if isinstance(v, Decimal) else v
            for k, v in config.items()
            if k
            in [
                "risk_assessment_enabled",
                "taker_fee_rate",
                "min_profit_after_slippage_percent",
                "max_allowed_slippage_percent_total",
                "min_depth_required_usd",
                "max_bid_ask_spread_percent_per_step",
                "stablecoin_preference",
            ]
        }
        config_json_str = json.dumps(config_to_cpp)

        # 编码为 bytes
        cycle_info_bytes = cycle_info_json_str.encode("utf-8")
        start_amount_bytes = start_amount_json_str.encode("utf-8")
        order_books_bytes = order_books_json_str.encode("utf-8", errors="replace")
        markets_bytes = markets_json_str.encode("utf-8", errors="replace")
        tickers_bytes = tickers_json_str.encode("utf-8", errors="replace")
        config_bytes = config_json_str.encode("utf-8")
        prepare_end_time = time.time()
        logger.info(
            f"[assess_risk_cpp_wrapper] 数据准备耗时: {(prepare_end_time - prepare_start_time) * 1000:.2f} ms"
        )
        # 创建输出缓冲区
        buffer_size = 8192  # 风险评估结果可能包含 details，稍微大一点
        output_buffer = ctypes.create_string_buffer(buffer_size)
        cpp_call_start_time = time.time()
        # --- 调用 C++ ---
        logger.debug("[assess_risk_cpp_wrapper] 调用 C++ assess_risk_cpp_buffered...")
        # 确保函数签名已在加载时设置正确
        # ops_lib.assess_risk_cpp_buffered.argtypes = [...]
        # ops_lib.assess_risk_cpp_buffered.restype = ctypes.c_int
        status_code = ops_lib.assess_risk_cpp_buffered(
            ctypes.c_char_p(cycle_info_bytes),
            ctypes.c_char_p(start_amount_bytes),
            ctypes.c_char_p(order_books_bytes),
            ctypes.c_char_p(markets_bytes),
            ctypes.c_char_p(tickers_bytes),
            ctypes.c_char_p(config_bytes),
            output_buffer,
            ctypes.c_int(buffer_size),
        )
        logger.debug(
            f"[assess_risk_cpp_wrapper] C++ assess_risk_cpp_buffered 返回状态码: {status_code}"
        )
        cpp_call_end_time = time.time()
        logger.info(
            f"[assess_risk_cpp_wrapper] C++ DLL 调用耗时: {(cpp_call_end_time - cpp_call_start_time) * 1000:.2f} ms"
        )
        # --- 处理结果 ---
        process_start_time = time.time()
        if status_code == 0:  # 成功
            result_str = ""
            try:
                result_bytes = output_buffer.value
                if result_bytes is None:
                    raise ValueError("Buffer value is None")
                result_str = result_bytes.decode("utf-8")
            except Exception as read_e:
                logger.error(
                    f"读取或解码 assess_risk 缓冲区时出错: {read_e}", exc_info=True
                )
                return {
                    "is_viable": False,
                    "reasons": [f"读取缓冲区错误: {read_e}"],
                    "error": f"Error reading buffer: {read_e}",
                }

            try:
                result_data = json.loads(result_str)
                # --- 转换回 Python 类型 ---
                if "estimated_profit_percent_after_slippage" in result_data:
                    try:
                        result_data["estimated_profit_percent_after_slippage"] = (
                            Decimal(
                                str(
                                    result_data[
                                        "estimated_profit_percent_after_slippage"
                                    ]
                                )
                            )
                        )
                    except:
                        result_data["estimated_profit_percent_after_slippage"] = (
                            Decimal("NaN")
                        )
                if "total_estimated_slippage_percent" in result_data:
                    try:
                        result_data["total_estimated_slippage_percent"] = Decimal(
                            str(result_data["total_estimated_slippage_percent"])
                        )
                    except:
                        result_data["total_estimated_slippage_percent"] = Decimal("NaN")
                if "details" in result_data and isinstance(
                    result_data["details"], list
                ):
                    for detail in result_data["details"]:
                        if "slippage_percent" in detail:
                            try:
                                detail["slippage_percent"] = Decimal(
                                    str(detail["slippage_percent"])
                                )
                            except:
                                detail["slippage_percent"] = Decimal("NaN")
                        if "spread_percent" in detail:
                            try:
                                detail["spread_percent"] = Decimal(
                                    str(detail["spread_percent"])
                                )
                            except:
                                detail["spread_percent"] = Decimal("NaN")
                        if "depth_usd" in detail:
                            try:
                                detail["depth_usd"] = Decimal(str(detail["depth_usd"]))
                            except:
                                detail["depth_usd"] = Decimal("0")

                if "error" in result_data and result_data["error"]:
                    logger.error(
                        f"C++ assess_risk_cpp_buffered 内部返回错误: {result_data['error']}"
                    )
                    result_data.setdefault("reasons", [result_data["error"]])
                    result_data.setdefault("is_viable", False)

                # 确保返回的字典包含所有预期的键，即使 C++ 没有设置
                result_data.setdefault("is_viable", False)
                result_data.setdefault("reasons", [])
                result_data.setdefault(
                    "estimated_profit_percent_after_slippage", Decimal("NaN")
                )
                result_data.setdefault(
                    "total_estimated_slippage_percent", Decimal("NaN")
                )
                result_data.setdefault("details", [])
                process_end_time = time.time()
                logger.info(
                    f"[assess_risk_cpp_wrapper] 结果处理耗时: {(process_end_time - process_start_time) * 1000:.2f} ms"
                )
                return result_data

            except json.JSONDecodeError as json_e:
                logger.error(
                    f"解析 assess_risk 缓冲区 JSON 时出错: {json_e}. 字符串: {result_str[:500]}"
                )
                return {
                    "is_viable": False,
                    "reasons": [f"JSON 解析错误: {json_e}"],
                    "error": f"JSON decode error (buffer): {json_e}",
                }
        else:
            # 处理 C++ 返回的错误码
            error_map = {
                -1: "输出缓冲区太小",
                -2: "内部 JSON 错误",
                -3: "其他内部错误",
                -5: "无效输入缓冲区",
            }
            reason = error_map.get(status_code, f"未知 C++ 错误码: {status_code}")
            logger.error(
                f"C++ assess_risk_cpp_buffered 返回错误码 {status_code}: {reason}"
            )
            return {
                "is_viable": False,
                "reasons": [f"C++ 错误: {reason}"],
                "error": f"C++ error code: {status_code}",
            }

    except Exception as e:
        logger.error(
            f"[assess_risk_cpp_wrapper] 准备调用 C++ 或处理结果时出错: {e}",
            exc_info=True,
        )
        return {
            "is_viable": False,
            "reasons": [f"Python 包装器错误: {e}"],
            "error": f"Python wrapper error: {e}",
        }

    # --- 不再需要 finally 块 ---


# --- C++ 全路径模拟包装器 (缓冲区版本) ---
async def simulate_full_cpp_wrapper(
    cycle_info: dict,
    actual_start_currency: str,
    actual_start_amount: Decimal,
    end_with_usdt: bool,
    current_tickers: dict,
    markets: dict,
    config: dict,
) -> dict:
    """
    (包装器) 调用 C++ DLL 中的 simulate_full_cpp_buffered 函数。
    由 Python 分配缓冲区，C++ 写入结果。
    """
    global cpp_ops_lib_loaded, ops_lib, logger
    if not cpp_ops_lib_loaded or not ops_lib:
        logger.error("操作 C++ 库未加载，无法调用 simulate_full_cpp_buffered。")
        return {
            "verified": False,
            "reason": "C++ 库未加载",
            "error": "C++ library not loaded",
        }

    # 准备输入数据
    (
        cycle_info_json_str,
        actual_start_currency_json_str,
        actual_start_amount_json_str,
        tickers_json_str,
        markets_json_str,
        config_json_str,
    ) = ("", "", "", "", "", "")
    try:
        # 清理和准备 JSON 字符串
        cleaned_nodes = []
        for node in cycle_info.get("nodes", []):
            try:
                cleaned_node = "".join(c for c in str(node) if c.isalnum())
                assert cleaned_node
                cleaned_nodes.append(cleaned_node)
            except:
                raise ValueError(f"无效节点: '{node}'")
        cleaned_trades = []
        for trade in cycle_info.get("trades", []):
            try:
                cleaned_pair = "".join(
                    c for c in str(trade.get("pair")) if c.isalnum() or c == "/"
                )
                assert cleaned_pair
                from_node_clean = "".join(
                    c for c in str(trade.get("from")) if c.isalnum()
                )
                assert from_node_clean
                to_node_clean = "".join(c for c in str(trade.get("to")) if c.isalnum())
                assert to_node_clean
                if (
                    from_node_clean not in cleaned_nodes
                    or to_node_clean not in cleaned_nodes
                ):
                    continue
                trade_type = trade.get("type")
                assert trade_type in ("BUY", "SELL")
                cleaned_trades.append(
                    {
                        "from": from_node_clean,
                        "to": to_node_clean,
                        "pair": cleaned_pair,
                        "type": trade_type,
                    }
                )
            except:
                raise ValueError(f"无效交易: '{trade}'")
        cycle_info_to_cpp = {"nodes": cleaned_nodes, "trades": cleaned_trades}
        cycle_info_json_str = json.dumps(cycle_info_to_cpp)
        cleaned_start_currency = "".join(
            c for c in actual_start_currency if c.isalnum()
        )
        assert cleaned_start_currency
        actual_start_currency_json_str = cleaned_start_currency
        actual_start_amount_json_str = str(actual_start_amount)
        end_with_usdt_int = 1 if end_with_usdt else 0
        tickers_to_cpp = {}
        involved_symbols_sim = set(t["pair"] for t in cleaned_trades)
        sim_currencies = set(cleaned_nodes)
        sim_currencies.add(cleaned_start_currency)
        if end_with_usdt:
            sim_currencies.add("USDT")
        for c1 in sim_currencies:
            for c2 in sim_currencies:
                if c1 != c2:
                    involved_symbols_sim.add(f"{c1}/{c2}")
                    involved_symbols_sim.add(f"{c2}/{c1}")
            if c1 != "USDT":
                involved_symbols_sim.add(f"{c1}/USDT")
                involved_symbols_sim.add(f"USDT/{c1}")
        for symbol in involved_symbols_sim:
            if not symbol or not all(c.isalnum() or c == "/" for c in symbol):
                continue
            t = current_tickers.get(symbol)
            bid_val = t.get("bid") if t else None
            ask_val = t.get("ask") if t else None
            if bid_val is not None and ask_val is not None:
                try:
                    bid_str = str(bid_val)
                    ask_str = str(ask_val)
                    bid_str.encode("utf-8")
                    ask_str.encode("utf-8")
                    tickers_to_cpp[symbol] = {"bid": bid_str, "ask": ask_str}
                except:
                    pass
        tickers_json_str = json.dumps(tickers_to_cpp)
        markets_to_cpp = {}
        for symbol in tickers_to_cpp.keys():
            m = markets.get(symbol)
            if m:
                markets_to_cpp[symbol] = {
                    "base": m.get("base", ""),
                    "quote": m.get("quote", ""),
                }
        markets_json_str = json.dumps(markets_to_cpp)
        config_to_cpp = {
            k: str(v) if isinstance(v, Decimal) else v
            for k, v in config.items()
            if k in ["taker_fee_rate", "min_profit_full_sim_percent"]
        }
        config_json_str = json.dumps(config_to_cpp)

        # 编码为 bytes
        cycle_info_bytes = cycle_info_json_str.encode("utf-8")
        actual_start_currency_bytes = actual_start_currency_json_str.encode("utf-8")
        actual_start_amount_bytes = actual_start_amount_json_str.encode("utf-8")
        tickers_bytes = tickers_json_str.encode("utf-8", errors="replace")
        markets_bytes = markets_json_str.encode("utf-8", errors="replace")
        config_bytes = config_json_str.encode("utf-8")

        # 创建输出缓冲区
        buffer_size = 4096
        output_buffer = ctypes.create_string_buffer(buffer_size)

        # 调用 C++
        logger.debug(
            "[simulate_full_cpp_wrapper] 调用 C++ simulate_full_cpp_buffered..."
        )
        # 确保函数签名已正确设置
        ops_lib.simulate_full_cpp_buffered.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        ops_lib.simulate_full_cpp_buffered.restype = ctypes.c_int
        status_code = ops_lib.simulate_full_cpp_buffered(
            ctypes.c_char_p(cycle_info_bytes),
            ctypes.c_char_p(actual_start_currency_bytes),
            ctypes.c_char_p(actual_start_amount_bytes),
            ctypes.c_int(end_with_usdt_int),
            ctypes.c_char_p(tickers_bytes),
            ctypes.c_char_p(markets_bytes),
            ctypes.c_char_p(config_bytes),
            output_buffer,
            ctypes.c_int(buffer_size),
        )
        logger.debug(
            f"[simulate_full_cpp_wrapper] C++ simulate_full_cpp_buffered 返回状态码: {status_code}"
        )

        # 处理结果
        if status_code == 0:  # 成功
            result_str = ""
            try:
                result_bytes = output_buffer.value
                if result_bytes is None:
                    raise ValueError("Buffer value is None")
                result_str = result_bytes.decode("utf-8")
            except Exception as read_e:
                logger.error(
                    f"读取或解码 simulate_full 缓冲区时出错: {read_e}", exc_info=True
                )
                return {
                    "verified": False,
                    "reason": f"读取缓冲区错误: {read_e}",
                    "error": f"Error reading buffer: {read_e}",
                }

            try:
                result_data = json.loads(result_str)
                # 转换回 Python 类型
                if "profit_percent" in result_data:
                    try:
                        result_data["profit_percent"] = Decimal(
                            str(result_data["profit_percent"])
                        )
                    except:
                        result_data["profit_percent"] = Decimal("NaN")
                if "profit_amount" in result_data:
                    try:
                        result_data["profit_amount"] = Decimal(
                            str(result_data["profit_amount"])
                        )
                    except:
                        result_data["profit_amount"] = Decimal("NaN")
                if "final_amount" in result_data:
                    try:
                        result_data["final_amount"] = Decimal(
                            str(result_data["final_amount"])
                        )
                    except:
                        result_data["final_amount"] = Decimal("NaN")

                if "error" in result_data and result_data["error"]:
                    logger.error(
                        f"C++ simulate_full_cpp_buffered 内部返回错误: {result_data['error']}"
                    )
                    result_data.setdefault("reason", result_data["error"])
                    result_data.setdefault("verified", False)

                # 确保包含基本键
                result_data.setdefault("verified", False)
                result_data.setdefault("reason", "未知")
                result_data.setdefault("profit_percent", Decimal("NaN"))
                result_data.setdefault("profit_amount", Decimal("NaN"))
                result_data.setdefault("final_amount", Decimal("NaN"))
                result_data.setdefault("final_currency", "")

                return result_data

            except json.JSONDecodeError as json_e:
                logger.error(
                    f"解析 simulate_full 缓冲区 JSON 时出错: {json_e}. 字符串: {result_str[:500]}"
                )
                return {
                    "verified": False,
                    "reason": f"JSON 解析错误 (缓冲区): {json_e}",
                    "error": f"JSON decode error (buffer): {json_e}",
                }
        else:
            error_map = {
                -1: "输出缓冲区太小",
                -2: "内部 JSON 错误",
                -3: "其他内部错误",
                -5: "无效输入缓冲区",
            }
            reason = error_map.get(status_code, f"未知 C++ 错误码: {status_code}")
            logger.error(
                f"C++ simulate_full_cpp_buffered 返回错误码 {status_code}: {reason}"
            )
            return {
                "verified": False,
                "reason": f"C++ 错误: {reason}",
                "error": f"C++ error code: {status_code}",
            }

    except Exception as e:
        logger.error(
            f"[simulate_full_cpp_wrapper] 准备调用 C++ 或处理结果时出错: {e}",
            exc_info=True,
        )
        return {
            "verified": False,
            "reason": f"Python 包装器错误: {e}",
            "error": f"Python wrapper error: {e}",
        }


async def assess_arbitrage_risk(
    cycle_info: dict,
    start_amount: Decimal,  # 本次套利实际打算使用的起始金额 (cycle_start_node 计价)
    exchange: ccxtpro.Exchange,
    markets: dict,
    current_tickers: dict,  # 用于快速获取USDT汇率估算深度
    config: dict,
) -> dict:
    """
    (修正版) 在执行前评估套利机会的风险。
    - 修复了中间金额计算错误。
    - 修复了处理模拟成交失败的逻辑。
    - 使用传入的 Ticker 快照。

    Args:
        cycle_info: 套利路径信息.
        start_amount: 计划用于此路径的起始资金量 (以环路起始货币计价).
        exchange: ccxtpro 交易所实例.
        markets: 市场数据.
        current_tickers: 当前 Ticker 快照 (用于估算非 USDT 交易对的深度价值).
        config: 全局配置字典.

    Returns:
        一个包含评估结果的字典:
        {
            'is_viable': bool,           # 是否认为风险可接受
            'estimated_profit_percent_after_slippage': Decimal, # 考虑滑点和费用后的预估利润率
            'total_estimated_slippage_percent': Decimal, # 总预估滑点百分比
            'reasons': list[str],        # 风险点或不建议执行的原因列表
            'details': list[dict]        # 每一步的详细风险评估
        }
    """
    if not config.get("risk_assessment_enabled", False):
        return {
            "is_viable": True,
            "reasons": ["风险评估未启用"],
            "details": [],
            "estimated_profit_percent_after_slippage": cycle_info.get(
                "full_simulation_profit_percent", Decimal("-1")
            ),  # 使用全路径模拟利润作为默认
            "total_estimated_slippage_percent": Decimal("0"),
        }

    logger.info(
        f"[风险评估] 开始评估路径: {' -> '.join(cycle_info['nodes'])} 起始金额: {format_decimal(start_amount)} {cycle_info['nodes'][0]}"
    )

    trades = cycle_info["trades"]
    path_start_currency = cycle_info["nodes"][0]  # 路径定义的起始货币
    current_currency = path_start_currency
    # intermediate_amount 是模拟风险评估过程中，每一步执行后持有的金额
    intermediate_amount = start_amount  # 评估从路径定义的起始金额开始
    fee_rate = config["taker_fee_rate"]
    book_depth_limit = config.get("order_book_depth", 10)
    min_profit_req = config["min_profit_after_slippage_percent"]
    max_slip_req = config["max_allowed_slippage_percent_total"]
    min_depth_usd_req = config.get("min_depth_required_usd", Decimal("100.0"))
    max_spread_req = config["max_bid_ask_spread_percent_per_step"]

    reasons = []
    step_details = []
    total_slippage_cost_usd = Decimal("0")
    start_value_usd_est = Decimal("0")  # 用于计算总滑点百分比的初始价值估算

    # --- 估算初始金额的 USD 价值 ---
    try:
        if path_start_currency == "USDT":
            start_value_usd_est = start_amount
        else:
            ticker_fwd = f"{path_start_currency}/USDT"
            ticker_rev = f"USDT/{path_start_currency}"
            snap_fwd = current_tickers.get(ticker_fwd)
            snap_rev = current_tickers.get(ticker_rev)
            price = None
            if snap_fwd and snap_fwd.get("bid"):  # 使用 bid 估算卖出价值
                price = Decimal(str(snap_fwd["bid"]))
            elif snap_rev and snap_rev.get("ask"):
                ask_rev = Decimal(str(snap_rev["ask"]))
                if ask_rev > 0:
                    price = Decimal("1.0") / ask_rev

            if price and price > 0:
                start_value_usd_est = start_amount * price
            elif path_start_currency in config.get(
                "stablecoin_preference", []
            ):  # 近似其他稳定币
                start_value_usd_est = start_amount  # 近似为 1 USD
        logger.debug(
            f"[风险评估] 估算起始 USD 价值: ${format_decimal(start_value_usd_est, 2)}"
        )
    except Exception as e:
        logger.warning(f"[风险评估] 估算起始 USD 价值时出错: {e}")
        # 即使估算失败，也继续评估，只是总滑点百分比可能不准确

    # --- 遍历交易步骤 ---
    for i, trade in enumerate(trades):
        step_num = i + 1
        pair = trade["pair"]
        trade_type = trade["type"]
        from_currency = trade["from"]
        to_currency = trade["to"]
        step_detail = {
            "step": step_num,
            "pair": pair,
            "type": trade_type,
            "slippage_percent": Decimal("NaN"),
            "spread_percent": Decimal("NaN"),
            "depth_ok": False,
            "depth_usd": Decimal("0"),
            "limits_ok": True,
            "message": "",
        }

        # 检查传入金额是否有效
        if intermediate_amount <= 0:
            msg = f"步骤 {step_num}: 上一步计算得到的金额无效 ({intermediate_amount})，中止风险评估。"
            logger.error(f"[风险评估] {msg}")
            reasons.append(msg)
            step_detail["message"] = msg
            step_details.append(step_detail)
            intermediate_amount = Decimal("-1")  # 标记失败
            break  # 中止后续步骤评估

        # 检查货币是否匹配
        if current_currency != from_currency:
            msg = f"逻辑错误：步骤 {step_num} 需要发送 {from_currency}, 但当前持有 {current_currency}"
            logger.error(f"[风险评估] {msg}")
            reasons.append(f"步骤 {step_num}: {msg}")
            step_detail["message"] = msg
            intermediate_amount = Decimal("-1")  # 标记失败
            break

        market = markets.get(pair)
        if not market:
            msg = f"步骤 {step_num}: 找不到市场 {pair}"
            logger.error(f"[风险评估] {msg}")
            reasons.append(msg)
            step_detail["message"] = msg
            step_details.append(step_detail)
            intermediate_amount = Decimal("-1")  # 标记失败
            continue  # 跳到下一个循环（虽然已经失败了）

        base_curr = market["base"]
        quote_curr = market["quote"]
        limits = market.get("limits", {})
        precision = market.get("precision", {})
        min_amount = Decimal(str(limits.get("amount", {}).get("min", "0")))
        min_cost = Decimal(str(limits.get("cost", {}).get("min", "0")))

        # --- 1. 获取订单簿 ---
        order_book = None
        try:
            order_book = await exchange.fetch_l2_order_book(
                pair, limit=book_depth_limit
            )
            if (
                not order_book
                or not order_book.get("bids")
                or not order_book.get("asks")
            ):
                raise ValueError(
                    f"订单簿数据不完整 (bids: {len(order_book.get('bids',[]))}, asks: {len(order_book.get('asks',[]))})"
                )
        except (
            CCXTNetworkError,
            RequestTimeout,
            ExchangeNotAvailable,
            RateLimitExceeded,
            BadSymbol,
            ValueError,
        ) as e:
            msg = f"步骤 {step_num} ({pair}): 获取订单簿失败: {type(e).__name__} - {e}"
            logger.warning(f"[风险评估] {msg}")
            reasons.append(msg + " (无法评估滑点/流动性)")
            step_detail["message"] = msg
            step_details.append(step_detail)
            intermediate_amount = Decimal("-1")  # 标记失败
            continue  # 跳到下一个循环

        top_bid = (
            Decimal(str(order_book["bids"][0][0]))
            if order_book["bids"]
            else Decimal("0")
        )
        top_ask = (
            Decimal(str(order_book["asks"][0][0]))
            if order_book["asks"]
            else Decimal("Infinity")
        )

        # --- 2. 检查价差 ---
        if top_bid > 0 and top_ask != Decimal("Infinity"):
            spread = top_ask - top_bid
            spread_percent = (
                (spread / top_ask) * 100 if top_ask > 0 else Decimal("Infinity")
            )
            step_detail["spread_percent"] = spread_percent
            if spread_percent > max_spread_req:
                msg = f"价差过高 ({spread_percent:.3f}% > {max_spread_req}%)"
                logger.warning(f"[风险评估] 步骤 {step_num} ({pair}): {msg}")
                reasons.append(f"步骤 {step_num}: {msg}")
                step_detail["message"] += msg + "; "
        else:
            spread_percent = Decimal("Infinity")
            step_detail["spread_percent"] = spread_percent
            msg = f"订单簿缺少买/卖单，无法计算价差"
            logger.warning(f"[风险评估] 步骤 {step_num} ({pair}): {msg}")
            reasons.append(f"步骤 {step_num}: {msg}")
            step_detail["message"] += msg + "; "

        # --- 3. 估算滑点并检查深度/限制 ---
        estimated_executed_amount = Decimal("0")  # 模拟执行后收到的数量 (扣费前)
        sim_average_price = None  # 模拟执行的平均价格
        slippage_percent_step = Decimal("NaN")
        accumulated_base = Decimal("0")
        cost_accumulated = Decimal("0")
        accumulated_quote = Decimal("0")
        amount_sold = Decimal("0")
        depth_usd_available = Decimal("0")

        # 估算本次交易的 USD 价值
        approx_trade_value_usd = Decimal("0")
        try:
            if current_currency == "USDT":
                approx_trade_value_usd = intermediate_amount
            else:
                # 使用快照估值
                ticker_fwd = f"{current_currency}/USDT"
                ticker_rev = f"USDT/{current_currency}"
                snap_fwd = current_tickers.get(ticker_fwd)
                snap_rev = current_tickers.get(ticker_rev)
                price = None
                if snap_fwd and snap_fwd.get("bid"):
                    price = Decimal(str(snap_fwd["bid"]))
                elif snap_rev and snap_rev.get("ask"):
                    ask_rev = Decimal(str(snap_rev["ask"]))
                    if ask_rev > 0:
                        price = Decimal("1.0") / ask_rev
                if price:
                    approx_trade_value_usd = intermediate_amount * price
                elif current_currency in config.get("stablecoin_preference", []):
                    approx_trade_value_usd = intermediate_amount  # 近似稳定币

            logger.debug(
                f"[风险评估] 步骤 {step_num} ({pair}): 估计交易价值 ${format_decimal(approx_trade_value_usd, 2)}"
            )
        except Exception as e:
            logger.warning(f"[风险评估] 估算步骤 {step_num} 交易价值时出错: {e}")

        # --- 模拟订单簿成交 ---
        try:
            if (
                trade_type == "BUY"
            ):  # 买入 base, 花费 quote (intermediate_amount 是 quote)
                amount_to_spend = intermediate_amount

                # 检查最低成本限制
                if min_cost > 0 and amount_to_spend < min_cost:
                    msg = f"预估花费 {format_decimal(amount_to_spend)} {quote_curr} 低于最小成本 {min_cost}"
                    logger.warning(f"[风险评估] 步骤 {step_num} ({pair}): {msg}")
                    reasons.append(f"步骤 {step_num}: {msg}")
                    step_detail["limits_ok"] = False
                    step_detail["message"] += msg + "; "
                    # 不中止，继续模拟看是否能成交

                # 遍历卖单 (asks) 估算深度和成交
                for price_level, amount_level in order_book["asks"]:
                    price = Decimal(str(price_level))
                    amount = Decimal(str(amount_level))
                    cost_at_level = price * amount

                    # 估算此层级的 USD 价值 (用于深度检查)
                    level_usd_value = Decimal("0")
                    if quote_curr == "USDT":
                        level_usd_value = cost_at_level
                    else:  # 尝试从快照估值
                        ticker_fwd_q = f"{quote_curr}/USDT"
                        ticker_rev_q = f"USDT/{quote_curr}"
                        snap_fwd_q = current_tickers.get(ticker_fwd_q)
                        snap_rev_q = current_tickers.get(ticker_rev_q)
                        price_q = None
                        if snap_fwd_q and snap_fwd_q.get("bid"):
                            price_q = Decimal(str(snap_fwd_q["bid"]))
                        elif snap_rev_q and snap_rev_q.get("ask"):
                            ask_rev_q = Decimal(str(snap_rev_q["ask"]))
                            price_q = (
                                Decimal("1.0") / ask_rev_q if ask_rev_q > 0 else None
                            )
                        if price_q:
                            level_usd_value = cost_at_level * price_q
                    depth_usd_available += level_usd_value

                    # 模拟成交
                    remaining_spend = amount_to_spend - cost_accumulated
                    if remaining_spend <= 0:
                        break
                    buy_amount_base_at_level = min(remaining_spend / price, amount)
                    cost_this_level = buy_amount_base_at_level * price
                    accumulated_base += buy_amount_base_at_level
                    cost_accumulated += cost_this_level

                # --- 计算买入结果 ---
                estimated_executed_amount = (
                    accumulated_base  # 买单执行后得到的是 base 数量 (扣费前)
                )
                if accumulated_base > 0:
                    sim_average_price = cost_accumulated / accumulated_base
                    if top_ask > 0:
                        # 滑点 = (实际均价 - 最优价) / 最优价 * 100 (负数表示有利或价格优于 top ask)
                        slippage_percent_step = (
                            (sim_average_price - top_ask) / top_ask
                        ) * 100
                    # 检查最低买入量限制
                    if min_amount > 0 and estimated_executed_amount < min_amount:
                        msg = f"预估买入量 {format_decimal(estimated_executed_amount)} {base_curr} 低于最小量 {min_amount}"
                        logger.warning(f"[风险评估] 步骤 {step_num} ({pair}): {msg}")
                        reasons.append(f"步骤 {step_num}: {msg}")
                        step_detail["limits_ok"] = False
                        step_detail["message"] += msg + "; "
                else:  # 无法成交
                    sim_average_price = None
                    slippage_percent_step = Decimal("Infinity")
                    msg = "订单簿深度不足或花费过低，无法模拟买入"
                    reasons.append(f"步骤 {step_num} ({pair}): {msg}")
                    step_detail["limits_ok"] = False
                    step_detail["message"] += msg + "; "
                    intermediate_amount = Decimal("-1")  # 标记失败

            elif (
                trade_type == "SELL"
            ):  # 卖出 base, 收到 quote (intermediate_amount 是 base)
                amount_to_sell = intermediate_amount

                # 检查最低卖出量限制
                if min_amount > 0 and amount_to_sell < min_amount:
                    msg = f"预估卖出量 {format_decimal(amount_to_sell)} {base_curr} 低于最小量 {min_amount}"
                    logger.warning(f"[风险评估] 步骤 {step_num} ({pair}): {msg}")
                    reasons.append(f"步骤 {step_num}: {msg}")
                    step_detail["limits_ok"] = False
                    step_detail["message"] += msg + "; "
                    # 不中止，继续模拟

                # 遍历买单 (bids) 估算深度和成交
                for price_level, amount_level in order_book["bids"]:
                    price = Decimal(str(price_level))
                    amount = Decimal(str(amount_level))

                    # 估算此层级的 USD 价值
                    level_usd_value = Decimal("0")
                    if base_curr == "USDT":
                        level_usd_value = (
                            amount * price
                        )  # 如果卖的是USDT...不太可能但处理一下
                    else:
                        ticker_fwd_b = f"{base_curr}/USDT"
                        ticker_rev_b = f"USDT/{base_curr}"
                        snap_fwd_b = current_tickers.get(ticker_fwd_b)
                        snap_rev_b = current_tickers.get(ticker_rev_b)
                        price_b = None
                        if snap_fwd_b and snap_fwd_b.get("bid"):
                            price_b = Decimal(str(snap_fwd_b["bid"]))
                        elif snap_rev_b and snap_rev_b.get("ask"):
                            ask_rev_b = Decimal(str(snap_rev_b["ask"]))
                            price_b = (
                                Decimal("1.0") / ask_rev_b if ask_rev_b > 0 else None
                            )
                        if price_b:
                            level_usd_value = (
                                amount * price_b
                            )  # 估算这层 base 的 USD 价值
                    depth_usd_available += level_usd_value

                    # 模拟成交
                    remaining_sell = amount_to_sell - amount_sold
                    if remaining_sell <= 0:
                        break
                    sell_amount_base_at_level = min(remaining_sell, amount)
                    quote_received_this_level = sell_amount_base_at_level * price
                    accumulated_quote += quote_received_this_level
                    amount_sold += sell_amount_base_at_level

                # --- 计算卖出结果 ---
                estimated_executed_amount = (
                    accumulated_quote  # 卖单执行后收到的是 quote 数量 (扣费前)
                )
                if amount_sold > 0:
                    sim_average_price = accumulated_quote / amount_sold
                    if top_bid > 0:
                        # 滑点 = (最优价 - 实际均价) / 最优价 * 100 (负数表示有利或价格优于 top bid)
                        slippage_percent_step = (
                            (top_bid - sim_average_price) / top_bid
                        ) * 100
                    # 检查最低成本限制（卖出所得）
                    if min_cost > 0 and estimated_executed_amount < min_cost:
                        msg = f"预估卖出所得 {format_decimal(estimated_executed_amount)} {quote_curr} 低于最小成本 {min_cost}"
                        logger.warning(f"[风险评估] 步骤 {step_num} ({pair}): {msg}")
                        # 通常不认为这是硬性失败条件，但记录
                        step_detail["message"] += msg + "; "
                        # 如果需要严格执行，可以在这里设置 step_detail['limits_ok'] = False
                else:  # 无法成交
                    sim_average_price = None
                    slippage_percent_step = Decimal("Infinity")
                    msg = "订单簿深度不足或卖出量过低，无法模拟卖出"
                    reasons.append(f"步骤 {step_num} ({pair}): {msg}")
                    step_detail["limits_ok"] = False
                    step_detail["message"] += msg + "; "
                    intermediate_amount = Decimal("-1")  # 标记失败

            # --- 检查订单簿深度 ---
            step_detail["depth_usd"] = depth_usd_available
            if depth_usd_available >= min_depth_usd_req:
                step_detail["depth_ok"] = True
                # logger.debug(f"[风险评估] 步骤 {step_num} ({pair}): 深度满足要求 (${format_decimal(depth_usd_available, 2)} >= ${min_depth_usd_req})")
            else:
                msg = f"订单簿深度不足 (仅约 ${format_decimal(depth_usd_available, 2)} < ${min_depth_usd_req})"
                logger.warning(f"[风险评估] 步骤 {step_num} ({pair}): {msg}")
                reasons.append(f"步骤 {step_num}: {msg}")
                step_detail["depth_ok"] = False
                step_detail["message"] += msg + "; "

            # --- 记录滑点并更新中间金额 (如果模拟未失败) ---
            if intermediate_amount >= 0:
                step_detail["slippage_percent"] = slippage_percent_step
                # 估算滑点成本
                slippage_cost_step_usd = Decimal("0")
                if (
                    not slippage_percent_step.is_nan()
                    and not slippage_percent_step.is_infinite()
                    and approx_trade_value_usd > 0
                ):
                    # 注意：滑点计算方式统一为负数代表不利
                    # 成本 = 交易额 * abs(不利滑点百分比) / 100
                    slippage_cost_step_usd = approx_trade_value_usd * (
                        max(Decimal("0"), slippage_percent_step) / 100
                    )  # 只计算不利滑点的成本
                    # 如果买单，slip>0 是不利；如果卖单，slip>0 是有利。统一为 slip>0 为不利。
                    # 买： slip = (avg - ask)/ask > 0 不利
                    # 卖： slip = (bid - avg)/bid > 0 不利 (avg < bid)
                    # 因此，直接用 slip > 0 判断是否不利滑点
                    if slippage_percent_step > 0:
                        slippage_cost_step_usd = approx_trade_value_usd * (
                            slippage_percent_step / 100
                        )
                    else:
                        slippage_cost_step_usd = Decimal("0")  # 有利滑点成本为0

                total_slippage_cost_usd += slippage_cost_step_usd

                avg_price_str = (
                    format_decimal(sim_average_price)
                    if sim_average_price is not None
                    else "N/A"
                )
                slip_str = (
                    f"{slippage_percent_step:+.4f}%"
                    if not slippage_percent_step.is_nan()
                    and not slippage_percent_step.is_infinite()
                    else "N/A"
                )
                cost_str = (
                    f"${format_decimal(slippage_cost_step_usd, 4)}"
                    if slippage_cost_step_usd > 0
                    else "$0.0000"
                )

                logger.debug(
                    f"[风险评估] 步骤 {step_num} ({pair}): 模拟均价 {avg_price_str}, 滑点 {slip_str}, 估计成本 {cost_str}"
                )

                # --- 更新滑点和费用后的中间金额 ---
                # **关键**: 扣除手续费
                fee_to_deduct = estimated_executed_amount * fee_rate
                next_intermediate_amount = estimated_executed_amount - fee_to_deduct

                # **关键**: 更新持有货币
                if trade_type == "BUY":
                    current_currency = base_curr
                elif trade_type == "SELL":
                    current_currency = quote_curr

                intermediate_amount = next_intermediate_amount
                logger.debug(
                    f"[风险评估] 步骤 {step_num} 后，考虑滑点和费用，中间金额: {format_decimal(intermediate_amount)} {current_currency}"
                )

            else:  # 滑点无限大（无法成交）
                logger.warning(
                    f"[风险评估] 步骤 {step_num} ({pair}): 模拟成交失败，中止后续计算。"
                )
                step_detail["message"] += "模拟成交失败; "
                # intermediate_amount 已经是 -1

        except (
            ValueError,
            DecimalInvalidOperation,
            TypeError,
        ) as calc_e:  # 捕获计算中可能的错误
            msg = f"步骤 {step_num} ({pair}): 模拟执行计算时出错: {type(calc_e).__name__} - {calc_e}"
            logger.error(f"[风险评估] {msg}", exc_info=True)
            reasons.append(msg)
            step_detail["message"] = msg
            intermediate_amount = Decimal("-1")  # 标记利润计算无效
        except Exception as e:  # 捕获其他意外错误
            msg = f"步骤 {step_num} ({pair}): 模拟执行时发生意外错误: {e}"
            logger.error(f"[风险评估] {msg}", exc_info=True)
            reasons.append(msg)
            step_detail["message"] = msg
            intermediate_amount = Decimal("-1")

        step_details.append(step_detail)

        # 如果中间金额无效，提前退出循环
        if intermediate_amount < 0:
            logger.warning(
                f"[风险评估] 因步骤 {step_num} 错误或失败，提前中止路径评估。"
            )
            break

    # --- 4. 计算最终滑点后利润和总滑点百分比 ---
    estimated_profit_percent_after_slippage = Decimal("-999")
    total_estimated_slippage_percent = Decimal("-999")

    if intermediate_amount >= 0 and current_currency == path_start_currency:
        profit_amount = intermediate_amount - start_amount
        if start_amount > 0:
            estimated_profit_percent_after_slippage = (
                profit_amount / start_amount
            ) * 100
        else:
            estimated_profit_percent_after_slippage = Decimal("0")

        # 估算总滑点百分比（基于初始USD价值估算）
        if start_value_usd_est > 0:
            total_estimated_slippage_percent = (
                total_slippage_cost_usd / start_value_usd_est
            ) * 100
        elif total_slippage_cost_usd == 0:  # 如果初始价值为0且成本为0
            total_estimated_slippage_percent = Decimal("0")
        else:  # 初始价值为0但成本不为0
            total_estimated_slippage_percent = Decimal("Infinity")

        logger.info(f"[风险评估] 路径评估完成:")
        logger.info(f"  - 起始 ({path_start_currency}): {format_decimal(start_amount)}")
        logger.info(
            f"  - 结束 ({current_currency}): {format_decimal(intermediate_amount)}"
        )
        logger.info(
            f"  - 考虑滑点和费用后预估利润率: {estimated_profit_percent_after_slippage:.4f}%"
        )
        logger.info(
            f"  - 总预估滑点百分比 (基于初始价值): {total_estimated_slippage_percent:.4f}% (总成本约 ${format_decimal(total_slippage_cost_usd, 4)})"
        )

        # 根据阈值添加原因
        if estimated_profit_percent_after_slippage < min_profit_req:
            reasons.append(
                f"预估滑点后利润率 ({estimated_profit_percent_after_slippage:.4f}%) 低于要求 ({min_profit_req}%)"
            )
        if total_estimated_slippage_percent > max_slip_req:
            reasons.append(
                f"总预估滑点 ({total_estimated_slippage_percent:.4f}%) 高于阈值 ({max_slip_req}%)"
            )

    elif intermediate_amount < 0:
        # 如果是因为模拟失败导致 intermediate_amount < 0
        if not any(
            "模拟执行" in r or "计算错误" in r or "意外错误" in r for r in reasons
        ):
            reasons.append("因某一步骤模拟成交失败或低于限制，无法计算最终利润")
        logger.warning("[风险评估] 无法计算最终滑点后利润。")
    elif current_currency != path_start_currency:
        reasons.append(
            f"风险评估后最终货币 ({current_currency}) 与起始 ({path_start_currency}) 不符"
        )
        logger.error(f"[风险评估] 风险评估后货币不符！")

    # --- 5. 判断最终可行性 ---
    is_viable = True
    # 关键失败条件
    if intermediate_amount < 0:
        is_viable = False  # 计算出错或模拟失败
    if estimated_profit_percent_after_slippage < min_profit_req:
        is_viable = False  # 利润不足
    if total_estimated_slippage_percent > max_slip_req:
        is_viable = False  # 总滑点过高
    # 检查是否有任何一步深度不足或不满足限制 (如果 limits_ok 为 False)
    if any(not step["depth_ok"] for step in step_details):
        is_viable = False
    if any(not step["limits_ok"] for step in step_details):
        is_viable = False
    # 确保最终货币匹配 (如果计算成功)
    if intermediate_amount >= 0 and current_currency != path_start_currency:
        is_viable = False

    logger.info(f"[风险评估] 最终评估结果: {'可行' if is_viable else '不可行'}")
    if reasons:
        for reason in reasons:
            logger.info(f"  - 原因/警告: {reason}")

    return {
        "is_viable": is_viable,
        "estimated_profit_percent_after_slippage": (
            estimated_profit_percent_after_slippage
            if intermediate_amount >= 0
            else Decimal("-999")
        ),
        "total_estimated_slippage_percent": (
            total_estimated_slippage_percent
            if start_value_usd_est > 0
            else Decimal("-999")
        ),  # 初始价值为0则无法计算
        "reasons": reasons,
        "details": step_details,
    }


# --- 格式化 Decimal 辅助函数 (增强健壮性) ---
def format_decimal(d, precision=8):
    """向下取整格式化 Decimal 并返回字符串，增加健壮性和调试日志"""
    original_d = d
    try:
        if not isinstance(d, Decimal):
            d = Decimal(str(d))  # 优先使用字符串转换

        if not d.is_finite():
            # logger.warning(f"format_decimal 接收到非有限数: {d}")
            return str(d)  # 直接返回 'Infinity', '-Infinity', 'NaN'

        if not isinstance(precision, int) or precision < 0:
            precision = 8

        quantizer = Decimal("1E-" + str(precision))
        quantized_d = d.quantize(quantizer, rounding=ROUND_DOWN)
        return str(quantized_d)
    except (TypeError, ValueError, DecimalInvalidOperation) as convert_e:
        logger.error(
            f"format_decimal 无法将输入 {repr(original_d)} (type: {type(original_d)}) 转换为 Decimal 或量化: {convert_e}"
        )
        return "格式错误"
    except Exception as e:
        logger.error(
            f"format_decimal 对值 {repr(d)} (原始: {repr(original_d)}) 使用精度 {precision} 时发生未知错误: {e}",
            exc_info=True,
        )
        return "量化错误"


# --- 解析订单结果辅助函数 (增强版：计算均价和滑点) ---
def parse_order_result(
    order_info: dict,
    base_currency: str,
    quote_currency: str,
    side: str,
    expected_price: Decimal = None,  # <--- 新增：预期价格
    requested_amount: Decimal = None,
    requested_cost: Decimal = None,
) -> dict:
    """
    (增强版) 解析 CCXT 市价订单结果并格式化为标准字典。
    计算实际成交均价和滑点（如果提供了预期价格）。
    """
    now = time.time()
    status = "error"
    message = "订单信息不完整或解析失败"
    order_id = order_info.get("id")
    executed_base_amount = Decimal("0")  # 实际成交的基础货币量 (filled)
    executed_quote_amount = Decimal("0")  # 实际花费/收到的报价货币量 (cost)
    average_price = None  # <--- 新增：实际成交均价
    slippage_percent = None  # <--- 新增：滑点百分比
    received_amount = None  # 最终到手数量 (扣费后)
    received_currency = None  # 最终到手货币
    spent_amount = None  # 最终花费数量 (扣费前，等于 filled 或 cost)
    spent_currency = None  # 最终花费货币
    fee_amount = None
    fee_currency = None

    try:
        order_status = order_info.get("status")
        # 对于市价单，closed 通常意味着完全成交
        if order_status not in ["filled", "closed"]:
            logger.warning(f"订单 {order_id} 状态为 '{order_status}'，结果可能不准确。")
            message = f"订单 {order_id} 状态为 '{order_status}'"
            # 继续尝试解析，因为部分信息可能可用

        # --- 从订单信息中提取核心数据 ---
        filled_base_str = order_info.get("filled", "0")
        cost_quote_str = order_info.get("cost", "0")
        fee_info = order_info.get("fee")

        try:
            executed_base_amount = Decimal(filled_base_str)
            executed_quote_amount = Decimal(cost_quote_str)
        except (DecimalInvalidOperation, ValueError):
            logger.error(
                f"解析订单 {order_id} 的 filled ('{filled_base_str}') 或 cost ('{cost_quote_str}') 时失败。"
            )
            # 即使解析失败，也尝试继续处理其他信息

        # --- 计算实际成交均价 ---
        # 确保 executed_base_amount > 0 避免除零
        if executed_base_amount > 0 and executed_quote_amount > 0:
            average_price = executed_quote_amount / executed_base_amount
            logger.debug(
                f"订单 {order_id}: 实际成交均价计算为 {format_decimal(average_price)}"
            )
        else:
            logger.warning(
                f"订单 {order_id}: filled ({executed_base_amount}) 或 cost ({executed_quote_amount}) 为零，无法计算实际均价。"
            )

        # --- 计算滑点 (如果提供了预期价格和计算了实际均价) ---
        if expected_price is not None and average_price is not None:
            try:
                expected_price = Decimal(str(expected_price))  # 确保是 Decimal
                if expected_price > 0:
                    if side == "buy":
                        # 买单：实际价比预期价高 -> 负滑点 (不好)
                        slippage_percent = (
                            (average_price - expected_price) / expected_price
                        ) * 100
                    elif side == "sell":
                        # 卖单：实际价比预期价低 -> 负滑点 (不好)
                        slippage_percent = (
                            (average_price - expected_price) / expected_price
                        ) * 100
                        # 或者用另一种等价方式：((expected_price - average_price) / expected_price) * 100，这样正滑点代表有利
                        # 为了统一，我们都用 (实际 - 预期) / 预期，然后解读：买单负滑点好，卖单正滑点好
                        # 修正：统一 买单：实际 > 预期 是坏事；卖单：实际 < 预期 是坏事
                        # 因此，买单滑点 = (实际 - 预期)/预期 * 100  (负数好)
                        #      卖单滑点 = (预期 - 实际)/预期 * 100  (负数好)
                        # 采用新的计算方式，统一负数为不利滑点
                        if side == "buy":
                            slippage_percent = (
                                (average_price - expected_price) / expected_price
                            ) * Decimal("100")
                        elif side == "sell":
                            slippage_percent = (
                                (expected_price - average_price) / expected_price
                            ) * Decimal("100")

                    logger.debug(
                        f"订单 {order_id}: 预期价={format_decimal(expected_price)}, 实际均价={format_decimal(average_price)}, 滑点={slippage_percent:.4f}%"
                    )
                else:
                    logger.warning(
                        f"订单 {order_id}: 提供的预期价格无效 ({expected_price})，无法计算滑点。"
                    )
            except (DecimalInvalidOperation, ValueError, TypeError) as slip_e:
                logger.error(f"计算订单 {order_id} 滑点时出错: {slip_e}")
                slippage_percent = None  # 计算失败
        elif expected_price is None:
            logger.debug(f"订单 {order_id}: 未提供预期价格，不计算滑点。")
        elif average_price is None:
            logger.debug(f"订单 {order_id}: 无法计算实际均价，不计算滑点。")

        # --- 处理手续费 ---
        if (
            fee_info
            and isinstance(fee_info, dict)
            and "cost" in fee_info
            and "currency" in fee_info
        ):
            try:
                fee_amount = Decimal(str(fee_info.get("cost", "0")))
                fee_currency = fee_info.get("currency")
            except (DecimalInvalidOperation, ValueError):
                logger.warning(
                    f"解析订单 {order_id} 的手续费信息 cost: '{fee_info.get('cost')}' 失败。"
                )
                fee_amount = Decimal("0")  # 假设为 0
        else:
            logger.warning(f"订单 {order_id} 未提供有效的手续费信息: {fee_info}。")
            fee_amount = Decimal("0")  # 假设为 0

        # --- 根据买卖方向计算最终收支 (spent 是指消耗的资产量，received 是指获得的资产量) ---
        if side == "buy":  # 买入 base, 花费 quote
            received_amount = executed_base_amount  # 收到的是 base
            received_currency = base_currency
            spent_amount = executed_quote_amount  # 花费的是 quote
            spent_currency = quote_currency
            # 如果手续费是以收到的 base 货币支付 (币安常见)
            if fee_currency == base_currency:
                received_amount -= fee_amount
                logger.debug(
                    f"买入手续费以 {base_currency} 支付，扣除 {format_decimal(fee_amount)}"
                )
            # 如果手续费是以花费的 quote 货币支付 (例如用BNB抵扣，或其他交易所可能如此)
            elif fee_currency == quote_currency:
                logger.debug(
                    f"买入手续费以 {quote_currency} 支付 ({format_decimal(fee_amount)})"
                )
            # 如果手续费是其他货币（例如BNB）
            elif fee_currency:
                logger.debug(
                    f"买入手续费以 {fee_currency} 支付 ({format_decimal(fee_amount)})"
                )

        elif side == "sell":  # 卖出 base, 收到 quote
            received_amount = executed_quote_amount  # 收到的是 quote
            received_currency = quote_currency
            spent_amount = executed_base_amount  # 花费的是 base
            spent_currency = base_currency
            # 如果手续费是以收到的 quote 货币支付 (币安常见)
            if fee_currency == quote_currency:
                received_amount -= fee_amount
                logger.debug(
                    f"卖出手续费以 {quote_currency} 支付，扣除 {format_decimal(fee_amount)}"
                )
            # 如果手续费是以卖出的 base 货币支付
            elif fee_currency == base_currency:
                logger.debug(
                    f"卖出手续费以 {base_currency} 支付 ({format_decimal(fee_amount)})"
                )
            # 如果手续费是其他货币（例如BNB）
            elif fee_currency:
                logger.debug(
                    f"卖出手续费以 {fee_currency} 支付 ({format_decimal(fee_amount)})"
                )

        # --- 处理 API 可能未返回 cost 或 filled 的情况 ---
        if (
            side == "buy"
            and spent_amount == 0
            and requested_cost
            and requested_cost > 0
        ):
            spent_amount = requested_cost  # 用请求的花费作为估算
            logger.warning(
                f"订单 {order_id} 未返回 cost, 使用请求值 {format_decimal(spent_amount)} {spent_currency} 估算。"
            )
        if (
            side == "sell"
            and spent_amount == 0
            and requested_amount
            and requested_amount > 0
        ):
            spent_amount = requested_amount  # 用请求的卖出量作为估算
            logger.warning(
                f"订单 {order_id} 未返回 filled, 使用请求值 {format_decimal(spent_amount)} {spent_currency} 估算。"
            )

        # 确保非负
        received_amount = (
            max(Decimal("0"), received_amount)
            if received_amount is not None
            else Decimal("0")
        )
        spent_amount = (
            max(Decimal("0"), spent_amount)
            if spent_amount is not None
            else Decimal("0")
        )
        fee_amount = (
            max(Decimal("0"), fee_amount) if fee_amount is not None else Decimal("0")
        )

        # 如果之前解析失败，但这里有些值，状态可能不是完全错误
        if status == "error" and (received_amount > 0 or spent_amount > 0):
            status = "partial_ok"  # 标记为部分成功
            message = f"订单 {order_id} 状态 '{order_status}' 或数据解析不完整，但提取到部分数据。"
        elif status != "partial_ok":  # 如果没被标记为部分成功，且没错误，则为 ok
            status = "ok"
            message = f"成功执行 {side} on {order_info.get('symbol', 'N/A')}. Order ID: {order_id}"

        logger.info(f"[解析结果] {message}")
        logger.info(
            f"[解析结果] 花费: {format_decimal(spent_amount)} {spent_currency}, 收到: {format_decimal(received_amount)} {received_currency}, 手续费: {format_decimal(fee_amount)} {fee_currency or 'N/A'}"
        )

    except Exception as e:
        logger.error(
            f"[解析结果] 解析订单结果时出错 (Order ID: {order_id}): {e}", exc_info=True
        )
        status = "error"  # 确保是错误状态
        message = f"解析订单结果时出错: {e}"
        # 保留尽可能多的原始信息或请求信息
        spent_amount = requested_amount if side == "sell" else requested_cost
        spent_currency = base_currency if side == "sell" else quote_currency
        received_amount = Decimal("0")  # 无法确定收到多少
        received_currency = quote_currency if side == "sell" else base_currency

    return {
        "status": status,
        "message": message,
        "order_id": order_id,
        "side": side,
        "symbol": order_info.get("symbol"),
        "spent_amount": spent_amount or Decimal("0"),  # 确保有默认值
        "spent_currency": spent_currency,
        "received_amount": received_amount or Decimal("0"),  # 确保有默认值
        "received_currency": received_currency,
        "average_price": average_price,  # <--- 新增
        "expected_price": expected_price,  # <--- 新增 (传入的值)
        "slippage_percent": slippage_percent,  # <--- 新增
        "fee_amount": fee_amount or Decimal("0"),  # 确保有默认值
        "fee_currency": fee_currency,
        "timestamp": now,
        "raw_order_info": order_info,  # 包含原始订单信息供调试
    }


# --- 真实交易执行函数 (市价单 - 增强版，传入预期价格) ---
async def execute_real_market_sell(
    exchange: ccxtpro.Exchange,
    symbol: str,
    amount_to_sell: Decimal,
    markets: dict,
    expected_price: Decimal = None,  # <--- 新增
    max_retries: int = 0,
    retry_delay_sec: float = 1.0,
):
    """(真实交易) 执行市价卖出订单，传入预期价格，返回标准化结果。"""
    logger.info(
        f"[交易执行] 请求: 市价卖出 {format_decimal(amount_to_sell)} on {symbol} (预期价: {format_decimal(expected_price) if expected_price else 'N/A'})"
    )
    market = markets.get(symbol)
    if not market:
        return {"status": "error", "message": f"市场 {symbol} 未找到"}
    if not exchange.has.get("createMarketSellOrder"):
        return {"status": "error", "message": f"交易所不支持市价卖单 for {symbol}"}

    base_currency = market["base"]
    quote_currency = market["quote"]
    formatted_amount = amount_to_sell  # 先用原始值

    # 精度和限制检查
    try:
        # 格式化数量以满足交易所精度要求
        formatted_amount_str = exchange.amount_to_precision(
            symbol, float(amount_to_sell)
        )
        formatted_amount = Decimal(formatted_amount_str)
        if formatted_amount != amount_to_sell:
            logger.debug(
                f"卖出数量精度调整: {amount_to_sell} -> {formatted_amount} {base_currency}"
            )

        min_amount_limit = market.get("limits", {}).get("amount", {}).get("min")
        if min_amount_limit is not None and formatted_amount < Decimal(
            str(min_amount_limit)
        ):
            msg = f"卖出数量 {format_decimal(formatted_amount)} {base_currency} 低于市场最小要求 {min_amount_limit} {base_currency} for {symbol}"
            logger.error(f"[交易执行] 错误: {msg}")
            return {"status": "error", "message": msg}

    except Exception as e:
        logger.error(f"[交易执行] 准备卖单时出错 ({symbol}): {e}", exc_info=True)
        return {"status": "error", "message": f"准备卖单时出错: {e}"}

    attempt = 0
    last_exception = None
    order_info = None

    while attempt <= max_retries:
        attempt += 1
        logger.info(
            f"[交易执行] 尝试 #{attempt} (卖出 {symbol} - {format_decimal(formatted_amount)} {base_currency})..."
        )
        try:
            order_info = await exchange.create_market_sell_order(
                symbol, float(formatted_amount)
            )
            logger.info(
                f"[交易执行] 市价卖出订单提交成功 on {symbol}. Order ID: {order_info.get('id', 'N/A')}"
            )
            last_exception = None
            break  # 成功则跳出循环
        except (
            CCXTNetworkError,
            RequestTimeout,
            ExchangeNotAvailable,
            RateLimitExceeded,
        ) as e:
            last_exception = e
            logger.warning(
                f"[交易执行] 尝试 #{attempt} 失败 (卖出 {symbol}): {type(e).__name__} - {e}"
            )
            if attempt >= max_retries:  # 注意这里是 >=
                logger.error(f"卖出 {symbol} 达到最大重试次数 {max_retries+1}，放弃。")
                break
            await asyncio.sleep(
                retry_delay_sec * (1 + random.uniform(-0.2, 0.2))
            )  # 加入随机避免同时重试
        except InsufficientFunds as e:
            last_exception = e
            logger.error(f"[交易执行] 资金不足 (卖出 {symbol}): {e}")
            break
        except InvalidOrder as e:
            last_exception = e
            logger.error(f"[交易执行] 无效订单 (卖出 {symbol}): {e}")
            break
        except Exception as e:
            last_exception = e
            logger.error(f"[交易执行] 未知错误 (卖出 {symbol}): {e}", exc_info=True)
            break

    if order_info:
        # --- 调用增强版解析函数，传入预期价格 ---
        return parse_order_result(
            order_info,
            base_currency,
            quote_currency,
            "sell",
            expected_price=expected_price,  # <--- 传递预期价格
            requested_amount=formatted_amount,
        )
    else:
        msg = f"市价卖出失败 ({symbol})" + (
            f": {type(last_exception).__name__} - {last_exception}"
            if last_exception
            else ": 未知原因或未尝试"
        )
        logger.error(f"[交易执行] {msg}")
        # 返回包含预期价格的错误信息
        return {
            "status": "error",
            "message": msg,
            "symbol": symbol,
            "side": "sell",
            "expected_price": expected_price,
            "requested_amount": formatted_amount,
        }


async def execute_real_market_buy(
    exchange: ccxtpro.Exchange,
    symbol: str,
    markets: dict,
    expected_price: Decimal = None,  # <--- 新增
    amount_to_buy: Decimal = None,
    cost_to_spend: Decimal = None,
    max_retries: int = 0,
    retry_delay_sec: float = 1.0,
):
    """
    (真实交易 - 增强版) 执行市价买入订单，传入预期价格，返回标准化结果。
    优先使用 cost_to_spend (花费多少 quote 购买)。如果 cost_to_spend 为 None，则使用 amount_to_buy (购买多少 base)。
    """
    log_msg_parts = [f"[交易执行] 请求: 市价买入 on {symbol}"]
    if amount_to_buy:
        log_msg_parts.append(f"目标买入 {format_decimal(amount_to_buy)}")
    if cost_to_spend:
        log_msg_parts.append(f"花费 {format_decimal(cost_to_spend)}")
    log_msg_parts.append(
        f"(预期价: {format_decimal(expected_price) if expected_price else 'N/A'})"
    )
    logger.info(" ".join(log_msg_parts))

    market = markets.get(symbol)
    if not market:
        return {"status": "error", "message": f"市场 {symbol} 未找到"}
    # 检查是否支持市价单 (至少一种方式)
    has_market_buy = exchange.has.get("createMarketBuyOrder", False)
    # 注意：CCXT 的 has 属性可能不区分 cost 参数，主要看 createMarketBuyOrder 是否存在
    # 币安明确支持 quoteOrderQty，通过 params 传递
    # if not has_market_buy:
    #     return {'status': 'error', 'message': f'交易所不支持市价买单 for {symbol}'}

    base_currency = market["base"]
    quote_currency = market["quote"]
    # 决定是否使用按金额购买 (quoteOrderQty)
    use_quote_qty = (
        config.get("use_quote_order_qty_for_buy", True)
        and cost_to_spend is not None
        and cost_to_spend > 0
    )

    params = {}  # 用于传递 quoteOrderQty
    amount_param = None  # 传递给 create_market_buy_order 的 amount 参数
    formatted_value = None  # 用于记录请求数量或金额

    # --- 准备参数和检查限制 ---
    try:
        if use_quote_qty:
            # 检查最小成本限制
            min_cost_limit = market.get("limits", {}).get("cost", {}).get("min")
            cost_to_spend_precise = cost_to_spend  # 使用原始精度进行比较
            if min_cost_limit is not None and cost_to_spend_precise < Decimal(
                str(min_cost_limit)
            ):
                msg = f"请求花费 {format_decimal(cost_to_spend_precise)} {quote_currency} 低于市场最小成本 {min_cost_limit} {quote_currency} for {symbol}"
                logger.error(f"[交易执行] 错误: {msg}")
                return {"status": "error", "message": msg}
            # 格式化花费金额（通常币安对 quoteOrderQty 精度要求不高，但最好处理一下）
            # 币安 quoteOrderQty 似乎不需要 price 精度，但需要 cost 精度（如果有）
            # cost_precision = market.get('precision', {}).get('cost') # 或 quotePrecision? 查文档
            # formatted_cost_str = exchange.decimal_to_precision(float(cost_to_spend), rounding_mode=ROUND_DOWN, precision=cost_precision) if cost_precision else str(cost_to_spend)
            # params = {'quoteOrderQty': float(formatted_cost_str)}
            params = {
                "quoteOrderQty": float(cost_to_spend)
            }  # 直接使用 float，让 ccxt 处理
            amount_param = None  # 使用 quoteOrderQty 时，amount 必须是 None
            formatted_value = cost_to_spend  # 记录请求花费
            logger.info(
                f"准备使用 quoteOrderQty 方式买入，花费 {cost_to_spend} {quote_currency}"
            )

        elif amount_to_buy is not None and amount_to_buy > 0:
            # 使用按数量购买
            formatted_amount_str = exchange.amount_to_precision(
                symbol, float(amount_to_buy)
            )
            formatted_amount = Decimal(formatted_amount_str)
            if formatted_amount != amount_to_buy:
                logger.debug(
                    f"买入数量精度调整: {amount_to_buy} -> {formatted_amount} {base_currency}"
                )

            min_amount_limit = market.get("limits", {}).get("amount", {}).get("min")
            if min_amount_limit is not None and formatted_amount < Decimal(
                str(min_amount_limit)
            ):
                msg = f"买入数量 {format_decimal(formatted_amount)} {base_currency} 低于市场最小要求 {min_amount_limit} {base_currency} for {symbol}"
                logger.error(f"[交易执行] 错误: {msg}")
                return {"status": "error", "message": msg}

            amount_param = float(formatted_amount)  # 传递给函数的 amount
            formatted_value = formatted_amount  # 记录请求数量
            logger.info(
                f"准备使用 amount 方式买入，目标 {formatted_amount} {base_currency}"
            )
        else:
            return {
                "status": "error",
                "message": "买入订单必须提供有效的 amount_to_buy 或 cost_to_spend",
            }

    except Exception as e:
        logger.error(f"[交易执行] 准备买单时出错 ({symbol}): {e}", exc_info=True)
        return {"status": "error", "message": f"准备买单时出错: {e}"}

    attempt = 0
    last_exception = None
    order_info = None

    while attempt <= max_retries:
        attempt += 1
        log_attempt_detail = (
            f"花费 {cost_to_spend}" if use_quote_qty else f"买入 {formatted_value}"
        )
        logger.info(
            f"[交易执行] 尝试 #{attempt} (买入 {symbol} - {log_attempt_detail})..."
        )
        try:
            # 调用 ccxt 函数，传入 amount 和 params
            # 如果 use_quote_qty=True, amount_param 为 None, params={'quoteOrderQty': ...}
            # 如果 use_quote_qty=False, amount_param 为 float, params={}
            order_info = await exchange.create_market_buy_order(
                symbol, amount_param, params
            )
            logger.info(
                f"[交易执行] 市价买入订单提交成功 on {symbol}. Order ID: {order_info.get('id', 'N/A')}"
            )
            last_exception = None
            break  # 成功则跳出循环
        except (
            CCXTNetworkError,
            RequestTimeout,
            ExchangeNotAvailable,
            RateLimitExceeded,
        ) as e:
            last_exception = e
            logger.warning(
                f"[交易执行] 尝试 #{attempt} 失败 (买入 {symbol}): {type(e).__name__} - {e}"
            )
            if attempt >= max_retries:
                logger.error(f"买入 {symbol} 达到最大重试次数 {max_retries+1}，放弃。")
                break
            await asyncio.sleep(retry_delay_sec * (1 + random.uniform(-0.2, 0.2)))
        except ArgumentsRequired as e:
            last_exception = e
            logger.error(
                f"[交易执行] 参数错误 (买入 {symbol}): {e}. 检查交易所是否支持所选方式及参数。"
            )
            break
        except InsufficientFunds as e:
            last_exception = e
            logger.error(f"[交易执行] 资金不足 (买入 {symbol}): {e}")
            break
        except InvalidOrder as e:
            last_exception = e
            logger.error(f"[交易执行] 无效订单 (买入 {symbol}): {e}")
            break
        except Exception as e:
            last_exception = e
            logger.error(f"[交易执行] 未知错误 (买入 {symbol}): {e}", exc_info=True)
            break

    if order_info:
        # --- 调用增强版解析函数，传入预期价格和请求值 ---
        req_amount_for_parse = formatted_value if not use_quote_qty else None
        req_cost_for_parse = formatted_value if use_quote_qty else None
        return parse_order_result(
            order_info,
            base_currency,
            quote_currency,
            "buy",
            expected_price=expected_price,  # <--- 传递预期价格
            requested_amount=req_amount_for_parse,
            requested_cost=req_cost_for_parse,
        )
    else:
        msg = f"市价买入失败 ({symbol})" + (
            f": {type(last_exception).__name__} - {last_exception}"
            if last_exception
            else ": 未知原因或未尝试"
        )
        logger.error(f"[交易执行] {msg}")
        # 返回包含预期价格的错误信息
        req_amount = formatted_value if not use_quote_qty else None
        req_cost = formatted_value if use_quote_qty else None
        return {
            "status": "error",
            "message": msg,
            "symbol": symbol,
            "side": "buy",
            "expected_price": expected_price,
            "requested_amount": req_amount,
            "requested_cost": req_cost,
        }


# --- 真实闪兑执行函数 (调用增强版市价单) ---
async def execute_real_swap(
    exchange: ccxtpro.Exchange,
    from_currency: str,
    to_currency: str,
    from_amount: Decimal,
    markets: dict,
    current_tickers_snapshot: dict,  # <--- 新增：传入 Ticker 快照获取预期价
    max_retries: int = 1,
    retry_delay_sec: float = 1.0,
):
    """
    (真实交易 - 增强版) 使用市价单执行货币间的闪兑/转换，并返回标准化结果。
    使用传入的 Ticker 快照来确定预期价格。
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    logger.info(
        f"[闪兑执行] 请求: {format_decimal(from_amount)} {from_currency} -> {to_currency}"
    )

    if from_currency == to_currency:
        logger.warning("[闪兑执行] 源货币和目标货币相同，无需转换。")
        # 返回一个模拟成功的状态，表示无需操作
        return {
            "status": "ok",
            "message": "源货币和目标货币相同",
            "order_id": None,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "requested_from_amount": from_amount,
            "executed_from_amount": from_amount,
            "received_to_amount": from_amount,
            "average_price": Decimal("1.0"),  # 象征性价格
            "expected_price": Decimal("1.0"),
            "slippage_percent": Decimal("0"),
            "fee_amount": Decimal("0"),
            "fee_currency": None,
            "timestamp": time.time(),
        }

    if from_amount <= 0:
        logger.error("[闪兑执行] 错误: 转换金额必须为正数。")
        return {
            "status": "error",
            "message": "转换金额必须为正数。",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "requested_from_amount": from_amount,
        }

    result = None
    symbol_buy_target = (
        f"{to_currency}/{from_currency}"  # 例如 USDC/USDT - 买 USDC 花 USDT
    )
    symbol_sell_target = (
        f"{from_currency}/{to_currency}"  # 例如 USDT/USDC - 卖 USDT 换 USDC
    )

    # --- 尝试策略 1: 买入目标货币 (花费源货币) ---
    if symbol_buy_target in markets:
        logger.info(
            f"[闪兑执行] 找到交易对 {symbol_buy_target}. 尝试市价买入 {to_currency} 花费 {format_decimal(from_amount)} {from_currency}."
        )
        # 从快照获取预期买价 (Ask)
        expected_buy_price = None
        ticker_snap = current_tickers_snapshot.get(symbol_buy_target)
        if (
            ticker_snap
            and ticker_snap.get("ask")
            and Decimal(str(ticker_snap["ask"])) > 0
        ):
            expected_buy_price = Decimal(str(ticker_snap["ask"]))
            logger.debug(f"  预期买价 (来自快照): {format_decimal(expected_buy_price)}")
        else:
            logger.warning(
                f"  警告: 无法从 Ticker 快照获取 {symbol_buy_target} 的有效预期买价。"
            )

        # --- 调用增强版买入函数 ---
        buy_result = await execute_real_market_buy(
            exchange,
            symbol_buy_target,
            markets=markets,
            expected_price=expected_buy_price,  # <--- 传入预期价
            cost_to_spend=from_amount,  # 按花费金额购买
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
        )

        if buy_result["status"] in ("ok", "partial_ok"):
            result = {  # 调整字段名以匹配 swap 的期望输出，并包含价格/滑点信息
                "status": buy_result["status"],  # 传递状态
                "message": f"通过买入 {symbol_buy_target} 成功: {buy_result['message']}",
                "order_id": buy_result["order_id"],
                "from_currency": from_currency,
                "to_currency": to_currency,
                "requested_from_amount": from_amount,
                "executed_from_amount": buy_result[
                    "spent_amount"
                ],  # 买单实际花费的 from_currency
                "received_to_amount": buy_result[
                    "received_amount"
                ],  # 买单实际收到的 to_currency
                "average_price": buy_result["average_price"],  # <--- 新增
                "expected_price": buy_result["expected_price"],  # <--- 新增
                "slippage_percent": buy_result["slippage_percent"],  # <--- 新增
                "fee_amount": buy_result["fee_amount"],
                "fee_currency": buy_result["fee_currency"],
                "timestamp": buy_result["timestamp"],
            }
            logger.info(
                f"[闪兑执行] 买入 {symbol_buy_target} 完成. 均价: {format_decimal(result['average_price']) if result['average_price'] else 'N/A'}, 滑点: {f'{result["slippage_percent"]:.4f}%' if result['slippage_percent'] is not None else 'N/A'}"
            )
        else:
            logger.warning(
                f"[闪兑执行] 尝试买入 {symbol_buy_target} 失败: {buy_result['message']}"
            )
            # 不立即返回失败，尝试策略 2

    # --- 尝试策略 2: 卖出源货币 (换取目标货币) ---
    if not result and symbol_sell_target in markets:
        logger.info(
            f"[闪兑执行] 策略 1 失败或不适用. 尝试策略 2: 找到交易对 {symbol_sell_target}. 尝试市价卖出 {format_decimal(from_amount)} {from_currency}."
        )
        # 从快照获取预期卖价 (Bid)
        expected_sell_price = None
        ticker_snap = current_tickers_snapshot.get(symbol_sell_target)
        if (
            ticker_snap
            and ticker_snap.get("bid")
            and Decimal(str(ticker_snap["bid"])) > 0
        ):
            expected_sell_price = Decimal(str(ticker_snap["bid"]))
            logger.debug(
                f"  预期卖价 (来自快照): {format_decimal(expected_sell_price)}"
            )
        else:
            logger.warning(
                f"  警告: 无法从 Ticker 快照获取 {symbol_sell_target} 的有效预期卖价。"
            )

        # --- 调用增强版卖出函数 ---
        sell_result = await execute_real_market_sell(
            exchange,
            symbol_sell_target,
            amount_to_sell=from_amount,
            markets=markets,
            expected_price=expected_sell_price,  # <--- 传入预期价
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
        )

        if sell_result["status"] in ("ok", "partial_ok"):
            result = {  # 调整字段名并包含价格/滑点信息
                "status": sell_result["status"],
                "message": f"通过卖出 {symbol_sell_target} 成功: {sell_result['message']}",
                "order_id": sell_result["order_id"],
                "from_currency": from_currency,
                "to_currency": to_currency,
                "requested_from_amount": from_amount,
                "executed_from_amount": sell_result[
                    "spent_amount"
                ],  # 卖单实际花费的 from_currency
                "received_to_amount": sell_result[
                    "received_amount"
                ],  # 卖单实际收到的 to_currency
                "average_price": sell_result["average_price"],  # <--- 新增
                "expected_price": sell_result["expected_price"],  # <--- 新增
                "slippage_percent": sell_result["slippage_percent"],  # <--- 新增
                "fee_amount": sell_result["fee_amount"],
                "fee_currency": sell_result["fee_currency"],
                "timestamp": sell_result["timestamp"],
            }
            logger.info(
                f"[闪兑执行] 卖出 {symbol_sell_target} 完成. 均价: {format_decimal(result['average_price']) if result['average_price'] else 'N/A'}, 滑点: {f'{result["slippage_percent"]:.4f}%' if result['slippage_percent'] is not None else 'N/A'}"
            )
        else:
            logger.warning(
                f"[闪兑执行] 尝试卖出 {symbol_sell_target} 失败: {sell_result['message']}"
            )
            # 如果策略 2 也失败，则最终失败

    # --- 最终结果 ---
    if result:
        logger.info(
            f"[闪兑执行] 成功: 花费约 {format_decimal(result['executed_from_amount'])} {from_currency}, 收到约 {format_decimal(result['received_to_amount'])} {to_currency}"
        )
        return result
    else:
        final_message = (
            f"无法找到合适的交易对或执行出错来转换 {from_currency} -> {to_currency}"
        )
        logger.error(f"[闪兑执行] 失败: {final_message}")
        return {
            "status": "error",
            "message": final_message,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "requested_from_amount": from_amount,
        }


# --- 核心自动套利执行器 (增强版 - 捕获价格并生成详细报告) ---
async def execute_arbitrage_path(
    exchange: ccxtpro.Exchange,
    cycle_info: dict,
    markets: dict,
    application: Application,
):
    """
    (核心函数 - v2.2) 执行完整的套利路径，捕获预期价格，计算滑点，并生成包含余额变化的详细报告。

    Args:
        exchange: ccxtpro.Exchange 实例.
        cycle_info: 套利路径信息 (字典，包含 'nodes' 和 'trades' 列表).
        markets: 市场数据 (字典).
        application: Telegram Application 实例 (用于发送消息).
    """
    global global_balances, user_chat_id, is_trading_active, config, global_tickers, logger
    global current_execution_task
    global last_execution_duration_g

    path_str = " -> ".join(cycle_info.get("nodes", ["?"]))  # 安全获取节点信息
    cycle_start_node = cycle_info.get("nodes", ["?"])[0]
    final_report_sections = collections.defaultdict(list)  # 用于分段存储报告信息
    execution_step_details = []  # 存储每一步核心交易的详细结果
    initial_swap_details = None  # 存储初始闪兑的详细结果
    final_swap_details = None  # 存储最终闪兑的详细结果

    # --- 初始化用于记录初始资金的变量 ---
    initial_funding_currency = ""  # 记录最初动用的是哪个币种 (例如 USDT)
    initial_funding_amount = Decimal("0")  # 记录最初动用了多少

    # --- 其他初始化 ---
    actual_start_amount_used = Decimal(
        "0"
    )  # 闪兑后实际用于路径起始的金额 (cycle_start_node 计价)
    initial_swap_executed = False
    trade_successful = True  # 假设成功，过程中失败则设为 False
    current_amount = Decimal("0")
    current_currency = ""
    task_start_time = time.time()  # 记录任务开始时间

    # --- 用于记录余额变化的变量 ---
    involved_currencies = set(
        cycle_info.get("nodes", [])
    )  # 获取路径中的所有节点货币 (使用 set 去重)
    initial_involved_balances = {}  # 存储期初余额
    final_involved_balances = {}  # 存储期末余额
    # --- 结束 ---

    # --- 清理全局任务引用的辅助函数 ---
    def _clear_global_task_reference():
        global current_execution_task
        try:
            current_task_obj = asyncio.current_task()
            # 检查当前任务是否就是全局变量引用的那个任务
            if current_execution_task is current_task_obj:
                current_execution_task = None
                logger.info(
                    f"全局执行任务引用已清除 (Task: {current_task_obj.get_name()})"
                )
            elif current_execution_task is None:
                # 可能已经被其他地方清除了，也OK
                logger.info(
                    f"全局执行任务引用已被清除 (Task: {current_task_obj.get_name()})"
                )
            else:
                # 全局变量指向了别的任务，或者本任务不是预期的执行任务
                logger.warning(
                    f"任务 '{current_task_obj.get_name()}' 完成，但全局引用指向其他任务 ('{current_execution_task.get_name() if current_execution_task else 'None'}')。未清除全局引用。"
                )
        except Exception as clear_e:
            logger.error(f"清除全局任务引用时发生内部错误: {clear_e}", exc_info=True)

    # --- 辅助函数结束 ---

    acquired_lock = False
    try:
        # --- 1. 获取交易锁 ---
        if not await is_trading_active.acquire():
            logger.warning(f"交易锁已被占用，跳过执行请求: {path_str}")
            # 没有获取到锁，不需要（也不能）释放锁，也不需要清理全局引用
            return
        acquired_lock = True
        logger.info(
            f"--- [执行开始] 获得交易锁，开始执行路径: {path_str} (目标: 最终持有 USDT) ---"
        )

        # --- 2. 捕获执行开始时的 Ticker 快照 ---
        current_tickers_snapshot = global_tickers.copy()
        if not current_tickers_snapshot:
            msg = f"无法获取 Ticker 快照，无法确定预期价格。路径: {path_str}"
            logger.error(f"[执行中止] {msg}")
            if user_chat_id:
                await application.bot.send_message(
                    chat_id=user_chat_id, text=f"❌ 套利中止: 无法获取价格快照。"
                )
            is_trading_active.release()  # 确保释放锁
            logger.info(f"--- [执行中止] 交易锁已释放 (无价格快照) ---")
            _clear_global_task_reference()  # 清理引用
            return

        # --- 3. 记录初始相关余额 ---
        involved_currencies.add(
            "BNB"
        )  # 假设 BNB 可能用于支付手续费 (币安常见)，总是追踪它
        # 也可以根据配置的 fee_currency 来添加，但 BNB 最常见
        logger.debug(f"将追踪以下货币的余额变化: {involved_currencies}")
        initial_balances_snapshot = global_balances.copy()  # 获取当前全局余额快照
        for curr in involved_currencies:
            # 使用 format_decimal 确保存储的是格式化后的字符串或错误标记
            initial_involved_balances[curr] = initial_balances_snapshot.get(
                curr, Decimal("0")
            )
        logger.info(
            f"记录初始相关余额: { {c: format_decimal(b) for c, b in initial_involved_balances.items()} }"
        )
        # --- 记录结束 ---

        # --- 开始核心执行逻辑 ---
        try:
            # --- 4. 检查余额、计算需求、执行初始闪兑 ---
            min_start_usd_equiv = config.get(
                "min_trade_amount_usd_equivalent", Decimal("6.0")
            )
            stablecoin_prefs = config.get("stablecoin_preference", ["USDT", "USDC"])

            # 检查可用稳定币余额
            current_balance_map = initial_balances_snapshot  # 使用快照进行计算
            available_stables = {
                s: current_balance_map.get(s, Decimal("0"))
                for s in stablecoin_prefs
                if current_balance_map.get(s, Decimal("0")) > 0
            }
            total_stable_usd_value = sum(available_stables.values())  # 简单加总近似

            logger.info("[执行] 评估可用稳定币余额...")
            if total_stable_usd_value < min_start_usd_equiv:
                msg = f"可用稳定币总价值 (${format_decimal(total_stable_usd_value, 2)}) 低于最低交易要求 (${min_start_usd_equiv})。"
                logger.error(f"[执行中止] {msg} Path: {path_str}")
                if user_chat_id:
                    await application.bot.send_message(
                        chat_id=user_chat_id, text=f"❌ 套利中止: {msg}"
                    )
                trade_successful = False  # 标记失败

            # --- 5. 计算所需 cycle_start_node 数量 (如果未失败) ---
            target_total_amount_primary = Decimal("0")
            if trade_successful:
                price_primary_ask_vs_usdt = None
                try:
                    if cycle_start_node == "USDT":
                        price_primary_ask_vs_usdt = Decimal("1.0")
                    else:
                        # 从快照中获取价格
                        ticker_primary_usdt = f"{cycle_start_node}/USDT"
                        ticker_usdt_primary = f"USDT/{cycle_start_node}"
                        snap_ticker_fwd = current_tickers_snapshot.get(
                            ticker_primary_usdt
                        )
                        snap_ticker_rev = current_tickers_snapshot.get(
                            ticker_usdt_primary
                        )

                        if snap_ticker_fwd and snap_ticker_fwd.get("ask"):
                            ask_fwd = Decimal(str(snap_ticker_fwd["ask"]))
                            if ask_fwd > 0:
                                price_primary_ask_vs_usdt = ask_fwd
                        # 仅在正向找不到或无效时才尝试反向
                        if (
                            price_primary_ask_vs_usdt is None
                            and snap_ticker_rev
                            and snap_ticker_rev.get("bid")
                        ):
                            bid_rev = Decimal(str(snap_ticker_rev["bid"]))
                            if bid_rev > 0:
                                price_primary_ask_vs_usdt = Decimal("1.0") / bid_rev

                    if price_primary_ask_vs_usdt and price_primary_ask_vs_usdt > 0:
                        target_usd_value_for_calc = min_start_usd_equiv * Decimal(
                            "1.05"
                        )  # 增加 5% buffer
                        target_total_amount_primary = (
                            target_usd_value_for_calc / price_primary_ask_vs_usdt
                        )
                        logger.info(
                            f"[执行] 目标需要约 {format_decimal(target_total_amount_primary)} {cycle_start_node} (基于 ${target_usd_value_for_calc:.2f} 和预期买价 {format_decimal(price_primary_ask_vs_usdt)} USDT from snapshot)"
                        )
                    else:
                        raise ValueError(
                            f"无法从快照估算 {cycle_start_node}/USDT 预期买价"
                        )
                except (ValueError, DecimalInvalidOperation, TypeError) as e:
                    msg = f"无法确定所需起始金额 ({cycle_start_node}): {e}"
                    logger.error(f"[执行中止] {msg} Path: {path_str}")
                    if user_chat_id:
                        await application.bot.send_message(
                            chat_id=user_chat_id, text=f"❌ 套利中止: {msg}"
                        )
                    trade_successful = False

            # --- 6. 计算缺口并执行初始闪兑 (如果需要且未失败) ---
            if trade_successful:
                current_primary_balance = current_balance_map.get(
                    cycle_start_node, Decimal("0")
                )
                amount_needed_via_swap = max(
                    Decimal("0"), target_total_amount_primary - current_primary_balance
                )

                if amount_needed_via_swap > 0:
                    logger.info(
                        f"[执行] 当前 {cycle_start_node} 余额 {format_decimal(current_primary_balance)} 不足，需闪兑约 {format_decimal(amount_needed_via_swap)} {cycle_start_node}"
                    )
                    # 选择闪兑来源稳定币
                    source_stable = None
                    for s in stablecoin_prefs:
                        if s != cycle_start_node and available_stables.get(s, 0) > 0:
                            source_stable = s
                            break  # 使用第一个可用的其他稳定币

                    if not source_stable:
                        msg = f"需要闪兑 ({format_decimal(amount_needed_via_swap)} {cycle_start_node})，但没有其他可用的稳定币来源 ({stablecoin_prefs})。"
                        logger.error(f"[执行中止] {msg} Path: {path_str}")
                        if user_chat_id:
                            await application.bot.send_message(
                                chat_id=user_chat_id, text=f"❌ 套利中止: {msg}"
                            )
                        trade_successful = False
                    else:
                        # 估算需要花费多少 source_stable (使用 Ticker 快照)
                        estimated_cost_of_source = Decimal("-1")
                        swap_symbol_buy = f"{cycle_start_node}/{source_stable}"  # 买 cycle_start_node 花 source_stable
                        ticker_snap = current_tickers_snapshot.get(swap_symbol_buy)
                        if ticker_snap and ticker_snap.get("ask"):
                            ask_price = Decimal(str(ticker_snap["ask"]))
                            if ask_price > 0:
                                # 需要 amount_needed_via_swap 的 cycle_start_node, 考虑2%buffer
                                estimated_cost_of_source = (
                                    amount_needed_via_swap * ask_price * Decimal("1.02")
                                )
                                logger.debug(
                                    f"  估算闪兑成本 (买入法): {format_decimal(estimated_cost_of_source)} {source_stable} @ {format_decimal(ask_price)}"
                                )

                        # 如果买入法失败，尝试卖出法
                        if estimated_cost_of_source <= 0:
                            swap_symbol_sell = f"{source_stable}/{cycle_start_node}"  # 卖 source_stable 换 cycle_start_node
                            ticker_snap_rev = current_tickers_snapshot.get(
                                swap_symbol_sell
                            )
                            if ticker_snap_rev and ticker_snap_rev.get("bid"):
                                bid_price = Decimal(str(ticker_snap_rev["bid"]))
                                if bid_price > 0:
                                    # 需要 amount_needed_via_swap 的 cycle_start_node
                                    estimated_cost_of_source = (
                                        amount_needed_via_swap / bid_price
                                    ) * Decimal(
                                        "1.02"
                                    )  # 加 2% buffer
                                    logger.debug(
                                        f"  估算闪兑成本 (卖出法): {format_decimal(estimated_cost_of_source)} {source_stable} @ {format_decimal(bid_price)}"
                                    )

                        if estimated_cost_of_source <= 0:
                            msg = f"无法从快照估算从 {source_stable} 闪兑到 {cycle_start_node} 的成本。"
                            logger.error(f"[执行中止] {msg} Path: {path_str}")
                            if user_chat_id:
                                await application.bot.send_message(
                                    chat_id=user_chat_id, text=f"❌ 套利中止: {msg}"
                                )
                            trade_successful = False
                        else:
                            source_balance = available_stables.get(
                                source_stable, Decimal("0")
                            )  # 再次确认余额
                            if source_balance < estimated_cost_of_source:
                                msg = f"{source_stable} 余额 ({format_decimal(source_balance)}) 不足以按估算 ({format_decimal(estimated_cost_of_source)}) 获取所需 {cycle_start_node}。"
                                logger.error(f"[执行中止] {msg} Path: {path_str}")
                                if user_chat_id:
                                    await application.bot.send_message(
                                        chat_id=user_chat_id, text=f"❌ 套利中止: {msg}"
                                    )
                                trade_successful = False
                            else:
                                # 确定实际用于闪兑的金额 (不超过余额)
                                amount_source_to_use = min(
                                    source_balance, estimated_cost_of_source
                                )
                                logger.info(
                                    f"[执行闪兑] 执行: 用 {format_decimal(amount_source_to_use)} {source_stable} 换取 {cycle_start_node}"
                                )

                                # --- 调用增强版闪兑，传入快照 ---
                                swap_result = await execute_real_swap(
                                    exchange,
                                    source_stable,
                                    cycle_start_node,
                                    amount_source_to_use,
                                    markets,
                                    current_tickers_snapshot,  # <--- 传入快照
                                    config["max_trade_retries"],
                                    config["trade_retry_delay_sec"],
                                )

                                initial_swap_details = (
                                    swap_result  # 存储结果，无论成功失败
                                )

                                if swap_result and swap_result.get("status") in (
                                    "ok",
                                    "partial_ok",
                                ):
                                    received = swap_result.get(
                                        "received_to_amount", Decimal("0")
                                    )
                                    cost = swap_result.get(
                                        "executed_from_amount", Decimal("0")
                                    )
                                    logger.info(
                                        f"[执行闪兑] {swap_result.get('message', '闪兑完成但消息缺失')}"
                                    )
                                    logger.info(
                                        f"[执行闪兑] 结果: 花费 {format_decimal(cost)} {source_stable}, 收到 {format_decimal(received)} {cycle_start_node}"
                                    )

                                    # 记录初始资金信息
                                    initial_funding_currency = swap_result.get(
                                        "from_currency", source_stable
                                    )
                                    initial_funding_amount = cost
                                    logger.info(
                                        f"记录初始资金: {initial_funding_amount} {initial_funding_currency}"
                                    )

                                    # 追踪手续费币种
                                    fee_curr_init = swap_result.get("fee_currency")
                                    if fee_curr_init:
                                        involved_currencies.add(fee_curr_init)

                                    current_primary_balance += received
                                    initial_swap_executed = True
                                else:
                                    msg = f"初始闪兑 ({source_stable} -> {cycle_start_node}) 失败: {swap_result.get('message') if swap_result else '未知错误'}"
                                    logger.error(f"[执行中止] {msg} Path: {path_str}")
                                    if user_chat_id:
                                        await application.bot.send_message(
                                            chat_id=user_chat_id,
                                            text=f"❌ 套利中止: {msg}",
                                        )
                                    trade_successful = False

                # --- 闪兑后检查 ---
                if trade_successful:
                    # 检查闪兑后余额是否足够 (允许一点点误差)
                    if current_primary_balance < target_total_amount_primary * Decimal(
                        "0.98"
                    ):  # 允许 2% 误差
                        msg = f"初始闪兑后，{cycle_start_node} 余额 ({format_decimal(current_primary_balance)}) 仍显著低于目标 ({format_decimal(target_total_amount_primary)})。"
                        logger.error(f"[执行中止] {msg} Path: {path_str}")
                        if user_chat_id:
                            await application.bot.send_message(
                                chat_id=user_chat_id, text=f"❌ 套利中止: {msg}"
                            )
                        trade_successful = False
                    else:
                        logger.info(
                            f"[执行] 闪兑后检查通过，当前 {cycle_start_node} 余额: {format_decimal(current_primary_balance)}"
                        )
                        actual_start_amount_used = (
                            current_primary_balance  # 确认用于环路的起始金额
                        )

            # --- 7. 处理无需初始闪兑的情况，并最终确认初始资金 (如果未失败) ---
            if trade_successful and not initial_swap_executed:
                logger.info(
                    f"[执行] {cycle_start_node} 余额 {format_decimal(current_primary_balance)} 已满足要求，无需初始闪兑。"
                )
                actual_start_amount_used = (
                    current_primary_balance  # 起始金额就是当前主货币余额
                )
                initial_funding_currency = cycle_start_node
                initial_funding_amount = actual_start_amount_used
                logger.info(
                    f"无需初始闪兑，记录初始资金: {initial_funding_amount} {initial_funding_currency}"
                )

            # --- 8. 最终检查起始金额有效性 (如果未失败) ---
            if trade_successful:
                # 再次使用快照价格估算 actual_start_amount_used 的 USD 价值
                start_usd_value_est = Decimal("0")
                if cycle_start_node == "USDT":
                    start_usd_value_est = actual_start_amount_used
                else:
                    # 尝试从快照获取估值价格 (bid price when selling cycle_start_node for USDT)
                    ticker_fwd = f"{cycle_start_node}/USDT"
                    ticker_rev = f"USDT/{cycle_start_node}"
                    snap_fwd = current_tickers_snapshot.get(ticker_fwd)
                    snap_rev = current_tickers_snapshot.get(ticker_rev)
                    valuation_price = None
                    if snap_fwd and snap_fwd.get("bid"):
                        bid_fwd = Decimal(str(snap_fwd["bid"]))
                        if bid_fwd > 0:
                            valuation_price = bid_fwd
                    if valuation_price is None and snap_rev and snap_rev.get("ask"):
                        ask_rev = Decimal(str(snap_rev["ask"]))
                        if ask_rev > 0:
                            valuation_price = Decimal("1.0") / ask_rev

                    if valuation_price:
                        start_usd_value_est = actual_start_amount_used * valuation_price
                    else:
                        logger.warning(
                            f"无法从快照获取 {cycle_start_node} 的美元估值价格。"
                        )

                # 检查最终确定的起始金额是否有效且大致满足最低要求 (允许 5% 偏差)
                if (
                    actual_start_amount_used <= 0
                    or start_usd_value_est < min_start_usd_equiv * Decimal("0.95")
                ):
                    msg = f"最终确定的起始金额 {format_decimal(actual_start_amount_used)} {cycle_start_node} (约 ${format_decimal(start_usd_value_est,2)}) 无效或低于要求 (${min_start_usd_equiv*Decimal('0.95'):.2f})。"
                    logger.error(f"[执行中止] {msg} Path: {path_str}")
                    if user_chat_id:
                        await application.bot.send_message(
                            chat_id=user_chat_id, text=f"❌ 套利中止: {msg}"
                        )
                    trade_successful = False
                else:
                    logger.info(
                        f"===> [执行] 实际路径起始 ({cycle_start_node}): {format_decimal(actual_start_amount_used, 8)} (估值约 ${format_decimal(start_usd_value_est,2)}) {'(由初始闪兑获得)' if initial_swap_executed else ''}"
                    )
                    current_amount = actual_start_amount_used
                    current_currency = cycle_start_node

            # --- 9. 按顺序执行套利路径交易 (如果未失败) ---
            if trade_successful:
                trades = cycle_info["trades"]
                num_trades_in_cycle = len(trades)

                for i, trade in enumerate(trades):
                    step_num = i + 1
                    pair = trade["pair"]
                    trade_type = trade["type"]
                    from_curr = trade["from"]
                    to_curr = trade["to"]

                    if current_currency != from_curr:
                        msg = f"逻辑错误: 步骤 {step_num} 需要发送 {from_curr}, 但当前持有 {current_currency}"
                        logger.error(f"[执行步骤 {step_num}] {msg}")
                        execution_step_details.append(
                            {"step": step_num, "status": "error", "message": msg}
                        )  # 记录错误
                        trade_successful = False
                        break

                    logger.info(
                        f"--- > [执行步骤 {step_num}/{num_trades_in_cycle}] {trade_type} on {pair} (持有: {format_decimal(current_amount, 8)} {current_currency})"
                    )

                    market_info = markets.get(pair)
                    if not market_info:
                        msg = f"市场 {pair} 未找到"
                        logger.error(f"[执行步骤 {step_num}] {msg}")
                        execution_step_details.append(
                            {"step": step_num, "status": "error", "message": msg}
                        )
                        trade_successful = False
                        break

                    # 从快照获取此步骤的预期价格
                    expected_step_price = None
                    ticker_snap = current_tickers_snapshot.get(pair)
                    if ticker_snap:
                        price_key = "ask" if trade_type == "BUY" else "bid"
                        price_val = ticker_snap.get(price_key)
                        if price_val:
                            try:
                                price_dec = Decimal(str(price_val))
                                if price_dec > 0:
                                    expected_step_price = price_dec
                            except (ValueError, DecimalInvalidOperation, TypeError):
                                pass  # 忽略无效价格
                    if expected_step_price is None:
                        logger.warning(
                            f"步骤 {step_num}: 无法从快照获取 {pair} 的有效预期 {trade_type} 价格。滑点将无法计算。"
                        )

                    order_result = None
                    try:
                        # --- 执行买卖订单 ---
                        if trade_type == "BUY":
                            if config.get(
                                "use_quote_order_qty_for_buy", True
                            ) and current_currency == market_info.get("quote"):
                                order_result = await execute_real_market_buy(
                                    exchange,
                                    pair,
                                    markets,
                                    expected_price=expected_step_price,  # 传入预期价
                                    cost_to_spend=current_amount,
                                    max_retries=config["max_trade_retries"],
                                    retry_delay_sec=config["trade_retry_delay_sec"],
                                )
                            else:  # 其他买入情况 (例如按数量买) - 这里可能需要更复杂的逻辑
                                msg = f"买入逻辑不支持或配置不符: BUY {pair} 需要 {market_info.get('quote')} 但持有 {current_currency} (或未配置 cost 购买)"
                                logger.error(f"[执行步骤 {step_num}] {msg}")
                                execution_step_details.append(
                                    {
                                        "step": step_num,
                                        "status": "error",
                                        "message": msg,
                                    }
                                )
                                trade_successful = False
                                break

                        elif trade_type == "SELL":
                            if current_currency == market_info.get("base"):
                                order_result = await execute_real_market_sell(
                                    exchange,
                                    pair,
                                    current_amount,
                                    markets,
                                    expected_price=expected_step_price,  # 传入预期价
                                    max_retries=config["max_trade_retries"],
                                    retry_delay_sec=config["trade_retry_delay_sec"],
                                )
                            else:
                                msg = f"卖出逻辑错误: SELL {pair} 需要 {market_info.get('base')} 但持有 {current_currency}"
                                logger.error(f"[执行步骤 {step_num}] {msg}")
                                execution_step_details.append(
                                    {
                                        "step": step_num,
                                        "status": "error",
                                        "message": msg,
                                    }
                                )
                                trade_successful = False
                                break
                        else:
                            msg = f"未知交易类型 {trade_type}"
                            logger.error(f"[执行步骤 {step_num}] {msg}")
                            execution_step_details.append(
                                {"step": step_num, "status": "error", "message": msg}
                            )
                            trade_successful = False
                            break

                        # --- 处理订单结果 ---
                        order_result = order_result or {}  # 确保 order_result 是字典
                        order_result["step"] = step_num  # 添加步骤号
                        execution_step_details.append(order_result)  # 存储详细结果

                        if order_result.get("status") in ("ok", "partial_ok"):
                            logger.info(
                                f"[执行步骤 {step_num}] {order_result.get('message', '订单完成但消息缺失')}"
                            )
                            avg_p = order_result.get("average_price")
                            slip_p = order_result.get("slippage_percent")
                            logger.info(
                                f"  实际均价: {format_decimal(avg_p) if avg_p else 'N/A'}, "
                                f"滑点: {f'{slip_p:.4f}%' if slip_p is not None else 'N/A'}"
                            )

                            current_amount = order_result.get(
                                "received_amount", Decimal("0")
                            )
                            current_currency = order_result.get("received_currency", "")

                            # 追踪手续费币种
                            fee_curr_step = order_result.get("fee_currency")
                            if fee_curr_step:
                                involved_currencies.add(fee_curr_step)

                            if current_amount <= 0:
                                msg = "交易后金额变为非正数"
                                logger.error(
                                    f"[执行步骤 {step_num}] {msg} ({current_amount})，中止套利。"
                                )
                                trade_successful = False
                                break  # break 内部循环
                        else:  # 订单失败
                            msg = f"执行失败: {order_result.get('message', '未知错误')}"
                            logger.error(
                                f"[执行步骤 {step_num}] {trade_type} {pair} {msg}"
                            )
                            trade_successful = False
                            break  # break 内部循环

                    except Exception as step_e:
                        msg = f"步骤 {step_num} ({trade_type} {pair}) 遇到意外错误: {step_e}"
                        logger.error(f"[执行步骤 {step_num}] {msg}", exc_info=True)
                        execution_step_details.append(
                            {"step": step_num, "status": "error", "message": msg}
                        )
                        trade_successful = False
                        break  # break 内部循环

            # --- 10. 执行最终闪兑回 USDT (如果需要且未失败) ---
            if trade_successful:
                logger.info(
                    f"--- > [执行] 核心路径完成，当前持有 {format_decimal(current_amount)} {current_currency}。"
                )
                if current_currency != "USDT":
                    logger.info(
                        f"--- > [执行最终闪兑] 需要将 {current_currency} 转换回 USDT..."
                    )
                    # ---> 调用增强版闪兑，传入快照 <---
                    final_swap_result = await execute_real_swap(
                        exchange,
                        current_currency,
                        "USDT",
                        current_amount,
                        markets,
                        current_tickers_snapshot,  # <--- 传入快照
                        config["max_trade_retries"],
                        config["trade_retry_delay_sec"],
                    )
                    final_swap_details = final_swap_result  # 存储最终闪兑结果

                    if final_swap_result and final_swap_result.get("status") in (
                        "ok",
                        "partial_ok",
                    ):
                        logger.info(
                            f"[执行最终闪兑] {final_swap_result.get('message', '闪兑完成但消息缺失')}"
                        )
                        fswap_spent = final_swap_result.get("executed_from_amount", 0)
                        fswap_rcvd = final_swap_result.get("received_to_amount", 0)
                        fswap_avg_p = final_swap_result.get("average_price")
                        fswap_slip = final_swap_result.get("slippage_percent")
                        logger.info(
                            f"  结果: 花费 {format_decimal(fswap_spent)} {current_currency}, "
                            f"收到 {format_decimal(fswap_rcvd)} USDT"
                        )
                        logger.info(
                            f"  实际均价: {format_decimal(fswap_avg_p) if fswap_avg_p else 'N/A'}, "
                            f"滑点: {f'{fswap_slip:.4f}%' if fswap_slip is not None else 'N/A'}"
                        )

                        # 更新当前金额和货币
                        current_amount = fswap_rcvd
                        current_currency = "USDT"

                        # 追踪手续费币种
                        fee_curr_final = final_swap_result.get("fee_currency")
                        if fee_curr_final:
                            involved_currencies.add(fee_curr_final)
                    else:
                        msg = f"最终闪兑 ({current_currency} -> USDT) 失败: {final_swap_result.get('message') if final_swap_result else '未知错误'}"
                        logger.error(f"[执行最终闪兑] {msg}")
                        # 最终闪兑失败，标记整个交易不完全成功
                        trade_successful = False
                        # 记录错误信息
                        fail_info = final_swap_result or {
                            "status": "error",
                            "message": msg,
                        }
                        fail_info["step"] = "F"  # 标记为最终步骤
                        # execution_step_details.append(fail_info) # 或者不加到核心步骤里
                else:
                    logger.info(f"--- > [执行] 最终货币已是 USDT，无需最终闪兑。")

        # --- 捕获执行逻辑中的顶层异常 ---
        except Exception as e:
            logger.error(
                f"[执行错误] 执行套利路径 {path_str} 时发生内部错误: {e}", exc_info=True
            )
            final_report_sections["错误"].append(
                f"执行过程中发生严重错误: {e}"
            )  # 记录错误信息
            trade_successful = False  # 标记失败

        # --- 11. finally 块：生成报告并释放锁 ---
        finally:
            if acquired_lock and is_trading_active.locked():  # 检查是否持有锁
                task_end_time = time.time()
                task_duration = task_end_time - task_start_time
                last_execution_duration_g = task_duration  # <--- 更新全局变量
                logger.info(
                    f"--- [执行结束] 路径: {path_str} 执行流程完成 (耗时: {task_duration:.3f} 秒) ---"
                )

                # --- 获取最终相关余额 ---
                # 注意：这里需要重新获取一次实时余额，而不是用任务开始时的快照
                try:
                    final_balances_live = await exchange.fetch_balance()
                    final_balances_free = final_balances_live.get("free", {})
                    for curr in involved_currencies:
                        final_involved_balances[curr] = Decimal(
                            str(final_balances_free.get(curr, "0"))
                        )
                    logger.info(
                        f"记录最终相关余额 (实时): { {c: format_decimal(b) for c, b in final_involved_balances.items()} }"
                    )
                except Exception as final_bal_e:
                    logger.error(
                        f"获取最终实时余额失败: {final_bal_e}. 余额变化报告可能不准确。"
                    )
                    # 使用任务结束时的全局余额作为后备
                    final_balances_snapshot_end = global_balances.copy()
                    for curr in involved_currencies:
                        final_involved_balances[curr] = final_balances_snapshot_end.get(
                            curr, Decimal("0")
                        )
                    logger.info(
                        f"记录最终相关余额 (后备快照): { {c: format_decimal(b) for c, b in final_involved_balances.items()} }"
                    )

                # --- 构建最终报告 ---
                final_report = f"**📊 套利执行报告 (目标: USDT)**\n\n"
                final_report += f"路径: `{path_str}`\n"
                final_report += (
                    f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                final_report += f"本次执行耗时: `{task_duration:.3f}` 秒\n"
                start_amount_report = (
                    format_decimal(initial_funding_amount, 8)
                    if initial_funding_amount > 0
                    else "未知"
                )
                start_currency_report = (
                    initial_funding_currency if initial_funding_currency else "未知来源"
                )
                final_report += (
                    f"初始投入 ({start_currency_report}): `{start_amount_report}`\n"
                )
                final_report += f"_(注: 这是本次套利路径**实际使用**的起始资金)_\n"

                # --- 初始闪兑详情 ---
                if initial_swap_details:
                    final_report += "\n**初始闪兑:**\n```\n"
                    iswap_from = initial_swap_details.get("from_currency", "?")
                    iswap_to = initial_swap_details.get("to_currency", "?")
                    iswap_spent = initial_swap_details.get(
                        "executed_from_amount", Decimal("0")
                    )
                    iswap_rcvd = initial_swap_details.get(
                        "received_to_amount", Decimal("0")
                    )
                    iswap_avg_p = initial_swap_details.get("average_price")
                    iswap_exp_p = initial_swap_details.get("expected_price")
                    iswap_slip = initial_swap_details.get("slippage_percent")
                    iswap_fee = initial_swap_details.get("fee_amount", Decimal("0"))
                    iswap_fee_c = initial_swap_details.get("fee_currency", "")
                    final_report += f"操作: {iswap_from} -> {iswap_to}\n"
                    final_report += (
                        f"状态: {initial_swap_details.get('status', '未知')}\n"
                    )
                    if initial_swap_details.get("status") in ("ok", "partial_ok"):
                        final_report += (
                            f"花费: {format_decimal(iswap_spent, 8)} {iswap_from}\n"
                        )
                        final_report += (
                            f"收到: {format_decimal(iswap_rcvd, 8)} {iswap_to}\n"
                        )
                        final_report += f"预期价: {format_decimal(iswap_exp_p) if iswap_exp_p else 'N/A'}\n"
                        final_report += f"实际均价: {format_decimal(iswap_avg_p) if iswap_avg_p else 'N/A'}\n"
                        final_report += f"滑点: {f'{iswap_slip:.4f}%' if iswap_slip is not None else 'N/A'}\n"
                        final_report += (
                            f"手续费: {format_decimal(iswap_fee, 8)} {iswap_fee_c}\n"
                        )
                    else:
                        final_report += f"失败信息: {initial_swap_details.get('message', '未知错误')}\n"
                    final_report += "```\n"

                # --- 核心交易步骤详情 ---
                if execution_step_details:
                    final_report += "\n**核心交易步骤:**\n"
                    (
                        col_step,
                        col_op,
                        col_pair,
                        col_exp,
                        col_avg,
                        col_slip,
                        col_spent,
                        col_rcvd,
                        col_fee,
                    ) = (4, 14, 12, 16, 16, 9, 28, 28, 20)
                    header = (
                        f"{'步':<{col_step}} {'操作':<{col_op}} {'交易对':<{col_pair}} "
                        f"{'预期价':<{col_exp}} {'实际均价':<{col_avg}} {'滑点%':<{col_slip}} "
                        f"{'花费':<{col_spent}} {'收到':<{col_rcvd}} {'手续费':<{col_fee}}"
                    )
                    separator = "-" * len(header)
                    final_report += f"<pre>\n{header}\n{separator}\n"
                    for step_detail in execution_step_details:
                        step = step_detail.get("step", "?")
                        status = step_detail.get("status", "未知")
                        if status == "error":
                            final_report += f"{str(step):<{col_step}} {'- 错误 -':<{col_op + col_pair + col_exp + col_avg + col_slip + col_spent + col_rcvd + col_fee + 8}}\n"
                            final_report += f"{'':<{col_step}} MSG: {step_detail.get('message', '未知错误')}\n"
                            continue
                        side = step_detail.get("side", "?").upper()
                        pair = step_detail.get("symbol", "?")
                        op_str = f"{side} {pair}"
                        exp_p = step_detail.get("expected_price")
                        avg_p = step_detail.get("average_price")
                        slip_p = step_detail.get("slippage_percent")
                        spent_a = step_detail.get("spent_amount", Decimal("0"))
                        spent_c = step_detail.get("spent_currency", "?")
                        rcvd_a = step_detail.get("received_amount", Decimal("0"))
                        rcvd_c = step_detail.get("received_currency", "?")
                        fee_a = step_detail.get("fee_amount", Decimal("0"))
                        fee_c = step_detail.get("fee_currency", "")
                        exp_p_str = (
                            format_decimal(exp_p) if exp_p is not None else "N/A"
                        )
                        avg_p_str = (
                            format_decimal(avg_p) if avg_p is not None else "N/A"
                        )
                        slip_p_str = f"{slip_p:+.4f}" if slip_p is not None else "N/A"
                        spent_str = f"{format_decimal(spent_a, 8)} {spent_c}"
                        rcvd_str = f"{format_decimal(rcvd_a, 8)} {rcvd_c}"
                        fee_str = f"{format_decimal(fee_a, 8)} {fee_c}"
                        line = (
                            f"{str(step):<{col_step}} {op_str:<{col_op}} {pair:<{col_pair}} "
                            f"{exp_p_str:<{col_exp}} {avg_p_str:<{col_avg}} {slip_p_str:<{col_slip}} "
                            f"{spent_str:<{col_spent}} {rcvd_str:<{col_rcvd}} {fee_str:<{col_fee}}"
                        )
                        final_report += f"{line}\n"
                    final_report += "</pre>\n"

                # --- 最终闪兑详情 ---
                if final_swap_details:
                    final_report += "\n**最终闪兑:**\n```\n"
                    fswap_from = final_swap_details.get("from_currency", "?")
                    fswap_to = final_swap_details.get("to_currency", "?")
                    fswap_spent = final_swap_details.get(
                        "executed_from_amount", Decimal("0")
                    )
                    fswap_rcvd = final_swap_details.get(
                        "received_to_amount", Decimal("0")
                    )
                    fswap_avg_p = final_swap_details.get("average_price")
                    fswap_exp_p = final_swap_details.get("expected_price")
                    fswap_slip = final_swap_details.get("slippage_percent")
                    fswap_fee = final_swap_details.get("fee_amount", Decimal("0"))
                    fswap_fee_c = final_swap_details.get("fee_currency", "")
                    final_report += f"操作: {fswap_from} -> {fswap_to}\n"
                    final_report += (
                        f"状态: {final_swap_details.get('status', '未知')}\n"
                    )
                    if final_swap_details.get("status") in ("ok", "partial_ok"):
                        final_report += (
                            f"花费: {format_decimal(fswap_spent, 8)} {fswap_from}\n"
                        )
                        final_report += (
                            f"收到: {format_decimal(fswap_rcvd, 8)} {fswap_to}\n"
                        )
                        final_report += f"预期价: {format_decimal(fswap_exp_p) if fswap_exp_p else 'N/A'}\n"
                        final_report += f"实际均价: {format_decimal(fswap_avg_p) if fswap_avg_p else 'N/A'}\n"
                        final_report += f"滑点: {f'{fswap_slip:.4f}%' if fswap_slip is not None else 'N/A'}\n"
                        final_report += (
                            f"手续费: {format_decimal(fswap_fee, 8)} {fswap_fee_c}\n"
                        )
                    else:
                        final_report += f"失败信息: {final_swap_details.get('message', '未知错误')}\n"
                    final_report += "```\n"

                # --- 报告涉及资产余额变化 ---
                final_report += "\n**涉及资产余额变化:**\n<pre>"
                bal_change_lines = []
                max_curr_len_bal = 0
                involved_currencies_sorted = sorted(
                    list(involved_currencies)
                )  # 排序以获得一致的顺序
                for curr in involved_currencies_sorted:
                    max_curr_len_bal = max(max_curr_len_bal, len(curr))
                for curr in involved_currencies_sorted:
                    initial_bal = initial_involved_balances.get(curr, Decimal("0"))
                    final_bal = final_involved_balances.get(curr, Decimal("0"))
                    change = final_bal - initial_bal
                    # 只显示有变化的行，或者设置一个阈值
                    if abs(change) > Decimal("1e-10"):  # 仅显示余额有显著变化的行
                        change_sign = "+" if change > 0 else ""
                        precision_change = 8 if abs(change) > Decimal("1e-7") else 10
                        line = (
                            f"{curr:<{max_curr_len_bal}} : "
                            f"{format_decimal(initial_bal, 8)} -> {format_decimal(final_bal, 8)} "
                            f"({change_sign}{format_decimal(change, precision_change)})"
                        )
                        bal_change_lines.append(line)
                if not bal_change_lines:  # 如果没有任何变化
                    bal_change_lines.append("(无显著余额变化)")
                final_report += "\n".join(bal_change_lines)
                final_report += "</pre>\n"
                # --- 余额变化报告结束 ---

                # --- 报告最终结果和路径利润 ---
                final_report += "\n**最终路径结果:**\n"
                if trade_successful and current_currency == "USDT":
                    final_usdt_amount = current_amount
                    profit_amount_usdt = Decimal("NaN")
                    profit_percent = Decimal("NaN")
                    if (
                        initial_funding_currency == "USDT"
                        and initial_funding_amount > 0
                    ):
                        profit_amount_usdt = final_usdt_amount - initial_funding_amount
                        profit_percent = (
                            profit_amount_usdt / initial_funding_amount
                        ) * 100
                        final_report += f"最终持有 ({current_currency}): `{format_decimal(final_usdt_amount, 8)}`\n"
                        if profit_amount_usdt > 0:
                            final_report += f"🟢 **路径净利润 (USDT):** `{format_decimal(profit_amount_usdt, 8)}` (`{profit_percent:+.4f}%`)\n"
                        else:
                            final_report += f"🔴 **路径净亏损 (USDT):** `{format_decimal(profit_amount_usdt, 8)}` (`{profit_percent:+.4f}%`)\n"
                        final_report += f"_(注: 利润/亏损基于上述'初始投入'计算)_\n"
                    elif initial_funding_amount > 0:
                        final_report += f"最终持有 ({current_currency}): `{format_decimal(final_usdt_amount, 8)}`\n"
                        final_report += f"ℹ️ 初始资金来自 `{initial_funding_currency}` (`{format_decimal(initial_funding_amount)}`)，仅报告 USDT 结果，未计算利润率。\n"
                    else:
                        final_report += f"最终持有 ({current_currency}): `{format_decimal(final_usdt_amount, 8)}`\n"
                        final_report += (
                            f"❌ 未能确定有效的初始资金信息，无法计算利润。\n"
                        )
                elif trade_successful and current_currency != "USDT":
                    final_report += f"🔴 **套利未完全成功 (最终闪兑失败)**\n"
                    final_report += f"最终状态: 持有 `{format_decimal(current_amount, 8)} {current_currency}` (未能成功转换为 USDT)\n"
                    if (
                        final_swap_details
                        and final_swap_details.get("status") == "error"
                    ):
                        final_report += f"失败原因: 最终闪兑 - {final_swap_details.get('message','未知')}\n"
                    else:
                        final_report += f"失败原因: 最终闪兑步骤失败。\n"
                else:  # trade_successful is False
                    final_report += f"🔴 **套利中止或失败**\n"
                    last_amount_report = (
                        format_decimal(current_amount, 8)
                        if current_amount > 0
                        else "未知"
                    )
                    last_currency_report = (
                        current_currency if current_currency else "未知"
                    )
                    if current_amount > 0 and current_currency:
                        final_report += f"最后状态: 持有 `{last_amount_report} {last_currency_report}`\n"
                    # 查找失败原因
                    fail_reason = "未知"
                    if final_report_sections["错误"]:
                        fail_reason = "; ".join(final_report_sections["错误"])
                    elif (
                        execution_step_details
                        and execution_step_details[-1].get("status") == "error"
                    ):
                        fail_reason = f"步骤 {execution_step_details[-1].get('step','?')} - {execution_step_details[-1].get('message','未知')}"
                    elif (
                        initial_swap_details
                        and initial_swap_details.get("status") == "error"
                    ):
                        fail_reason = (
                            f"初始闪兑 - {initial_swap_details.get('message','未知')}"
                        )
                    elif (
                        final_swap_details
                        and final_swap_details.get("status") == "error"
                    ):
                        fail_reason = (
                            f"最终闪兑 - {final_swap_details.get('message','未知')}"
                        )
                    elif not trade_successful:
                        fail_reason = "检查日志了解详情"  # 如果 trade_successful 为 False 但没有明确错误记录
                    final_report += f"失败原因: {fail_reason}\n"

                # --- 发送报告 ---
                logger.info(
                    "最终执行报告:\n"
                    + final_report.replace("`", "")
                    .replace("*", "")
                    .replace("<pre>", "```")
                    .replace("</pre>", "```")
                )  # 日志用 Markdown
                if user_chat_id:
                    try:
                        max_len = 4000
                        parts = [
                            final_report[i : i + max_len]
                            for i in range(0, len(final_report), max_len)
                        ]
                        for part in parts:
                            await application.bot.send_message(
                                chat_id=user_chat_id,
                                text=part,
                                parse_mode=ParseMode.HTML,
                            )
                    except Exception as e:
                        logger.error(f"发送最终交易报告失败: {e}")
                        # Fallback to plain text

                # --- 释放锁 ---
                is_trading_active.release()
                logger.info(f"--- [执行结束] 交易锁已释放 (路径: {path_str}) ---")

            elif not acquired_lock:  # 仅在日志中记录一下
                logger.debug(
                    f"--- [执行] 交易锁在 finally 块检查时未被持有 (路径: {path_str}) ---"
                )

            # --- 清除全局任务引用 ---
            _clear_global_task_reference()

    # --- 顶层异常处理 ---
    except Exception as e:
        logger.error(
            f"[执行错误] 执行套利路径 {path_str} 时发生顶层错误: {e}", exc_info=True
        )
        if user_chat_id:
            try:
                await application.bot.send_message(
                    chat_id=user_chat_id,
                    text=f"❌ 执行套利 `{path_str}` 时发生严重错误: {e}",
                )
            except Exception:
                pass
        # 异常发生，确保锁被释放 (如果已获取)

    finally:
        # --- 确保锁被释放 (无论如何) ---
        if acquired_lock and is_trading_active.locked():
            is_trading_active.release()
            logger.info(
                f"--- [执行结束/错误] 交易锁已在 finally 中释放 (路径: {path_str}) ---"
            )
        # --- 清除全局任务引用 ---
        _clear_global_task_reference()  # 确保即使顶层异常也清理


# --- 余额更新任务 ---
async def update_balance_task(exchange: ccxtpro.Exchange, update_interval_sec: int):
    """后台任务，定期获取并更新全局账户余额。"""
    global global_balances, balance_update_task  # 允许任务自身记录更新时间
    logger.info(f"启动余额更新任务，每 {update_interval_sec} 秒更新一次...")
    while True:
        try:
            # logger.debug("正在获取账户余额...")
            balance_data = await exchange.fetch_balance()  # <--- 获取余额

            # --- ***修改点: 在成功获取数据后立即更新时间戳*** ---
            current_time = time.time()
            if balance_update_task:
                balance_update_task.last_update_time = current_time
            # --- ***修改结束*** ---

            # 我们主要关心 'free' 余额
            free_balances = balance_data.get("free", {})
            # 将余额转换为 Decimal 类型存储，并过滤掉极小值（如 dust）
            new_balances = {}
            for currency, amount_str in free_balances.items():
                try:
                    amount = Decimal(amount_str)
                    if amount > Decimal("1e-12"):  # 避免极其小的非零值
                        new_balances[currency] = amount
                except (ValueError, DecimalInvalidOperation):
                    logger.warning(
                        f"无法将余额字符串 '{amount_str}' (货币: {currency}) 转换为 Decimal。"
                    )

            global_balances = new_balances
            # logger.debug(f"余额更新成功. 更新时间戳: {current_time}") # 可以添加日志确认

        except AuthenticationError as e:
            logger.error(
                f"余额更新失败: API 认证错误 - {e}. 请检查 API 密钥权限。任务将停止。"
            )
            break  # 认证错误，停止任务
        except (
            CCXTNetworkError,
            RequestTimeout,
            ExchangeNotAvailable,
            RateLimitExceeded,
        ) as e:
            logger.warning(
                f"获取余额时遇到临时网络或交易所问题: {type(e).__name__} - {e}. 将在下次循环重试。"
            )
            # 这些错误通常是暂时的，时间戳不会在此次循环更新
        except Exception as e:
            logger.error(f"获取余额时发生未知错误: {e}", exc_info=True)
            # 未知错误，时间戳不会在此次循环更新

        await asyncio.sleep(update_interval_sec)  # 等待下一次更新
    logger.warning("余额更新任务已停止。")


# --- 模拟闪兑函数 (保持不变, 用于验证或比较) ---
async def simulate_swap_order(
    exchange: ccxtpro.Exchange,
    from_currency: str,
    to_currency: str,
    from_amount: Decimal,
    markets: dict,
    current_tickers: dict,
):
    """(模拟闪兑) 使用 Ticker 快照模拟市价单闪兑，计算预估结果。"""
    logger.debug(
        f"[模拟闪兑] 模拟: {format_decimal(from_amount)} {from_currency} -> {to_currency}"
    )

    # 查找直接交易对
    direct_symbol_forward = (
        f"{to_currency}/{from_currency}"  # 例如 BNB/ETH - 用 ETH 买 BNB (检查 Ask)
    )
    direct_symbol_backward = (
        f"{from_currency}/{to_currency}"  # 例如 ETH/BNB - 卖 ETH 换 BNB (检查 Bid)
    )
    intermediate_currency = "USDT"  # 常用中介货币
    symbol_from_intermediate = (
        f"{from_currency}/{intermediate_currency}"  # ETH/USDT (检查 Bid)
    )
    symbol_to_intermediate = (
        f"{to_currency}/{intermediate_currency}"  # BNB/USDT (检查 Ask)
    )

    simulation_steps = []
    estimated_to_amount = Decimal("0")
    fee_rate = config.get("taker_fee_rate", Decimal("0.001"))  # 从全局配置获取费率
    final_symbol_used = None
    final_price_used = None
    final_method = None

    try:
        # 情况 1: 存在反向直接交易对 (例如 ETH/BNB)，需要卖出 from_currency
        ticker_back = current_tickers.get(direct_symbol_backward)
        if (
            ticker_back
            and ticker_back.get("bid")
            and Decimal(str(ticker_back["bid"])) > 0
        ):
            symbol = direct_symbol_backward
            bid_price = Decimal(str(ticker_back["bid"]))
            estimated_quote_received = from_amount * bid_price
            fee = estimated_quote_received * fee_rate
            estimated_to_amount = estimated_quote_received - fee
            simulation_steps.append(
                f"1. 模拟市价卖出 {format_decimal(from_amount)} {from_currency} 在 {symbol} @ 约 {format_decimal(bid_price)} -> 预计收到 {format_decimal(estimated_to_amount)} {to_currency} (已扣除手续费)"
            )
            logger.debug(
                f"[模拟闪兑] 通过 {symbol} 直接卖出 {from_currency} 获得 {to_currency}"
            )
            final_symbol_used = symbol
            final_price_used = bid_price
            final_method = "sell"

        # 情况 2: 存在正向直接交易对 (例如 BNB/ETH)，需要买入 to_currency
        elif (
            direct_symbol_forward in markets
            and direct_symbol_forward in current_tickers
        ):
            symbol = direct_symbol_forward
            ticker_fwd = current_tickers.get(symbol)
            if (
                ticker_fwd
                and ticker_fwd.get("ask")
                and Decimal(str(ticker_fwd["ask"])) > 0
            ):
                ask_price = Decimal(str(ticker_fwd["ask"]))
                estimated_base_received = from_amount / ask_price
                fee = estimated_base_received * fee_rate
                estimated_to_amount = estimated_base_received - fee
                simulation_steps.append(
                    f"1. 模拟市价买入 {to_currency} 使用 {format_decimal(from_amount)} {from_currency} 在 {symbol} @ 约 {format_decimal(ask_price)} -> 预计收到 {format_decimal(estimated_to_amount)} {to_currency} (已扣除手续费)"
                )
                logger.debug(
                    f"[模拟闪兑] 通过 {symbol} 直接买入 {to_currency} 使用 {from_currency}"
                )
                final_symbol_used = symbol
                final_price_used = ask_price
                final_method = "buy_cost"  # 假设按成本买
            else:
                logger.warning(
                    f"[模拟闪兑] {symbol} Ticker 无效或无买价，无法直接转换。"
                )
                pass  # 继续尝试中介路径

        # 情况 3: 需要通过中介货币 (仅在直接路径失败或不存在时尝试)
        if estimated_to_amount <= 0:  # 确保上面两种方式没成功
            ticker_from_int = current_tickers.get(symbol_from_intermediate)
            ticker_to_int = current_tickers.get(symbol_to_intermediate)

            if (
                ticker_from_int
                and ticker_from_int.get("bid")
                and Decimal(str(ticker_from_int["bid"])) > 0
                and ticker_to_int
                and ticker_to_int.get("ask")
                and Decimal(str(ticker_to_int["ask"])) > 0
            ):

                logger.info(
                    f"[模拟闪兑] 无直接路径或 Ticker 无效，尝试通过 {intermediate_currency} 中介..."
                )
                # 步骤 A: 卖出 from_currency 换取 intermediate_currency
                bid_price_from = Decimal(str(ticker_from_int["bid"]))
                intermediate_amount_gross = from_amount * bid_price_from
                fee1 = intermediate_amount_gross * fee_rate
                intermediate_amount_net = intermediate_amount_gross - fee1
                simulation_steps.append(
                    f"1. (中介)模拟卖出 {format_decimal(from_amount)} {from_currency} 在 {symbol_from_intermediate} @ 约 {format_decimal(bid_price_from)} -> 预计收到 {format_decimal(intermediate_amount_net)} {intermediate_currency}"
                )

                if intermediate_amount_net <= 0:
                    logger.warning(
                        f"[模拟闪兑] 中介路径: 第一步后中介金额非正数 ({intermediate_amount_net})。"
                    )
                    return None

                # 步骤 B: 使用 intermediate_currency 买入 to_currency
                ask_price_to = Decimal(str(ticker_to_int["ask"]))
                estimated_to_amount_gross = intermediate_amount_net / ask_price_to
                fee2 = estimated_to_amount_gross * fee_rate
                estimated_to_amount = estimated_to_amount_gross - fee2
                simulation_steps.append(
                    f"2. (中介)模拟买入 {to_currency} 使用 {format_decimal(intermediate_amount_net)} {intermediate_currency} 在 {symbol_to_intermediate} @ 约 {format_decimal(ask_price_to)} -> 预计收到 {format_decimal(estimated_to_amount)} {to_currency}"
                )
                logger.info(
                    f"[模拟闪兑] 通过 {intermediate_currency} 中介转换 {from_currency} -> {to_currency}"
                )
                final_method = "intermediate"  # 标记使用了中介

            else:
                logger.warning(
                    f"[模拟闪兑] 中介路径所需 Ticker ({symbol_from_intermediate} 或 {symbol_to_intermediate}) 无效。"
                )
                # Fall through to final check

        # --- 如果所有路径都失败 ---
        if estimated_to_amount <= 0:  # 检查是否仍未找到路径
            logger.warning(
                f"[模拟闪兑] 无法找到合适的直接或通过 {intermediate_currency} 的模拟路径来转换 {from_currency} -> {to_currency}。"
            )
            return None

        # --- 返回结果 ---
        if estimated_to_amount > 0:
            logger.debug(
                f"[模拟闪兑] 模拟完成: 预计用 {format_decimal(from_amount)} {from_currency} 获得约 {format_decimal(estimated_to_amount)} {to_currency}"
            )
            # 计算模拟的平均价格（如果可能）
            sim_avg_price = None
            sim_expected_price = final_price_used  # 预期价就是模拟时用的价格
            if final_method == "sell" and from_amount > 0:
                sim_avg_price = (
                    estimated_to_amount / (Decimal("1") - fee_rate)
                ) / from_amount  # 反推未扣费价格
            elif final_method == "buy_cost" and estimated_to_amount > 0:
                sim_avg_price = from_amount / (
                    estimated_to_amount / (Decimal("1") - fee_rate)
                )  # 反推未扣费价格

            return {
                "estimated_to_amount": estimated_to_amount,
                "steps": simulation_steps,
                "from_currency": from_currency,
                "to_currency": to_currency,
                "from_amount": from_amount,
                "method": final_method,
                "symbol_used": final_symbol_used,
                "average_price": sim_avg_price,  # 模拟均价 (近似)
                "expected_price": sim_expected_price,  # 模拟预期价
            }
        else:
            # 此处理论上不应到达
            logger.warning(
                f"[模拟闪兑] 模拟结果: 最终估算金额为非正数 ({estimated_to_amount})。"
            )
            return None

    except KeyError as e:
        logger.error(f"[模拟闪兑] 模拟时处理市场或 Ticker 信息时出错: 缺少键 {e}")
        return None
    except (DecimalInvalidOperation, ValueError) as e:
        logger.error(f"[模拟闪兑] 模拟时数值计算错误: {e}")
        return None
    except Exception as e:
        logger.error(f"[模拟闪兑] 模拟时发生未知错误: {e}", exc_info=True)
        return None


# --- WebSocket 监听任务 (分块) ---
async def watch_ticker_chunk_task(
    exchange: ccxtpro.Exchange,
    symbol_chunk: list,
    chunk_index: int,
    conn_status_list: list,  # 传入共享的状态列表
):
    global global_tickers, last_ticker_update_time
    chunk_size = len(symbol_chunk)
    logger.info(
        f"启动 WebSocket 块 {chunk_index+1}/{len(conn_status_list)} 监控任务 (监听 {chunk_size} 个交易对)..."
    )

    # 确保在启动时状态为 False
    if chunk_index < len(conn_status_list):
        conn_status_list[chunk_index] = False
    else:
        logger.error(f"块索引 {chunk_index} 超出状态列表范围!")
        return  # 无法更新状态，退出任务

    while True:  # 外层循环处理重连
        is_connected_this_cycle = False  # 用于标记本轮重连是否成功过
        try:
            logger.info(
                f"块 {chunk_index+1}: 尝试连接并监听 {chunk_size} 个 Tickers..."
            )
            # 内层循环持续获取更新
            while True:
                tickers_update = await exchange.watch_tickers(symbol_chunk)
                """if tickers_update:  # 只在非空时打印，避免刷屏
                    logger.info(
                        f"块 {chunk_index+1}: 收到原始 tickers_update: {tickers_update}"
                    )"""

                # 首次成功接收数据后，更新状态
                if not is_connected_this_cycle:
                    logger.info(f"块 {chunk_index+1}: WebSocket Ticker 流连接成功！")
                    conn_status_list[chunk_index] = True
                    is_connected_this_cycle = True

                # 更新全局状态
                now = time.time()
                last_ticker_update_time = now  # 任何块更新都刷新时间

                if tickers_update:
                    # logger.debug(f"块 {chunk_index+1}: 收到 {len(tickers_update)} 个 Ticker 更新.")
                    # --- !! 优化：仅更新必要的字段，减少内存占用和处理时间 !! ---
                    for symbol, ticker_data in tickers_update.items():
                        # 检查是否有 ask 和 bid，并且它们看起来像有效的数字
                        # 这是为了防止接收到不完整或无效的 ticker 更新污染全局状态
                        ask_val = None
                        bid_val = None
                        timestamp_ms = ticker_data.get("timestamp")  # 保留原始时间戳
                        dt = ticker_data.get("datetime")  # 保留原始 datetime

                        # --- 优先从 info 字段获取 bp 和 ap ---
                        raw_info = ticker_data.get("info", {})
                        if isinstance(raw_info, dict):
                            bid_val = raw_info.get("bp")  # 从 info 获取 bid price
                            ask_val = raw_info.get("ap")  # 从 info 获取 ask price
                            # 有些交易所可能在 info 中嵌套 data 字段
                            if (
                                bid_val is None
                                and ask_val is None
                                and "data" in raw_info
                                and isinstance(raw_info["data"], dict)
                            ):
                                bid_val = raw_info["data"].get("bp")
                                ask_val = raw_info["data"].get("ap")
                            # 如果 info 中有时间戳，可能更精确
                            timestamp_ms = raw_info.get("t", timestamp_ms)
                            # 尝试将 info 中的时间戳转为 datetime (如果需要且原始时间戳无效)
                            if dt is None and timestamp_ms:
                                try:
                                    dt = exchange.iso8601(timestamp_ms)
                                except:
                                    pass  # 转换失败就算了

                        # --- 如果 info 中没有，再尝试顶层的 bid/ask (虽然我们知道它现在是 None) ---
                        if bid_val is None:
                            bid_val = ticker_data.get("bid")
                        if ask_val is None:
                            ask_val = ticker_data.get("ask")

                        # --- 现在使用提取到的 bid_val 和 ask_val 进行检查和存储 ---
                        if ask_val is not None and bid_val is not None:
                            try:
                                # 确保转换成 Decimal 进行处理
                                bid_price_dec = Decimal(str(bid_val))
                                ask_price_dec = Decimal(str(ask_val))

                                # 增加一个基本的价格合理性检查 (可选但推荐)
                                # 例如，卖价应该高于买价，且都大于 0
                                if (
                                    ask_price_dec > 0
                                    and bid_price_dec > 0
                                    and ask_price_dec >= bid_price_dec
                                ):

                                    # ---> 使用提取到的值更新 global_tickers <---
                                    global_tickers[symbol] = {
                                        "symbol": symbol,
                                        "timestamp": timestamp_ms,  # 使用更精确的时间戳 (如果可用)
                                        "datetime": dt,  # 使用更精确的datetime (如果可用)
                                        "high": ticker_data.get(
                                            "high"
                                        ),  # 保留其他汇总信息
                                        "low": ticker_data.get("low"),
                                        "bid": bid_price_dec,  # <--- 存储 Decimal 格式的最佳买价
                                        "ask": ask_price_dec,  # <--- 存储 Decimal 格式的最佳卖价
                                        "last": ticker_data.get("last"),
                                        "baseVolume": ticker_data.get("baseVolume"),
                                        "quoteVolume": ticker_data.get("quoteVolume"),
                                        "info": raw_info,  # 存储原始 info 供调试
                                    }
                                    # ---> 更新时间戳 <---
                                    last_ticker_update_time = time.time()
                                    update_successful = (
                                        True  # (如果你采用了分步更新时间戳的方案)
                                    )
                                    # logger.debug(f"块 {chunk_index+1}: 更新 {symbol} agg_ticker: Bid={bid_val}, Ask={ask_val}")

                                else:
                                    logger.warning(
                                        f"块 {chunk_index+1}: {symbol} 提取的价格无效或不合理: Bid={bid_val}, Ask={ask_val}"
                                    )

                            except (
                                ValueError,
                                TypeError,
                                DecimalInvalidOperation,
                            ) as e:
                                logger.warning(
                                    f"块 {chunk_index+1}: 解析 {symbol} 的 bid/ask ({bid_val}/{ask_val}) 时出错: {e}"
                                )
                        # else:
                        #     logger.debug(f"块 {chunk_index+1}: 收到 {symbol} 的 Ticker 更新，缺少 ask 或 bid。")

        # --- 异常处理: 发生错误时将本块状态设为 False ---
        except (
            CCXTNetworkError,
            RequestTimeout,
            ConnectionRefusedError,
            ConnectionResetError,
            asyncio.TimeoutError,
        ) as e:  # <--- 使用 CCXTNetworkError 或 ccxt.NetworkError
            logger.warning(
                f"块 {chunk_index+1}: WebSocket 网络/超时错误: {type(e).__name__} - {e}. 尝试重新连接..."
            )
            conn_status_list[chunk_index] = False
            await asyncio.sleep(5 + random.uniform(0, 2))
            continue
        except (
            ExchangeNotAvailable
        ) as e:  # 这个通常直接从 ccxt 或 ccxt.base.errors 导入
            logger.error(f"块 {chunk_index+1}: 交易所暂时不可用: {e}. 等待后重试...")
            conn_status_list[chunk_index] = False
            await asyncio.sleep(30 + random.uniform(0, 5))
            continue
        # --- !! 结束修改 !! ---
        except asyncio.CancelledError:
            logger.info(f"块 {chunk_index+1}: WebSocket 监控任务被取消。")
            conn_status_list[chunk_index] = False
            break
        except Exception as e:
            logger.error(
                f"块 {chunk_index+1}: WebSocket 监控任务内部发生未知错误: {e}",
                exc_info=True,
            )  # 保持捕获通用异常
            conn_status_list[chunk_index] = False
            await asyncio.sleep(15 + random.uniform(0, 5))
            continue

    logger.warning(f"块 {chunk_index+1}: WebSocket Ticker 监控任务已停止。")
    conn_status_list[chunk_index] = False


# --- CCXT 连接 (异步版本) ---
async def connect_binance_pro():
    logger.info("正在使用 ccxt.pro 连接币安 API (异步)...")
    try:
        if (
            not API_KEY
            or not API_SECRET
            or API_KEY == "YOUR_API_KEY"
            or API_SECRET == "YOUR_SECRET_KEY"
        ):
            logger.critical("错误：必须设置有效的币安 API_KEY 和 API_SECRET！")
            return None, None

        exchange = ccxtpro.binance(
            {
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": False,  # ccxtpro 内部处理速率限制
                "options": {
                    "adjustForTimeDifference": True,  # 自动调整时间差
                    "defaultType": "spot",  # 确保是现货交易
                    # 'recvWindow': 5000, # 可以根据网络情况调整
                    # 'ws': {
                    #     'options': {
                    #         'pingInterval': 10000, # 10秒 ping 一次 (如果需要)
                    #         'pongTimeout': 5000,   # 5秒没收到 pong 则认为断开 (如果需要)
                    #     }
                    # }
                },
            }
        )
        exchange = ccxtpro.binance(
            {
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": False,
                "options": {
                    "adjustForTimeDifference": True,
                    "defaultType": "spot",
                },
            }
        )

        # 2️⃣ 关掉 sandbox（防止它切到 test 段）
        exchange.set_sandbox_mode(False)

        # 3️⃣ 全面覆盖所有 HTTP 端点
        # 注意：要和 CCXT 默认结构一一对应，包含 sapiV2/V3/V4、wapi、fapi、dapi 等
        exchange.urls["api"].update(
            {
                "public": "https://api2.binance.com/api/v3",
                "private": "https://api2.binance.com/api/v3",
                "v1": "https://api2.binance.com/api/v1",
            }
        )
        exchange.urls.update(
            {
                "sapi": "https://api2.binance.com/sapi/v1",
                "sapiV2": "https://api2.binance.com/sapi/v2",
                "sapiV3": "https://api2.binance.com/sapi/v3",
                "sapiV4": "https://api2.binance.com/sapi/v4",
                "wapi": "https://api2.binance.com/wapi/v3",
                "fapiPublic": "https://fapi2.binance.com/fapi/v1",
                "fapiPublicV2": "https://fapi2.binance.com/fapi/v2",
                "fapiPublicV3": "https://fapi2.binance.com/fapi/v3",
                "fapiPrivate": "https://fapi2.binance.com/fapi/v1",
                "dapiPublic": "https://dapi2.binance.com/dapi/v1",
                "dapiPrivate": "https://dapi2.binance.com/dapi/v1",
                # ...按需再补其它 endpoint
            }
        )
        # exchange.verbose = True # 调试时可以打开，查看详细请求和响应
        logger.info("正在异步加载市场数据...")
        markets = await exchange.load_markets()

        # 过滤掉非活跃或非现货市场（虽然上面筛选了，这里再确认一次）
        markets = {
            symbol: market
            for symbol, market in markets.items()
            if market.get("spot", False) and market.get("active", False)
        }

        logger.info(
            f"成功连接到币安并加载了 {len(markets)} 个活跃现货市场数据 (ccxt.pro)。"
        )

        # --- 首次获取余额 ---
        try:
            logger.info("正在首次获取账户余额...")
            balance_data = await exchange.fetch_balance()
            free_balances = balance_data.get("free", {})
            global global_balances  # 更新全局余额
            global_balances = {
                currency: Decimal(str(amount))
                for currency, amount in free_balances.items()
                if Decimal(str(amount)) > Decimal("1e-12")
            }
            # 仅记录持有的主要币种或价值较高的币种
            relevant_balances = {
                c: format_decimal(a)
                for c, a in global_balances.items()
                if a > Decimal("0.01")
            }  # 示例：只记录大于0.01的
            logger.info(f"首次余额获取成功. 主要持有: {relevant_balances}")
        except Exception as e:
            logger.error(f"首次获取余额失败: {e}. 余额信息将依赖后台任务更新。")

        return exchange, markets
    except ccxt.AuthenticationError as e:
        logger.error(f"API 身份验证失败: {e}。请检查 API 密钥权限。")
        return None, None
    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
        logger.error(f"网络或交易所错误导致连接失败: {type(e).__name__} - {e}")
        return None, None
    except ccxt.ExchangeError as e:
        logger.error(f"交易所连接错误: {e}。")
        return None, None
    except Exception as e:
        logger.error(f"连接或加载市场时发生意外错误: {e}", exc_info=True)
        return None, None


async def connect_xt_pro():
    logger.info("正在使用 ccxt.pro 连接xt API (异步)...")
    try:
        if (
            not API_KEY
            or not API_SECRET
            or API_KEY == "YOUR_API_KEY"
            or API_SECRET == "YOUR_SECRET_KEY"
        ):
            logger.critical("错误：必须设置有效的xt API_KEY 和 API_SECRET！")
            return None, None

        exchange = ccxtpro.xt(
            {
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": False,  # ccxtpro 内部处理速率限制
                "options": {
                    "adjustForTimeDifference": True,  # 自动调整时间差
                    "defaultType": "spot",  # 确保是现货交易
                    # 'recvWindow': 5000, # 可以根据网络情况调整
                    # 'ws': {
                    #     'options': {
                    #         'pingInterval': 10000, # 10秒 ping 一次 (如果需要)
                    #         'pongTimeout': 5000,   # 5秒没收到 pong 则认为断开 (如果需要)
                    #     }
                    # }
                },
            }
        )
        exchange = ccxtpro.xt(
            {
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": False,
                "options": {
                    "adjustForTimeDifference": True,
                    "defaultType": "spot",
                },
            }
        )

        # exchange.verbose = True # 调试时可以打开，查看详细请求和响应
        logger.info("正在异步加载市场数据...")
        markets = await exchange.load_markets()

        # 过滤掉非活跃或非现货市场（虽然上面筛选了，这里再确认一次）
        markets = {
            symbol: market
            for symbol, market in markets.items()
            if market.get("spot", False) and market.get("active", False)
        }

        logger.info(
            f"成功连接到币安并加载了 {len(markets)} 个活跃现货市场数据 (ccxt.pro)。"
        )

        # --- 首次获取余额 ---
        try:
            logger.info("正在首次获取账户余额...")
            balance_data = await exchange.fetch_balance()
            free_balances = balance_data.get("free", {})
            global global_balances  # 更新全局余额
            global_balances = {
                currency: Decimal(str(amount))
                for currency, amount in free_balances.items()
                if Decimal(str(amount)) > Decimal("1e-12")
            }
            # 仅记录持有的主要币种或价值较高的币种
            relevant_balances = {
                c: format_decimal(a)
                for c, a in global_balances.items()
                if a > Decimal("0.01")
            }  # 示例：只记录大于0.01的
            logger.info(f"首次余额获取成功. 主要持有: {relevant_balances}")
        except Exception as e:
            logger.error(f"首次获取余额失败: {e}. 余额信息将依赖后台任务更新。")

        return exchange, markets
    except ccxt.AuthenticationError as e:
        logger.error(f"API 身份验证失败: {e}。请检查 API 密钥权限。")
        return None, None
    except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
        logger.error(f"网络或交易所错误导致连接失败: {type(e).__name__} - {e}")
        return None, None
    except ccxt.ExchangeError as e:
        logger.error(f"交易所连接错误: {e}。")
        return None, None
    except Exception as e:
        logger.error(f"连接或加载市场时发生意外错误: {e}", exc_info=True)
        return None, None


# --- 图构建 (使用 Ticker 快照) ---
# python实现版本
def build_arbitrage_graph(markets, current_tickers_snapshot):
    """使用当前的 Ticker 快照构建套利图的边列表。"""
    graph_edges = []
    currencies = set()
    currency_to_index = {}
    index_to_currency = []

    # logger.debug("开始构建套利图边列表...")
    start_build_time = time.time()

    # 1. 收集所有涉及的货币并创建映射
    all_currencies_temp = set()
    valid_symbols_count = 0
    # 只需遍历快照中的交易对即可，因为快照已经包含了我们要监听的交易对
    for symbol, ticker in current_tickers_snapshot.items():
        market = markets.get(symbol)
        # 确保市场信息存在且有效，并且快照中的 ticker 有价格
        if (
            market
            and market.get("active")
            and market.get("spot")
            and market.get("base")
            and market.get("quote")
        ):
            # 检查快照中的价格是否有效 (必须是数字且大于0)
            ask_val = ticker.get("ask")
            bid_val = ticker.get("bid")
            valid_ask = False
            valid_bid = False
            try:
                valid_ask = ask_val is not None and Decimal(str(ask_val)) > 0
            except:
                pass
            try:
                valid_bid = bid_val is not None and Decimal(str(bid_val)) > 0
            except:
                pass

            if valid_ask and valid_bid:
                all_currencies_temp.add(market["base"])
                all_currencies_temp.add(market["quote"])
                valid_symbols_count += 1
            # else:
            #      logger.debug(f"构建图时跳过 {symbol}，快照价格无效: ask={ask_val}, bid={bid_val}")

    if not all_currencies_temp:
        logger.warning("没有有效的货币对用于构建图。")
        return None, None, None

    index_to_currency = sorted(list(all_currencies_temp))
    currency_to_index = {name: i for i, name in enumerate(index_to_currency)}
    num_currencies = len(index_to_currency)
    # logger.debug(f"参与图构建的唯一货币数: {num_currencies} (来自 {valid_symbols_count} 个快照中有效的交易对)")

    # 2. 构建边列表 (使用 Decimal 计算权重)
    fee_multiplier = Decimal("1.0") - config["taker_fee_rate"]
    processed_edges = 0
    skipped_edges = 0
    zero_weight_edges = 0

    # 再次遍历快照构建边
    for symbol, ticker in current_tickers_snapshot.items():
        market = markets.get(symbol)
        # 再次检查市场和价格有效性 (虽然理论上第一步已经过滤，但双重检查更安全)
        if (
            not market
            or not market.get("active")
            or not market.get("spot")
            or not market.get("base")
            or not market.get("quote")
        ):
            continue

        ask_val = ticker.get("ask")
        bid_val = ticker.get("bid")
        valid_ask = False
        valid_bid = False
        ask_price = None
        bid_price = None
        try:
            ask_price = Decimal(str(ask_val))
            valid_ask = ask_price > 0
        except:
            pass
        try:
            bid_price = Decimal(str(bid_val))
            valid_bid = bid_price > 0
        except:
            pass

        if not valid_ask or not valid_bid:
            skipped_edges += 1
            continue

        base = market["base"]
        quote = market["quote"]

        # 检查货币是否在我们的索引中
        if base not in currency_to_index or quote not in currency_to_index:
            skipped_edges += 1
            continue

        quote_idx = currency_to_index[quote]
        base_idx = currency_to_index[base]

        # 边：Quote -> Base (买入 Base, 花费 Quote) - 使用 Ask Price
        try:
            # 净汇率 = (1 / 买价) * (1 - 手续费)
            net_rate_q_to_b = (Decimal("1.0") / ask_price) * fee_multiplier
            if net_rate_q_to_b > 0:
                # 权重 = -log(净汇率)
                weight = -net_rate_q_to_b.ln()
                if abs(weight) < Decimal("1e-15"):
                    zero_weight_edges += 1
                graph_edges.append(
                    {
                        "from": quote_idx,
                        "to": base_idx,
                        "weight": float(weight),  # 转换为 float 给 C++
                        "pair": symbol,
                        "type": "BUY",
                    }
                )
                processed_edges += 1
            else:
                skipped_edges += 1
        except (
            OverflowError,
            ValueError,
            DecimalInvalidOperation,
        ) as e:  # 捕获可能的数学错误
            # logger.warning(f"计算买入边权重时出错 ({symbol}, ask={ask_price}): {e}")
            skipped_edges += 1

        # 边：Base -> Quote (卖出 Base, 收到 Quote) - 使用 Bid Price
        try:
            # 净汇率 = 卖价 * (1 - 手续费)
            net_rate_b_to_q = bid_price * fee_multiplier
            if net_rate_b_to_q > 0:
                # 权重 = -log(净汇率)
                weight = -net_rate_b_to_q.ln()
                if abs(weight) < Decimal("1e-15"):
                    zero_weight_edges += 1
                graph_edges.append(
                    {
                        "from": base_idx,
                        "to": quote_idx,
                        "weight": float(weight),  # 转换为 float 给 C++
                        "pair": symbol,
                        "type": "SELL",
                    }
                )
                processed_edges += 1
            else:
                skipped_edges += 1
        except (OverflowError, ValueError, DecimalInvalidOperation) as e:
            # logger.warning(f"计算卖出边权重时出错 ({symbol}, bid={bid_price}): {e}")
            skipped_edges += 1

    end_build_time = time.time()
    # logger.debug(f"图边列表构建完成. {processed_edges} 条有效边 (跳过 {skipped_edges} 条, 权重接近零 {zero_weight_edges} 条). 耗时 {end_build_time - start_build_time:.3f} 秒.")

    if not graph_edges:
        logger.warning("图构建未能生成任何有效的边。")
        return None, None, None

    return graph_edges, index_to_currency, currency_to_index


arbitrage_lib = None  # 用于 Bellman-Ford (arbitrage_cpp.dll)
graph_builder_lib = None  # 用于图构建 (build_arbitrage_graph.dll)
cpp_bf_lib_loaded = False
cpp_graph_lib_loaded = False


# --- C++ Bellman-Ford 调用 ---
# CppEdge 结构体定义 (两个 DLL 都可能需要，定义一次即可)
class CppEdge(ctypes.Structure):
    _fields_ = [
        ("from_node", ctypes.c_int),
        ("to_node", ctypes.c_int),
        ("weight", ctypes.c_double),
        ("pair_symbol", ctypes.c_char_p),
        ("trade_type", ctypes.c_char_p),
    ]


# --- 加载 Bellman-Ford DLL (arbitrage_cpp.dll) ---
try:
    if os.path.exists(BF_DLL_PATH):
        arbitrage_lib = ctypes.CDLL(BF_DLL_PATH)
        logger.info(f"成功加载 Bellman-Ford C++ 库: {BF_DLL_PATH}")
        # 定义 find_negative_cycles 签名
        arbitrage_lib.find_negative_cycles.argtypes = [
            ctypes.c_int,  # num_nodes
            ctypes.POINTER(CppEdge),  # edges
            ctypes.c_int,  # num_edges
            ctypes.c_int,  # max_depth
            ctypes.POINTER(ctypes.c_char_p),  # result_json_ptr (输出)
            ctypes.POINTER(ctypes.c_longlong),  # relaxation_count_ptr (输出)
        ]
        arbitrage_lib.find_negative_cycles.restype = ctypes.c_int  # 返回码
        # 定义 free_memory 签名 (针对 arbitrage_lib)
        arbitrage_lib.free_memory.argtypes = [ctypes.c_char_p]
        arbitrage_lib.free_memory.restype = None
        cpp_bf_lib_loaded = True
    else:
        logger.error(f"错误：找不到 Bellman-Ford C++ 库文件 '{BF_DLL_PATH}'。")
except OSError as e:
    logger.error(f"加载 Bellman-Ford C++ 库 '{BF_DLL_PATH}' 时发生 OS 错误: {e}。")
except Exception as e:
    logger.error(f"加载 Bellman-Ford C++ 库 '{BF_DLL_PATH}' 时发生未知错误: {e}。")

# --- 加载图构建 DLL (build_arbitrage_graph.dll) ---
try:
    if os.path.exists(GRAPH_DLL_PATH):
        graph_builder_lib = ctypes.CDLL(GRAPH_DLL_PATH)
        logger.info(f"成功加载图构建 C++ 库: {GRAPH_DLL_PATH}")
        # 定义 build_graph_cpp 签名
        graph_builder_lib.build_graph_cpp.argtypes = [
            ctypes.c_char_p,  # tickers_json_str
            ctypes.c_char_p,  # markets_json_str
            ctypes.c_double,  # taker_fee_rate
        ]
        graph_builder_lib.build_graph_cpp.restype = ctypes.c_char_p  # 返回 char*
        # 定义 free_memory 签名 (针对 graph_builder_lib)
        graph_builder_lib.free_memory.argtypes = [ctypes.c_char_p]
        graph_builder_lib.free_memory.restype = None
        cpp_graph_lib_loaded = True
    else:
        logger.error(f"错误：找不到图构建 C++ 库文件 '{GRAPH_DLL_PATH}'。")
except OSError as e:
    logger.error(f"加载图构建 C++ 库 '{GRAPH_DLL_PATH}' 时发生 OS 错误: {e}。")
except Exception as e:
    logger.error(f"加载图构建 C++ 库 '{GRAPH_DLL_PATH}' 时发生未知错误: {e}。")

# --- 加载操作 DLL (arbitrage_operations.dll) ---
try:
    if os.path.exists(OPS_DLL_PATH):
        ops_lib = ctypes.CDLL(OPS_DLL_PATH)
        logger.info(f"成功加载操作 C++ 库: {OPS_DLL_PATH}")

        # --- !! 修改点：使用新的函数名 assess_risk_cpp_buffered !! ---
        ops_lib.assess_risk_cpp_buffered.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,  # <-- output_buffer, buffer_size
        ]
        ops_lib.assess_risk_cpp_buffered.restype = ctypes.c_int  # <-- 返回状态码

        # --- !! 修改点：使用新的函数名 simulate_full_cpp_buffered !! ---
        ops_lib.simulate_full_cpp_buffered.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,  # <-- output_buffer, buffer_size
        ]
        ops_lib.simulate_full_cpp_buffered.restype = ctypes.c_int  # <-- 返回状态码

        # --- 移除 free_memory_ops 的加载 ---
        # ops_lib.free_memory_ops.argtypes = [ctypes.c_char_p]
        # ops_lib.free_memory_ops.restype = None

        cpp_ops_lib_loaded = True
    else:
        logger.error(f"错误：找不到操作 C++ 库文件 '{OPS_DLL_PATH}'。")
# --- !! 修改点：捕获 AttributeError !! ---
except AttributeError as e:
    logger.error(
        f"加载操作 C++ 库 '{OPS_DLL_PATH}' 时函数未找到: {e}。请检查 DLL 导出和 Python 函数名是否匹配。",
        exc_info=True,
    )
    cpp_ops_lib_loaded = False  # 标记为加载失败
except OSError as e:
    logger.error(f"加载操作 C++ 库 '{OPS_DLL_PATH}' 时发生 OS 错误: {e}。")
    cpp_ops_lib_loaded = False
except Exception as e:
    logger.error(f"加载操作 C++ 库 '{OPS_DLL_PATH}' 时发生未知错误: {e}。")
    cpp_ops_lib_loaded = False
# --- 新增: 控制 C++ 图构建的配置 (如果尚未添加) ---
if "use_cpp_graph_build" not in config:
    config["use_cpp_graph_build"] = True  # 默认启用 (如果 DLL 加载成功)


# --- C++ 图构建函数的 Python 包装器 ---
# --- C++ 图构建函数的 Python 包装器 ---
# --- C++ 图构建函数的 Python 包装器 ---
def build_graph_cpp_wrapper(markets, current_tickers_snapshot, config):
    """
    调用 图构建 DLL 中的 build_graph_cpp 函数来构建图。
    返回:
        tuple: (graph_edges, index_to_currency, currency_to_index) 或 (None, None, None) 如果失败。
    """
    global cpp_graph_lib_loaded, graph_builder_lib

    if not cpp_graph_lib_loaded or not graph_builder_lib:
        logger.error("图构建 C++ 库未加载，无法调用 build_graph_cpp。")
        return None, None, None

    graph_edges = None
    index_to_currency = None
    currency_to_index = None
    result_ptr_address = None  # <--- 用于存储原始指针地址 (整数)
    returned_json_str = None

    try:
        # 1. 准备输入数据
        relevant_market_data = {}
        for symbol, ticker_data in current_tickers_snapshot.items():
            market_info = markets.get(symbol)
            if market_info and market_info.get("active") and market_info.get("spot"):
                relevant_market_data[symbol] = {
                    "active": market_info.get("active", False),
                    "spot": market_info.get("spot", False),
                    "base": market_info.get("base"),
                    "quote": market_info.get("quote"),
                }
        tickers_json_str = json.dumps(current_tickers_snapshot)
        markets_json_str = json.dumps(relevant_market_data)
        tickers_json_bytes = tickers_json_str.encode("utf-8")
        markets_json_bytes = markets_json_str.encode("utf-8")
        taker_fee_rate = float(config["taker_fee_rate"])

        # 2. 调用 C++ 函数 (明确指定返回类型为地址)
        logger.debug("准备调用 C++ build_graph_cpp...")
        # --- !! 修改: 将 restype 临时改为 c_void_p 获取原始地址 !! ---
        graph_builder_lib.build_graph_cpp.restype = ctypes.c_void_p
        result_ptr_address = graph_builder_lib.build_graph_cpp(
            ctypes.c_char_p(tickers_json_bytes),
            ctypes.c_char_p(markets_json_bytes),
            ctypes.c_double(taker_fee_rate),
        )
        # 恢复 restype 为 c_char_p，以便后续可能的其他调用或标准用法
        graph_builder_lib.build_graph_cpp.restype = ctypes.c_char_p
        # --- !! 结束修改 !! ---

        logger.debug(f"C++ build_graph_cpp 返回原始指针地址: {result_ptr_address}")

        # 3. 处理返回结果
        if result_ptr_address:  # 检查地址是否非空
            try:
                # --- 使用获取到的地址来读取字符串 ---
                returned_bytes = ctypes.c_char_p(result_ptr_address).value
                if returned_bytes:
                    returned_json_str = returned_bytes.decode("utf-8")
                    logger.debug(
                        f"成功从 C++ 指针地址解码 JSON 字符串 (长度: {len(returned_json_str)})"
                    )
                else:
                    logger.error(
                        f"从 C++ 指针地址 {result_ptr_address} 获取的 value 为空或 None。"
                    )
                    # 即使字符串为空，仍然需要尝试释放原始指针地址
                    return None, None, None  # 返回失败

                # --- 现在使用复制过来的 Python 字符串进行 JSON 解析 ---
                result_data = json.loads(returned_json_str)

                if (
                    isinstance(result_data, dict)
                    and "nodes" in result_data
                    and "edges" in result_data
                ):
                    index_to_currency = result_data["nodes"]
                    graph_edges = result_data["edges"]
                    currency_to_index = {
                        name: i for i, name in enumerate(index_to_currency)
                    }
                    logger.debug(
                        f"成功解析 C++ 返回的 JSON 数据: {len(index_to_currency)} 节点, {len(graph_edges)} 边。"
                    )
                else:
                    logger.error(
                        f"C++ build_graph_cpp 返回的 JSON 结构无效 (解析后): {result_data}"
                    )
                    return None, None, None  # 返回失败

            # ... (except 块保持不变，处理解码、JSON解析等错误) ...
            except json.JSONDecodeError as e:
                logger.error(f"解析从 C++ 获取的 JSON 字符串时出错: {e}")
                logger.error(
                    f"出错的 JSON (前 1000 字符): {returned_json_str[:1000] if returned_json_str else 'None'}"
                )
                return None, None, None
            except UnicodeDecodeError as e:
                logger.error(f"解码从 C++ 获取的 bytes 时出错 (可能不是 UTF-8): {e}")
                return None, None, None
            except Exception as e:
                logger.error(
                    f"处理 C++ 返回指针或解析 JSON 时发生未知错误: {e}", exc_info=True
                )
                return None, None, None
        else:
            logger.error("C++ build_graph_cpp 调用失败 (返回 nullptr/0 地址)。")
            return None, None, None

    except Exception as call_e:
        logger.error(
            f"调用 C++ build_graph_cpp 或准备数据时出错: {call_e}", exc_info=True
        )
        return None, None, None
    finally:
        # 4. !!! 关键: 使用存储的原始地址 result_ptr_address 来释放 !!!
        if result_ptr_address:
            logger.debug(
                f"准备在 finally 块中释放 C++ 内存指针 (地址): {result_ptr_address}"
            )
            try:
                # --- !! 修改: 构造 c_char_p 对象传递 !! ---
                # 直接使用整数地址构造 c_char_p
                ptr_to_free = ctypes.c_char_p(result_ptr_address)
                graph_builder_lib.free_memory(ptr_to_free)
                # --- !! 结束修改 !! ---
                logger.debug("调用 C++ free_memory 完成。")
            except Exception as free_e:
                # 这里仍然可能出错，例如如果地址确实无效，但至少类型对了
                logger.error(
                    f"调用 C++ free_memory 时出错: {free_e}", exc_info=True
                )  # 添加 exc_info
        # else:
        #    logger.debug("result_ptr_address 为 None 或 0，无需调用 free_memory。")

    # ... (返回结果的逻辑保持不变) ...
    if (
        graph_edges is not None
        and index_to_currency is not None
        and currency_to_index is not None
    ):
        return graph_edges, index_to_currency, currency_to_index
    else:
        return None, None, None


# --- Bellman-Ford 实现选择 (增强错误处理和日志) ---
def find_negative_cycles_bellman_ford(
    graph_edges_list, index_to_currency, currency_to_index, markets
):
    """
    查找图中的负权重环路 (套利机会)。
    优先使用 C++ DLL (如果配置且加载成功)，否则回退到 Python 实现。
    """
    global last_cpp_relaxation_count, last_end_time, config, cpp_lib_loaded, arbitrage_lib
    num_currencies = len(index_to_currency)

    if not graph_edges_list:
        logger.warning("边列表为空，无法运行 Bellman-Ford。")
        return []

    # --- 使用 C++ DLL ---
    if config["use_cpp_bf"] and cpp_bf_lib_loaded and arbitrage_lib:
        start_bf_time = time.time()
        num_edges = len(graph_edges_list)
        CEdgeArray = CppEdge * num_edges
        c_edges = CEdgeArray()
        edge_string_buffers = []  # 保持对编码后字符串的引用，防止被垃圾回收
        json_result_ptr = ctypes.c_char_p(None)  # 初始化为 None
        relaxation_count_c = ctypes.c_longlong(0)

        try:
            # --- 准备 C 结构体 ---
            for i, edge_data in enumerate(graph_edges_list):
                try:
                    # 确保 pair 和 type 是字符串
                    pair_str = str(edge_data.get("pair", ""))
                    type_str = str(edge_data.get("type", ""))
                    # 编码为 bytes
                    pair_bytes = pair_str.encode("utf-8")
                    type_bytes = type_str.encode("utf-8")
                    # 存储 bytes 对象引用
                    edge_string_buffers.extend([pair_bytes, type_bytes])
                    # 填充 C 结构体
                    c_edges[i] = CppEdge(
                        ctypes.c_int(edge_data["from"]),
                        ctypes.c_int(edge_data["to"]),
                        ctypes.c_double(edge_data["weight"]),  # 确保是 float/double
                        ctypes.c_char_p(pair_bytes),
                        ctypes.c_char_p(type_bytes),
                    )
                except KeyError as ke:
                    logger.error(
                        f"准备 C 边数据时缺少键: {ke} (边 {i}, 数据: {edge_data})。跳过此边。"
                    )
                    # 可以选择将此边标记为无效或直接跳过，这里简单跳过
                    c_edges[i] = CppEdge(0, 0, 0.0, None, None)  # 填充无效数据
                except (TypeError, AttributeError, ValueError) as prep_err:
                    logger.error(
                        f"准备 C 边数据时发生类型或值错误 (边 {i}, 数据: {edge_data}): {prep_err}"
                    )
                    c_edges[i] = CppEdge(0, 0, 0.0, None, None)  # 填充无效数据

            # --- 调用 C++ 函数 ---
            logger.debug(
                f"Calling C++ find_negative_cycles with {num_currencies} nodes, {num_edges} edges, max_depth {config['max_arbitrage_depth']}"
            )
            result_code = arbitrage_lib.find_negative_cycles(
                ctypes.c_int(num_currencies),
                c_edges,
                ctypes.c_int(num_edges),
                ctypes.c_int(config["max_arbitrage_depth"]),
                ctypes.byref(json_result_ptr),  # 传递指针的地址
                ctypes.byref(relaxation_count_c),  # 传递指针的地址
            )

            # 记录松弛次数和耗时
            relaxation_count = relaxation_count_c.value
            last_cpp_relaxation_count = relaxation_count
            end_bf_time = time.time()
            bf_duration = end_bf_time - start_bf_time
            last_end_time = bf_duration
            # logger.debug(f"Bellman-Ford (C++ DLL) 完成，耗时 {bf_duration:.4f} 秒，松弛次数: {relaxation_count}.")

            # --- 处理结果 ---
            negative_cycles_result = []
            if result_code == 0:  # 0 表示成功
                if json_result_ptr and json_result_ptr.value:  # 检查指针和值是否有效
                    try:
                        json_string = json_result_ptr.value.decode("utf-8")
                        # logger.debug(f"C++ raw JSON result: {json_string[:500]}...") # 打印部分原始 JSON 调试
                        cycles_data = json.loads(json_string)
                        num_currencies_local = len(index_to_currency)  # 获取局部变量

                        # --- !! 严格的数据验证和转换 !! ---
                        for cycle_cpp in cycles_data:
                            cycle_py = {"depth": 0, "nodes": [], "trades": []}
                            valid_cycle_data = True

                            # 验证 depth
                            depth_val = cycle_cpp.get("depth")
                            if not isinstance(depth_val, int) or not (
                                1 < depth_val <= config["max_arbitrage_depth"]
                            ):
                                logger.warning(
                                    f"C++ 结果包含无效深度: {depth_val}。丢弃环路。"
                                )
                                valid_cycle_data = False
                                continue
                            cycle_py["depth"] = depth_val

                            # 验证和转换 nodes
                            nodes_cpp = cycle_cpp.get("nodes")
                            if (
                                not isinstance(nodes_cpp, list)
                                or len(nodes_cpp) != depth_val + 1
                            ):
                                logger.warning(
                                    f"C++ 结果节点数据无效或长度 ({len(nodes_cpp) if isinstance(nodes_cpp, list) else 'N/A'}) 与深度 ({depth_val}) 不符。丢弃环路。"
                                )
                                valid_cycle_data = False
                                continue
                            temp_nodes = []
                            for idx in nodes_cpp:
                                if not isinstance(idx, int) or not (
                                    0 <= idx < num_currencies_local
                                ):
                                    logger.error(
                                        f"C++ 结果包含无效节点索引: {idx}。丢弃环路。"
                                    )
                                    valid_cycle_data = False
                                    break
                                try:
                                    temp_nodes.append(index_to_currency[idx])
                                except IndexError:
                                    logger.error(
                                        f"C++ 节点索引 {idx} 在 index_to_currency 中越界。丢弃环路。"
                                    )
                                    valid_cycle_data = False
                                    break
                            if not valid_cycle_data:
                                continue
                            cycle_py["nodes"] = temp_nodes

                            # 验证和转换 trades
                            trades_cpp = cycle_cpp.get("trades")
                            if (
                                not isinstance(trades_cpp, list)
                                or len(trades_cpp) != depth_val
                            ):
                                logger.warning(
                                    f"C++ 结果交易数据无效或长度 ({len(trades_cpp) if isinstance(trades_cpp, list) else 'N/A'}) 与深度 ({depth_val}) 不符。丢弃环路。"
                                )
                                valid_cycle_data = False
                                continue
                            temp_trades = []
                            for trade_cpp in trades_cpp:
                                from_idx = trade_cpp.get("from_node")
                                to_idx = trade_cpp.get("to_node")
                                pair_val = trade_cpp.get("pair")
                                type_val = trade_cpp.get("type")
                                # 基本类型和值检查
                                if (
                                    not isinstance(from_idx, int)
                                    or not (0 <= from_idx < num_currencies_local)
                                    or not isinstance(to_idx, int)
                                    or not (0 <= to_idx < num_currencies_local)
                                    or not isinstance(pair_val, str)
                                    or not pair_val
                                    or not isinstance(type_val, str)
                                    or type_val not in ("BUY", "SELL")
                                ):
                                    logger.warning(
                                        f"C++ 交易数据包含无效字段或值: {trade_cpp}。丢弃环路。"
                                    )
                                    valid_cycle_data = False
                                    break
                                # 检查交易节点是否与环路节点对应 (可选但推荐)
                                expected_from_node = cycle_py["nodes"][
                                    len(temp_trades)
                                ]  # 当前交易应对应的起始节点
                                expected_to_node = cycle_py["nodes"][
                                    len(temp_trades) + 1
                                ]  # 当前交易应对应的结束节点
                                actual_from_node = index_to_currency[from_idx]
                                actual_to_node = index_to_currency[to_idx]
                                if (
                                    actual_from_node != expected_from_node
                                    or actual_to_node != expected_to_node
                                ):
                                    logger.warning(
                                        f"C++ 交易节点 ({actual_from_node}->{actual_to_node}) 与环路节点 ({expected_from_node}->{expected_to_node}) 不符。丢弃环路。"
                                    )
                                    valid_cycle_data = False
                                    break

                                # 添加到临时列表
                                temp_trades.append(
                                    {
                                        "from": actual_from_node,
                                        "to": actual_to_node,
                                        "pair": pair_val,
                                        "type": type_val,
                                    }
                                )

                            if not valid_cycle_data:
                                continue
                            cycle_py["trades"] = temp_trades

                            # 只有完全有效的数据才添加到结果列表
                            negative_cycles_result.append(cycle_py)
                            # logger.debug(f"C++ 提取到有效负环: {' -> '.join(cycle_py['nodes'])}")

                    except json.JSONDecodeError as e:
                        logger.error(
                            f"解析 C++ 返回的 JSON 时出错: {e}. JSON (前500字符): '{json_string[:500]}'"
                        )
                    except Exception as e:
                        logger.error(f"处理 C++ 结果时发生未知错误: {e}", exc_info=True)
                else:
                    # result_code == 0 但没有返回 JSON，说明没有找到负环
                    logger.debug("C++ 函数成功执行，但未找到负环。")
            elif result_code == 1:  # 1 通常表示未找到负环 (根据 C++ 实现)
                logger.debug("C++ 函数执行完成，未检测到负环。")
            else:  # 其他错误码
                logger.error(f"C++ 函数 find_negative_cycles 返回错误码: {result_code}")

            return negative_cycles_result

        except Exception as e:
            logger.error(f"调用 C++ 函数或准备数据时发生顶层错误: {e}", exc_info=True)
            return []  # 出错时返回空列表
        finally:
            # --- !! 关键: 确保清理 C++ 分配的内存 !! ---
            # logger.debug("Entering finally block for C++ cleanup...")
            if json_result_ptr and json_result_ptr.value:
                try:
                    # logger.debug(f"Attempting to free C++ memory at {json_result_ptr.value}...")
                    arbitrage_lib.free_memory(json_result_ptr)
                    # logger.debug("C++ allocated memory successfully freed.")
                except Exception as free_e:
                    logger.error(f"Freeing C++ memory failed: {free_e}")
            # 清理 Python 端的字符串 buffer 引用 (帮助垃圾回收)
            edge_string_buffers.clear()
            # logger.debug("Exiting finally block for C++ cleanup.")

    # --- 回退到 Python 实现 ---
    else:
        if config["use_cpp_bf"] and not cpp_lib_loaded:
            logger.warning("配置使用 C++ 但加载失败，回退到 Python 实现。")
        elif not config["use_cpp_bf"]:
            logger.debug("使用 Python 实现 Bellman-Ford。")

        start_bf_time = time.time()

        # 构建 Python 图 (使用 Decimal)
        graph_py = collections.defaultdict(list)
        for edge_data in graph_edges_list:
            try:
                graph_py[index_to_currency[edge_data["from"]]].append(
                    {
                        "to": index_to_currency[edge_data["to"]],
                        "weight": Decimal(
                            str(edge_data["weight"])
                        ),  # Python 使用 Decimal
                        "pair": edge_data["pair"],
                        "type": edge_data["type"],
                    }
                )
            except IndexError:
                logger.warning(
                    f"构建 Python 图时索引越界: from={edge_data.get('from')}, to={edge_data.get('to')}"
                )
                continue
            except Exception as build_py_e:
                logger.error(f"构建 Python 图时出错: {build_py_e}, 数据: {edge_data}")
                continue

        currencies_list = index_to_currency
        distance = {c: Decimal("Infinity") for c in currencies_list}
        predecessor = {c: None for c in currencies_list}
        predecessor_edge_info = {c: None for c in currencies_list}  # 记录导致更新的边

        if not currencies_list:
            return []
        start_node = currencies_list[0]  # 选择第一个货币作为起点
        distance[start_node] = Decimal("0")

        total_relaxation_checks = 0
        nodes_updated_in_last_round = set()

        # Bellman-Ford 主循环 (n-1 次松弛 + 1 次检测)
        for i in range(num_currencies):
            updated_in_this_round = False
            current_round_updated_nodes = set()  # 仅用于第 n 轮检测

            for u_idx, u in enumerate(currencies_list):
                # 优化: 如果节点在前一轮未更新，且距离仍为无穷大，则跳过
                if distance[u] == Decimal("Infinity"):
                    continue

                if u in graph_py:
                    for edge in graph_py[u]:
                        total_relaxation_checks += 1
                        v = edge["to"]
                        weight = edge["weight"]  # 使用 Decimal
                        new_potential_distance = distance[u] + weight

                        # 尝试松弛
                        if new_potential_distance < distance[v]:
                            # 检查下溢 (虽然 Decimal 通常处理得很好)
                            if new_potential_distance > Decimal("-Infinity"):
                                distance[v] = new_potential_distance
                                predecessor[v] = u
                                predecessor_edge_info[v] = edge  # 记录导致更新的边信息
                                updated_in_this_round = True
                                if (
                                    i == num_currencies - 1
                                ):  # 如果在第 n 轮仍然更新，则存在负环
                                    current_round_updated_nodes.add(v)
                            # elif i == num_currencies - 1: # 即使下溢，在第 n 轮也标记为负环信号
                            #     current_round_updated_nodes.add(v)

            # 优化：如果一轮没有更新，且不是最后一轮检测轮，则可以提前终止
            # if not updated_in_this_round and i < num_currencies - 1:
            #     logger.debug(f"Python BF: 第 {i+1} 轮无更新，提前终止松弛。")
            #     break

            if i == num_currencies - 1:  # 记录最后一轮（检测轮）更新的节点
                nodes_updated_in_last_round = current_round_updated_nodes

        end_bf_time = time.time()
        bf_duration = end_bf_time - start_bf_time
        logger.debug(
            f"Bellman-Ford (Python) 完成，耗时 {bf_duration:.4f} 秒，检查次数: {total_relaxation_checks}."
        )
        last_end_time = bf_duration  # 记录耗时给 status

        if not nodes_updated_in_last_round:
            logger.debug("Python BF: 未检测到负环。")
            return []  # 没有负环
        else:
            logger.debug(
                f"Python BF: 检测到 {len(nodes_updated_in_last_round)} 个节点在最后一轮被更新（负环信号）。开始提取..."
            )

        # --- Python 负环提取 (保持之前的逻辑，但增加日志) ---
        negative_cycles = []
        found_cycles_signature = set()  # 用于检测重复环路 (基于排序后的节点)
        processed_nodes_in_cycle = set()  # 避免从同一个环路的不同节点重复提取

        for start_node_in_cycle in list(
            nodes_updated_in_last_round
        ):  # 迭代副本以防修改
            if start_node_in_cycle in processed_nodes_in_cycle:
                continue

            # 1. 回溯 n 步找到环路上的一个节点 'node_in_cycle'
            current = start_node_in_cycle
            path_nodes_temp = []
            for _ in range(num_currencies + 1):  # 回溯 n+1 次确保进入环路
                path_nodes_temp.append(current)
                if current is None or predecessor[current] is None:
                    logger.debug(
                        f"从节点 {start_node_in_cycle} 回溯时路径中断于 {current}。"
                    )
                    current = None
                    break  # 路径中断
                current = predecessor[current]
            node_in_cycle = current  # current 现在应该在环上，或者为 None

            if node_in_cycle is None:
                # logger.debug(f"从节点 {start_node_in_cycle} 回溯未能找到环上节点。路径: {path_nodes_temp}")
                continue  # 无法回溯到环路上

            # 2. 从环路上的节点开始，再次回溯直到重复，提取环路节点
            cycle_nodes_final_rev = []  # 存储反向的环路节点
            visited_in_cycle_trace = {}  # 记录访问过的节点及其在列表中的位置
            step_count = 0
            tracer = node_in_cycle  # 从确定在环路上的节点开始

            while tracer is not None and tracer not in visited_in_cycle_trace:
                if step_count > num_currencies:  # 理论上环路长度 <= n
                    logger.warning(
                        f"提取环路时步骤过多 ({step_count} > {num_currencies})，可能存在逻辑错误。节点: {tracer}"
                    )
                    tracer = None
                    break
                visited_in_cycle_trace[tracer] = len(cycle_nodes_final_rev)
                cycle_nodes_final_rev.append(tracer)
                processed_nodes_in_cycle.add(tracer)  # 标记此节点已处理
                tracer = predecessor[tracer]
                step_count += 1

            if tracer is None or tracer not in visited_in_cycle_trace:
                # logger.debug(f"未能找到环路的闭合点。起始: {start_node_in_cycle}, 当前: {tracer}, 已访问: {list(visited_in_cycle_trace.keys())}")
                continue  # 未找到环路闭合点

            # 提取环路部分并反转
            cycle_start_index_in_rev = visited_in_cycle_trace[tracer]
            cycle_nodes_raw_rev = cycle_nodes_final_rev[cycle_start_index_in_rev:]
            cycle_nodes_final = list(reversed(cycle_nodes_raw_rev))

            # 确保环路闭合 (首尾节点相同)
            if (
                not cycle_nodes_final or cycle_nodes_final[0] != tracer
            ):  # 首节点应为闭合点
                logger.warning(
                    f"提取的 Python 环路未正确闭合: {cycle_nodes_final} (闭合点应为 {tracer})"
                )
                continue
            cycle_nodes_final.append(tracer)  # 添加闭合点到末尾

            cycle_depth = len(cycle_nodes_final) - 1
            if not (1 < cycle_depth <= config["max_arbitrage_depth"]):  # 深度至少为2
                # logger.debug(f"Python 提取的环路深度 {cycle_depth} 不符合要求 (2 < depth <= {config['max_arbitrage_depth']})")
                continue  # 深度不符

            # 3. 根据节点顺序提取交易边信息
            ordered_trades = []
            valid_cycle = True
            total_weight = Decimal("0")
            for i in range(cycle_depth):
                from_node = cycle_nodes_final[i]
                to_node = cycle_nodes_final[i + 1]
                edge_info = predecessor_edge_info.get(to_node)

                # 验证找到的边是否确实是从 from_node 指向 to_node
                if (
                    not edge_info
                    or predecessor.get(to_node) != from_node
                    or edge_info.get("to") != to_node
                ):
                    # logger.debug(f"前驱边信息不匹配 for {from_node}->{to_node}。尝试从图中查找...")
                    found_edge = False
                    if from_node in graph_py:
                        for edge in graph_py[from_node]:
                            if edge["to"] == to_node:
                                edge_info = edge
                                found_edge = True
                                break
                    if not found_edge:
                        logger.warning(
                            f"无法为环路 {cycle_nodes_final} 找到边 {from_node} -> {to_node} 的信息。"
                        )
                        valid_cycle = False
                        break
                else:
                    pass  # 使用 predecessor_edge_info[to_node]

                if edge_info:
                    ordered_trades.append(
                        {
                            "from": from_node,
                            "to": to_node,
                            "pair": edge_info["pair"],
                            "type": edge_info["type"],
                        }
                    )
                    total_weight += edge_info["weight"]  # 累加权重验证
                else:  # 理论上不会到这里，因为上面已经 break 了
                    valid_cycle = False
                    break

            # 4. 存储有效且不重复的环路
            if valid_cycle and len(ordered_trades) == cycle_depth:
                # 再次确认权重和是否为负 (可以增加容差)
                if total_weight >= Decimal("-1e-9"):
                    # logger.debug(f"提取的环路 {cycle_nodes_final} 权重和 {total_weight} 非负，丢弃。")
                    continue

                # 创建环路签名以去重 (排序节点列表，去掉重复的尾节点)
                cycle_signature_nodes = tuple(sorted(cycle_nodes_final[:-1]))
                if cycle_signature_nodes not in found_cycles_signature:
                    found_cycles_signature.add(cycle_signature_nodes)
                    cycle_info_py = {
                        "nodes": cycle_nodes_final,
                        "trades": ordered_trades,
                        "depth": cycle_depth,
                    }
                    negative_cycles.append(cycle_info_py)
                    # logger.debug(f"Python 提取到负环 (深度 {cycle_depth}, 权重和 {total_weight:.6f}): {' -> '.join(cycle_nodes_final)}")
                # else:
                # logger.debug(f"跳过重复环路 (Python): {cycle_nodes_final}")

        logger.debug(f"Python 负环提取完成，找到 {len(negative_cycles)} 个候选环路。")
        return negative_cycles


# --- 模拟验证函数 (旧版 - verify_profit_simulation) ---
# 这个函数模拟的是一个简单的环路，没有考虑初始和最终的强制闪兑
# 它主要用于快速过滤掉利润率过低的环路，可以保留作为一个可选的预过滤步骤
# 但在主循环中，我们现在依赖 simulate_full_execution_profit 进行更全面的模拟
def verify_profit_simulation(cycle_info, start_amount, current_tickers, markets):
    """(旧版模拟) 模拟一个简单的套利环路，不含强制首尾闪兑。"""
    global config
    trades = cycle_info["trades"]
    start_currency = cycle_info["nodes"][0]
    current_amount = Decimal(str(start_amount))  # 确保是 Decimal
    current_currency = start_currency
    log_msgs = []
    path_str = " -> ".join(cycle_info.get("nodes", ["?"]))  # 用于日志
    fee_rate = config["taker_fee_rate"]  # 使用配置中的手续费率

    try:
        # 构建日志头部
        fee_percentage_str = f"{fee_rate * 100:.4f}%"
        fee_column_label = f"手续费({fee_percentage_str})"
        header = f"  {'步骤':<4s} | {'操作':<14s} | {'交易对':<12s} | {'价格':<18s} | {'发送额/量':<28s} | {fee_column_label:<28s} | {'净收额/量':<28s}"
        separator = "  " + "-" * (len(header) - 2)
        log_msgs.append(f"--- 简单模拟 ({path_str}) ---")
        log_msgs.append(header)
        log_msgs.append(separator)
        log_msgs.append(f"  起始: {format_decimal(current_amount)} {current_currency}")

        for i, trade in enumerate(trades):
            step_start_amount = current_amount
            step_start_amount_formatted_str = format_decimal(current_amount)
            pair = trade["pair"]
            market = markets.get(pair)
            ticker = current_tickers.get(pair)

            # 验证市场和 Ticker 数据
            if not market:
                raise ValueError(f"模拟时缺少市场数据 for {pair}")
            if not ticker:
                raise ValueError(f"模拟时缺少 Ticker 数据 for {pair}")
            ask_price_raw = ticker.get("ask")
            bid_price_raw = ticker.get("bid")
            valid_ask = False
            valid_bid = False
            ask_price = None
            bid_price = None
            try:
                ask_price = Decimal(str(ask_price_raw))
                valid_ask = ask_price > 0
            except:
                pass
            try:
                bid_price = Decimal(str(bid_price_raw))
                valid_bid = bid_price > 0
            except:
                pass
            if not valid_ask and trade["type"] == "BUY":
                raise ValueError(f"模拟时 {pair} Ticker 缺少有效 ask 价")
            if not valid_bid and trade["type"] == "SELL":
                raise ValueError(f"模拟时 {pair} Ticker 缺少有效 bid 价")
            if current_currency != trade["from"]:
                raise ValueError(
                    f"货币不匹配：步骤 {i+1} 需要发送 {trade['from']}, 但当前持有 {current_currency}"
                )

            base_curr, quote_curr = market["base"], market["quote"]
            fee, net_amount = Decimal("0"), Decimal("0")
            price_str = "N/A"
            log_entry = f"  {str(i+1):<4s} | "

            if trade["type"] == "BUY":  # 买入 base, 花费 quote
                if current_currency == quote_curr and trade["to"] == base_curr:
                    price = ask_price  # 使用 ask 价买入
                    price_str = format_decimal(price)
                    trade_amount_gross_base = current_amount / price
                    fee = trade_amount_gross_base * fee_rate
                    net_amount = trade_amount_gross_base - fee
                    current_currency = base_curr  # 更新持有货币
                    cost_formatted_str = (
                        step_start_amount_formatted_str  # 花费的是 quote
                    )
                    fee_formatted_str = format_decimal(fee)  # 手续费是 base
                    net_amount_formatted_str = format_decimal(
                        net_amount
                    )  # 收到的是 base
                    log_entry += f"{('买入 ' + base_curr):<14s} | {pair:<12s} | {price_str:<18s} | {(cost_formatted_str + ' ' + quote_curr):<28s} | {(fee_formatted_str + ' ' + base_curr):<28s} | {(net_amount_formatted_str + ' ' + base_curr):<28s}"
                else:
                    raise ValueError(
                        f"买入逻辑错误: 需要 {quote_curr}->{base_curr}, 实际 {current_currency}->{trade['to']}"
                    )

            elif trade["type"] == "SELL":  # 卖出 base, 收到 quote
                if current_currency == base_curr and trade["to"] == quote_curr:
                    price = bid_price  # 使用 bid 价卖出
                    price_str = format_decimal(price)
                    trade_amount_gross_quote = current_amount * price
                    fee = trade_amount_gross_quote * fee_rate
                    net_amount = trade_amount_gross_quote - fee
                    current_currency = quote_curr  # 更新持有货币
                    sold_amount_str = step_start_amount_formatted_str  # 卖出的是 base
                    fee_formatted_str = format_decimal(fee)  # 手续费是 quote
                    net_amount_formatted_str = format_decimal(
                        net_amount
                    )  # 收到的是 quote
                    log_entry += f"{('卖出 ' + base_curr):<14s} | {pair:<12s} | {price_str:<18s} | {(sold_amount_str + ' ' + base_curr):<28s} | {(fee_formatted_str + ' ' + quote_curr):<28s} | {(net_amount_formatted_str + ' ' + quote_curr):<28s}"
                else:
                    raise ValueError(
                        f"卖出逻辑错误: 需要 {base_curr}->{quote_curr}, 实际 {current_currency}->{trade['to']}"
                    )
            else:
                raise ValueError(f"未知交易类型: {trade['type']}")

            log_msgs.append(log_entry)
            current_amount = net_amount  # 更新当前金额
            if current_amount <= 0:
                raise ValueError("模拟中金额变为非正数")

        # --- 验证最终结果 ---
        if current_currency != start_currency:
            # 理论上闭环应该回到起始货币
            raise ValueError(
                f"最终货币 {current_currency} 与起始 {start_currency} 不符"
            )

        final_amount = current_amount
        profit_amount = final_amount - start_amount
        profit_percent = (
            (profit_amount / start_amount) * Decimal("100")
            if start_amount > 0
            else Decimal("0")
        )

        log_msgs.append(f"  结束: {format_decimal(final_amount)} {current_currency}")
        log_msgs.append(
            f"  模拟利润: {format_decimal(profit_amount)} {start_currency} ({profit_percent:.4f}%)"
        )

        # 使用旧的验证阈值
        if profit_percent > config["min_profit_percent_verify"]:
            return {
                "verified": True,
                "profit_percent": profit_percent,
                "profit_amount": profit_amount,
                "final_amount": final_amount,
                "log_messages": log_msgs,
            }
        else:
            return {
                "verified": False,
                "profit_percent": profit_percent,
                "reason": f'利润率 {profit_percent:.4f}% 未达标 ({config["min_profit_percent_verify"]}%)',
                "log_messages": log_msgs,
            }

    except ValueError as ve:
        # logger.debug(f"简单模拟验证环路 {path_str} 时计算错误: {ve}") # 使用 Debug 级别
        error_log = f"  !! 验证计算错误: {ve}"
        if not log_msgs or "步骤" not in log_msgs[1]:  # 检查表头是否存在
            header = f"  {'步骤':<4s} | {'操作':<14s} | {'交易对':<12s} | {'价格':<18s} | {'发送额/量':<28s} | {'手续费':<28s} | {'净收额/量':<28s}"
            separator = "  " + "-" * (len(header) - 2)
            log_msgs.insert(1, header)
            log_msgs.insert(2, separator)
        log_msgs.append(error_log)
        return {
            "verified": False,
            "reason": f"计算错误: {ve}",
            "log_messages": log_msgs,
        }
    except Exception as e:
        logger.warning(
            f"简单模拟验证环路 {path_str} 时发生意外错误: {e}", exc_info=True
        )  # 使用 Warning 级别
        error_log = f"  !! 验证意外错误: {e}"
        if not log_msgs or "步骤" not in log_msgs[1]:
            header = f"  {'步骤':<4s} | {'操作':<14s} | {'交易对':<12s} | {'价格':<18s} | {'发送额/量':<28s} | {'手续费':<28s} | {'净收额/量':<28s}"
            separator = "  " + "-" * (len(header) - 2)
            log_msgs.insert(1, header)
            log_msgs.insert(2, separator)
        log_msgs.append(error_log)
        return {"verified": False, "reason": f"意外错误: {e}", "log_messages": log_msgs}


# --- 模拟完整执行路径 (包括初始/最终闪兑 - 增强版) ---
async def simulate_full_execution_profit(
    cycle_info: dict,
    actual_start_currency: str,  # 你想用什么货币开始模拟，通常是 USDT
    actual_start_amount: Decimal,  # 你打算用多少这种货币开始
    end_with_usdt: bool,  # 是否强制以 USDT 结束模拟
    current_tickers: dict,  # 使用传入的 Ticker 快照
    markets: dict,
    config: dict,
) -> dict:
    """
    (增强版) 模拟完整的潜在执行路径，包括必要的初始闪兑和最终闪兑回 USDT。
    使用传入的 Ticker 快照进行模拟。
    """
    sim_logs = []
    sim_current_currency = actual_start_currency
    sim_current_amount = actual_start_amount
    cycle_start_currency = cycle_info["nodes"][0]
    fee_rate = config["taker_fee_rate"]
    path_str = " -> ".join(cycle_info.get("nodes", ["?"]))

    sim_logs.append(f"--- 全路径模拟开始 ({path_str}) ---")
    sim_logs.append(
        f"实际起始: {format_decimal(actual_start_amount)} {actual_start_currency}"
    )
    sim_logs.append(
        f"目标环路: 从 {cycle_start_currency} 开始, {'强制结束于 USDT' if end_with_usdt else '按环路结束'}"
    )

    trades = cycle_info["trades"]
    start_trade_index = 0  # 核心环路从哪个索引开始模拟

    try:
        # --- 检查是否可以优化跳过第一步 ---
        skip_initial_swap_and_first_step = False
        if trades and len(trades) > 1:  # 至少需要两步才能优化
            first_trade = trades[0]
            # 条件：起始货币 != 环路货币，且第一步是从环路货币换回起始货币
            if (
                sim_current_currency != cycle_start_currency
                and first_trade["from"] == cycle_start_currency
                and first_trade["to"] == actual_start_currency
            ):
                skip_initial_swap_and_first_step = True
                start_trade_index = 1  # 从核心环路的第二个交易开始模拟
                sim_logs.append(
                    f"优化: 检测到环路第一步 ({cycle_start_currency} -> {actual_start_currency}) 与期望起始 ({actual_start_currency}) 抵消。"
                )
                sim_logs.append(
                    f"将跳过初始闪兑和核心环路第一步，直接从持有 {format_decimal(sim_current_amount)} {sim_current_currency} 开始模拟。"
                )

        # --- 1. 模拟初始闪兑 (如果需要且未被优化跳过) ---
        if (
            not skip_initial_swap_and_first_step
            and sim_current_currency != cycle_start_currency
        ):
            sim_logs.append(
                f"步骤 0: 需要将 {sim_current_currency} 转换为环路起始货币 {cycle_start_currency}"
            )
            # ---> 调用模拟闪兑，传入当前 Ticker 快照 <---
            swap_result = await simulate_swap_order(
                None,
                sim_current_currency,
                cycle_start_currency,
                sim_current_amount,
                markets,
                current_tickers,
            )
            if swap_result and swap_result.get("estimated_to_amount", Decimal("0")) > 0:
                received_amount = swap_result["estimated_to_amount"]
                sim_logs.append(
                    f"  - 模拟闪兑 ({sim_current_currency} -> {cycle_start_currency}):"
                )
                for step_log in swap_result["steps"]:
                    sim_logs.append(f"    - {step_log}")
                sim_logs.append(
                    f"  - 闪兑后持有 (估算): {format_decimal(received_amount)} {cycle_start_currency}"
                )
                sim_current_amount = received_amount
                sim_current_currency = cycle_start_currency
            else:
                reason = f"模拟初始闪兑 ({sim_current_currency} -> {cycle_start_currency}) 失败或无路径"
                sim_logs.append(f"!! 错误: {reason}")
                return {
                    "verified": False,
                    "reason": reason,
                    "log_messages": sim_logs,
                    "profit_percent": Decimal("-999"),
                    "final_amount": Decimal(0),
                    "final_currency": sim_current_currency,
                }

        elif not skip_initial_swap_and_first_step:  # 不需要初始闪兑
            sim_logs.append(
                f"步骤 0: 起始货币 ({sim_current_currency}) 已满足环路要求，无需初始闪兑。"
            )

        # --- 2. 模拟核心套利环路 (从 start_trade_index 开始) ---
        if start_trade_index < len(trades):  # 确保还有步骤需要模拟
            # --- 格式化头部 ---
            fee_percentage_str = f"{fee_rate * 100:.4f}%"
            fee_column_label = f"手续费({fee_percentage_str})"
            col_step, col_op, col_pair, col_price, col_sent, col_fee, col_net = (
                4,
                14,
                12,
                18,
                28,
                28,
                28,
            )
            header = f"  {'步':<{col_step}} | {'操作':<{col_op}} | {'交易对':<{col_pair}} | {'价格':<{col_price}} | {'发送':<{col_sent}} | {fee_column_label:<{col_fee}} | {'收到':<{col_net}}"
            separator = "  " + "-" * (len(header) - 2)
            sim_logs.append("--- 核心环路模拟 ---")
            sim_logs.append(header)
            sim_logs.append(separator)
            # -----------------

            for i, trade in enumerate(
                trades[start_trade_index:], start=start_trade_index
            ):
                step_num_display = i + 1  # 用于日志显示的步骤号
                step_start_amount = sim_current_amount
                step_start_amount_formatted_str = format_decimal(sim_current_amount)
                pair = trade["pair"]
                market = markets.get(pair)
                ticker = current_tickers.get(pair)  # 使用传入的快照

                # 验证
                if not market or not ticker:
                    raise ValueError(f"模拟时缺少市场/Ticker数据 for {pair}")
                if sim_current_currency != trade["from"]:
                    raise ValueError(
                        f"货币不匹配：步骤 {step_num_display} 需要 {trade['from']}, 持有 {sim_current_currency}"
                    )

                base_curr, quote_curr = market["base"], market["quote"]
                ask_price_raw = ticker.get("ask")
                bid_price_raw = ticker.get("bid")
                valid_ask = False
                valid_bid = False
                ask_price = None
                bid_price = None
                try:
                    ask_price = Decimal(str(ask_price_raw))
                    valid_ask = ask_price > 0
                except:
                    pass
                try:
                    bid_price = Decimal(str(bid_price_raw))
                    valid_bid = bid_price > 0
                except:
                    pass

                fee, net_amount = Decimal("0"), Decimal("0")
                price_str = "N/A"
                price = None
                log_entry = f"  {str(step_num_display):<{col_step}} | "

                if trade["type"] == "BUY":  # 买 base, 花 quote
                    if not valid_ask:
                        raise ValueError(f"{pair} 缺少有效 ask 价 ({ask_price_raw})")
                    price = ask_price
                    price_str = format_decimal(price)
                    trade_amount_gross_base = sim_current_amount / price
                    fee = trade_amount_gross_base * fee_rate
                    net_amount = trade_amount_gross_base - fee
                    next_currency = base_curr
                    cost_formatted_str = step_start_amount_formatted_str
                    fee_formatted_str = format_decimal(fee)
                    net_amount_formatted_str = format_decimal(net_amount)
                    log_entry += f"{('买入 ' + base_curr):<{col_op}} | {pair:<{col_pair}} | {price_str:<{col_price}} | {(cost_formatted_str + ' ' + quote_curr):<{col_sent}} | {(fee_formatted_str + ' ' + base_curr):<{col_fee}} | {(net_amount_formatted_str + ' ' + base_curr):<{col_net}}"

                elif trade["type"] == "SELL":  # 卖 base, 收 quote
                    if not valid_bid:
                        raise ValueError(f"{pair} 缺少有效 bid 价 ({bid_price_raw})")
                    price = bid_price
                    price_str = format_decimal(price)
                    trade_amount_gross_quote = sim_current_amount * price
                    fee = trade_amount_gross_quote * fee_rate
                    net_amount = trade_amount_gross_quote - fee
                    next_currency = quote_curr
                    sold_amount_str = step_start_amount_formatted_str
                    fee_formatted_str = format_decimal(fee)
                    net_amount_formatted_str = format_decimal(net_amount)
                    log_entry += f"{('卖出 ' + base_curr):<{col_op}} | {pair:<{col_pair}} | {price_str:<{col_price}} | {(sold_amount_str + ' ' + base_curr):<{col_sent}} | {(fee_formatted_str + ' ' + quote_curr):<{col_fee}} | {(net_amount_formatted_str + ' ' + quote_curr):<{col_net}}"
                else:
                    raise ValueError(f"未知交易类型: {trade['type']}")

                sim_logs.append(log_entry)
                sim_current_amount = net_amount
                sim_current_currency = next_currency
                if sim_current_amount <= 0:
                    raise ValueError("模拟中金额变为非正数")

        # --- 3. 模拟最终闪兑回 USDT (如果需要且配置要求) ---
        final_swap_simulated = False
        if end_with_usdt and sim_current_currency != "USDT":
            sim_logs.append(
                f"步骤 F: 需要将最终货币 {sim_current_currency} 转换回 USDT"
            )
            # ---> 调用模拟闪兑，传入当前 Ticker 快照 <---
            final_swap_result = await simulate_swap_order(
                None,
                sim_current_currency,
                "USDT",
                sim_current_amount,
                markets,
                current_tickers,
            )
            if (
                final_swap_result
                and final_swap_result.get("estimated_to_amount", Decimal("0")) > 0
            ):
                received_amount = final_swap_result["estimated_to_amount"]
                sim_logs.append(f"  - 模拟最终闪兑 ({sim_current_currency} -> USDT):")
                for step_log in final_swap_result["steps"]:
                    sim_logs.append(f"    - {step_log}")
                sim_logs.append(
                    f"  - 最终闪兑后持有 (估算): {format_decimal(received_amount)} USDT"
                )
                sim_current_amount = received_amount
                sim_current_currency = "USDT"
                final_swap_simulated = True
            else:
                reason = (
                    f"模拟最终闪兑回 USDT ({sim_current_currency} -> USDT) 失败或无路径"
                )
                sim_logs.append(
                    f"!! 警告: {reason}. 最终结果将以 {sim_current_currency} 计。"
                )
                # 最终闪兑失败，但核心环路可能已完成，利润计算会反映这一点

        elif end_with_usdt and sim_current_currency == "USDT":
            sim_logs.append(f"步骤 F: 最终货币已是 USDT，无需最终闪兑。")
            final_swap_simulated = True

        # --- 4. 计算最终利润 (相对于 actual_start_amount 和 actual_start_currency) ---
        profit_amount = Decimal("NaN")
        profit_percent = Decimal("NaN")
        final_amount = sim_current_amount
        final_currency = sim_current_currency
        verified = False
        reason = "未达标"  # 默认原因

        # 利润计算的目标货币
        profit_target_currency = "USDT" if end_with_usdt else actual_start_currency

        if final_currency == profit_target_currency:
            # 如果起始和最终（目标）货币相同
            if actual_start_currency == profit_target_currency:
                profit_amount = final_amount - actual_start_amount
                if actual_start_amount > 0:
                    profit_percent = (profit_amount / actual_start_amount) * Decimal(
                        "100"
                    )
                else:
                    profit_percent = Decimal("0")
            else:  # 起始不是 USDT，但最终换回了 USDT
                # 这种情况无法直接计算利润率，因为基数不同
                profit_amount = final_amount  # 最终得到的 USDT
                profit_percent = Decimal("NaN")  # 无法计算百分比
                reason = f"起始({actual_start_currency}), 目标(USDT), 无法计算利润率"
                sim_logs.append(f"ℹ️ {reason}")

            # 如果成功计算了百分比，进行验证
            if not profit_percent.is_nan():
                min_profit_req = config.get(
                    "min_profit_full_sim_percent", config["min_profit_percent_verify"]
                )
                if profit_percent > min_profit_req:
                    verified = True
                    reason = f"模拟利润率 {profit_percent:.4f}% 达到要求 (>{min_profit_req}%)"
                    sim_logs.append(f"✅ 验证通过: {reason}")
                else:
                    reason = f"模拟利润率 {profit_percent:.4f}% 未达要求 (>{min_profit_req}%)"
                    sim_logs.append(f"❌ 验证未通过: {reason}")

        else:  # 最终货币与目标不符 (例如最终闪兑失败)
            reason = f"最终模拟货币 ({final_currency}) 与目标 ({profit_target_currency}) 不符"
            sim_logs.append(f"!! 错误: {reason}")

        sim_logs.append(f"--- 全路径模拟结束 ---")
        sim_logs.append(
            f"初始投入 ({actual_start_currency}): {format_decimal(actual_start_amount)}"
        )
        sim_logs.append(
            f"最终持有 ({final_currency})  : {format_decimal(final_amount)}"
        )
        if not profit_percent.is_nan():
            sim_logs.append(
                f"模拟总利润 ({profit_target_currency}) : {format_decimal(profit_amount)} ({profit_percent:+.4f}%)"
            )
        else:
            sim_logs.append(f"模拟总利润: 无法以百分比表示 (起始/目标货币不同)")

        return {
            "verified": verified,
            "profit_percent": (
                profit_percent if not profit_percent.is_nan() else Decimal("-998")
            ),  # 返回特殊值表示无法计算
            "profit_amount": profit_amount,
            "final_amount": final_amount,
            "final_currency": final_currency,
            "log_messages": sim_logs,
            "reason": reason,
        }

    except ValueError as ve:
        error_log = f"!! 模拟计算错误: {ve}"
        sim_logs.append(error_log)
        # logger.debug(f"全路径模拟 {path_str} 时计算错误: {ve}")
        return {
            "verified": False,
            "reason": f"计算错误: {ve}",
            "log_messages": sim_logs,
            "profit_percent": Decimal("-997"),
            "final_amount": Decimal(0),
            "final_currency": sim_current_currency,
        }
    except Exception as e:
        error_log = f"!! 模拟意外错误: {e}"
        sim_logs.append(error_log)
        logger.warning(f"全路径模拟 {path_str} 时发生意外错误: {e}", exc_info=True)
        return {
            "verified": False,
            "reason": f"意外错误: {e}",
            "log_messages": sim_logs,
            "profit_percent": Decimal("-996"),
            "final_amount": Decimal(0),
            "final_currency": sim_current_currency,
        }


# --- Telegram Bot 命令处理函数 ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_chat_id, config, global_balances, websocket_connection_status
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_id = user.id

    # 授权检查
    if AUTHORIZED_USER_ID != 0 and user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("抱歉，您无权使用此机器人。")
        logger.warning(f"未授权用户尝试访问: ID={user_id}, Username={user.username}")
        return

    # 更新或确认 chat_id
    if user_chat_id != chat_id:
        user_chat_id = chat_id
        logger.info(
            f"授权用户 {user.username} (ID: {user_id}) 已连接。Chat ID: {chat_id}"
        )
    else:
        logger.info(f"授权用户 {user.username} (ID: {user_id}) 已在交互中。")

    # 准备状态信息
    config["running"] = True  # 确保计算是运行的 (如果之前暂停了)
    active_chunks = sum(1 for status in websocket_connection_status if status)
    total_chunks = len(websocket_connection_status)
    ws_status_str = (
        f"{active_chunks}/{total_chunks} 连接块活跃"
        if total_chunks > 0
        else "WebSocket未启动"
    )
    balance_status_str = (
        "正在获取..." if not global_balances else f"持有 {len(global_balances)} 种资产"
    )
    last_update_str = "从未"
    if balance_update_task and hasattr(balance_update_task, "last_update_time"):
        delta = time.time() - balance_update_task.last_update_time
        if delta < 60:
            last_update_str = f"{int(delta)}秒前"
        else:
            last_update_str = f"{int(delta/60)}分钟前"
        balance_status_str += f" (更新于 {last_update_str})"

    # 构建欢迎消息
    welcome_message = (
        f"欢迎使用币安套利监控机器人 (v2.1 - 增强报告), {user.mention_html()}!\n\n"
        f"<b>当前状态:</b>\n"
        f"  - 计算循环: {'运行中' if config['running'] else '已暂停'}\n"
        f"  - 自动交易: {'已启用 ✅' if config['auto_trade_enabled'] else '已禁用 ❌'}\n"
        f"  - WebSocket: {ws_status_str}\n"
        f"  - 账户余额: {balance_status_str}\n\n"
        f"<b>可用命令:</b>\n"
        f"  /status   - 查看详细状态和配置\n"
        f"  /set      - 修改配置 (见 /help)\n"
        f"  /trade    - 切换自动交易状态\n"
        f"  /pause    - 暂停套利计算\n"
        f"  /resume   - 恢复套利计算\n"
        f"  /balance  - 查看当前余额详情\n"
        f"  /help     - 显示帮助信息\n"
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID:
        return

    help_text = (
        "<b>币安套利机器人帮助文档 (v2.1)</b>\n\n"
        "<b>基础命令:</b>\n"
        "  <code>/start</code>    - 初始化机器人并显示欢迎信息。\n"
        "  <code>/status</code>   - 查看机器人详细运行状态、统计和当前配置。\n"
        "  <code>/balance</code>  - 显示当前获取到的非零账户余额及估值。\n"
        "  <code>/help</code>     - 显示此帮助信息。\n\n"
        "<b>控制命令:</b>\n"
        "  <code>/trade [on|off]</code> - 启用或禁用<b>自动真实交易</b>功能。<b><u>极其重要，请谨慎操作！</u></b>\n"
        "  <code>/pause</code>    - 暂停套利计算循环 (WebSocket 和余额更新会继续)。\n"
        "  <code>/resume</code>   - 恢复已暂停的套利计算循环。\n"
        "  <code>/toggle_cpp</code> - 快速切换 Bellman-Ford 算法的 C++ 或 Python 实现 (如果 C++ 库可用)。\n\n"
        "<b>配置命令:</b>\n"
        "  <code>/set [参数名] [值]</code> - 修改运行时配置。可用参数包括:\n"
        "    - <code>fee_rate [小数]</code>: 您的实际 Taker 手续费率 (例如 0.00075 代表 0.075%)。\n"
        "    - <code>min_profit [百分比]</code>: 全路径模拟验证的最低盈利阈值 (例如 0.05 代表 0.05%)。\n"
        "    - <code>interval [秒]</code>: 主计算循环的间隔时间 (例如 0.1)。\n"
        "    - <code>depth [整数]</code>: 允许的最大套利路径深度 (例如 5)。\n"
        "    - <code>volume_filter [金额]</code>: 市场24小时最低成交额(计价货币)过滤 (例如 100000)。\n"
        "    - <code>use_cpp [on|off]</code>: 手动指定是否使用 C++ 实现。\n"
        "    - <code>balance_interval [秒]</code>: 账户余额的更新频率 (例如 60)。\n\n"
        "<b>示例:</b>\n"
        "  <code>/set fee_rate 0.00075</code>\n"
        "  <code>/trade off</code>\n\n"
        "<b>注意:</b>\n"
        "  - 自动交易开启后，机器人发现满足条件的机会并通过风险评估后<b>会直接执行真实市价单交易</b>，请确保您完全理解风险并已正确配置手续费率和 API 密钥权限。\n"
        "  - 配置修改即时生效。\n"
        "  - 如果遇到问题，请检查日志或使用 <code>/status</code> 查看状态。"
    )
    # 使用 disable_web_page_preview 替代 LinkPreviewOptions (根据库版本可能需要调整)
    await update.message.reply_text(
        help_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID:
        return

    # --- 声明需要读取的全局变量 ---
    global snap_copy_duration_g, graph_build_duration_g, bf_call_duration_g
    global verification_duration_g, other_duration_g, last_cycle_duration
    # --- 新增: 读取统计变量 ---
    global cycle_count_total, stats_reporting_start_time, last_execution_duration_g

    # --- 状态信息收集 ---
    graph_build_mode = (
        "C++ DLL"
        if config["use_cpp_graph_build"] and cpp_graph_lib_loaded
        else "Python"
    )
    if config["use_cpp_graph_build"] and not cpp_graph_lib_loaded:
        graph_build_mode += " (加载失败!)"
    bf_mode = "C++ DLL" if config["use_cpp_bf"] and cpp_bf_lib_loaded else "Python"
    if config["use_cpp_bf"] and not cpp_bf_lib_loaded:
        bf_mode += " (加载失败!)"
    risk_mode = (
        "C++ DLL"
        if config["use_cpp_risk_assessment"] and cpp_ops_lib_loaded
        else "Python"
    )
    if config["use_cpp_risk_assessment"] and not cpp_ops_lib_loaded:
        risk_mode += " (加载失败!)"
    sim_mode = (
        "C++ DLL"
        if config["use_cpp_full_simulation"] and cpp_ops_lib_loaded
        else "Python"
    )
    if config["use_cpp_full_simulation"] and not cpp_ops_lib_loaded:
        sim_mode += " (加载失败!)"

    time_since_update = "从未"
    if last_ticker_update_time > 0:
        delta = time.time() - last_ticker_update_time
        if delta < 2:
            time_since_update = "刚刚"
        elif delta < 60:
            time_since_update = f"{delta:.0f} 秒前"
        else:
            time_since_update = f"{delta/60:.1f} 分钟前"
    active_chunks = sum(1 for status in websocket_connection_status if status)
    total_chunks = len(websocket_connection_status)
    ws_status_str = (
        f"{active_chunks}/{total_chunks} 连接块活跃" if total_chunks > 0 else "未启动"
    )
    total_symbols_intended = len(websocket_symbols)
    balance_summary = "未获取"
    balance_update_time_str = "未知"
    if global_balances:
        balance_summary = f"持有 {len(global_balances)} 种资产"
        if (
            balance_update_task
            and hasattr(balance_update_task, "last_update_time")
            and balance_update_task.last_update_time > 0
        ):
            delta = time.time() - balance_update_task.last_update_time
            if delta < 5:
                balance_update_time_str = "刚刚"
            elif delta < 120:
                balance_update_time_str = f"{int(delta)}秒前"
            else:
                balance_update_time_str = f"{int(delta/60)}分钟前"
            balance_summary += f" (更新于 {balance_update_time_str})"
        else:
            balance_summary += " (更新时间未知)"
    relax_count_str = (
        str(last_cpp_relaxation_count)
        if last_cpp_relaxation_count is not None
        else "N/A"
    )
    snap_copy_str = (
        f"{snap_copy_duration_g * 1000:.1f} ms"
        if snap_copy_duration_g is not None
        else "N/A"
    )
    graph_build_str = (
        f"{graph_build_duration_g * 1000:.1f} ms"
        if graph_build_duration_g is not None
        else "N/A"
    )
    bf_call_str = (
        f"{bf_call_duration_g * 1000:.1f} ms"
        if bf_call_duration_g is not None
        else "N/A"
    )
    verification_str = (
        f"{verification_duration_g * 1000:.1f} ms"
        if verification_duration_g is not None
        else "N/A"
    )
    other_time_str = (
        f"{other_duration_g * 1000:.1f} ms" if other_duration_g is not None else "N/A"
    )
    loop_time_str = (
        f"{last_cycle_duration*1000:.1f} ms"
        if last_cycle_duration is not None
        else "N/A"
    )

    opp_count = len(last_verified_opportunities)
    last_opp_str = "最近未发现已验证机会。"
    if last_verified_opportunities:
        last_opp_summary = last_verified_opportunities[-1]
        path_str = last_opp_summary.get("nodes_str", "未知路径")
        profit_perc = last_opp_summary.get("profit_percent", Decimal("NaN"))
        profit_str = f"{profit_perc:.4f}%" if not profit_perc.is_nan() else "N/A"
        last_opp_str = f"最新: {path_str} (模拟利润: {profit_str})"

    auto_trade_status_str = (
        "<b>已启用 ✅ (自动执行交易)</b>"
        if config["auto_trade_enabled"]
        else "<b>已禁用 ❌ (不执行交易)</b>"
    )

    # --- 新增：计算 CPS ---
    cps_str = "N/A"
    if stats_reporting_start_time > 0:
        elapsed_seconds = time.time() - stats_reporting_start_time
        if elapsed_seconds > 0.1:  # 避免过早计算和除零
            cps = cycle_count_total / elapsed_seconds
            cps_str = f"{cps:.2f} 周期/秒 (自启动以来)"
        else:
            cps_str = "计算中..."
    # --- 新增：格式化上次执行时间 ---
    last_exec_time_str = (
        f"{last_execution_duration_g:.3f} 秒"
        if last_execution_duration_g is not None
        else "无记录"
    )
    # --------------------

    # --- 构建状态文本 (添加统计部分) ---
    status_text = (
        f"--- 机器人状态 (v2.3 - 带统计) ---\n"
        f"<b>运行控制:</b>\n"
        f"  计算循环: {'运行中' if config['running'] else '已暂停'}\n"
        f"  自动交易: {auto_trade_status_str}\n"
        f"<b>连接与数据:</b>\n"
        f"  WebSocket: {ws_status_str} (目标监听 {total_symbols_intended} 对)\n"
        f"  缓存Tickers: {len(global_tickers)} (最后更新: {time_since_update})\n"
        f"  账户余额: {balance_summary}\n"
        f"<b>性能统计:</b>\n"  # <--- 新增板块
        f"  循环速率: {cps_str}\n"
        f"  上次交易执行耗时: {last_exec_time_str}\n"
        f"  上次计算统计:\n"
        f"    BF松弛次数: {relax_count_str}\n"
        f"    耗时分解 (总计: {loop_time_str}):\n"
        f"      快照复制: {snap_copy_str}\n"
        f"      图构建: {graph_build_str}\n"
        f"      BF 调用: {bf_call_str}\n"
        f"      验证处理: {verification_str}\n"
        f"      其他耗时: {other_time_str}\n"
        f"<b>配置摘要:</b>\n"
        f"  验证利润%: {config['min_profit_full_sim_percent']:.4f}%\n"
        f"  手续费率%: {config['taker_fee_rate'] * 100:.4f}%\n"
        # f"  计算间隔: {config['run_interval_seconds']} 秒 (注: 现由Ticker更新驱动)\n" # 间隔现在不直接控制循环
        f"  最大深度: {config['max_arbitrage_depth']}\n"
        f"  流动性过滤: ≥ {config['min_24h_quote_volume']} Quote\n"
        f"  图构建实现: {graph_build_mode}\n"
        f"  BF实现: {bf_mode}\n"
        f"  风险评估: {risk_mode}\n"
        f"  全路径模拟: {sim_mode}\n"
        f"<b>机会追踪:</b>\n"
        f"  最近发现机会数: {opp_count}\n"
        f"  {last_opp_str}\n"
    )
    await update.message.reply_text(f"{status_text}", parse_mode=ParseMode.HTML)


async def set_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID:
        return
    args = context.args
    if len(args) != 2:
        await update.message.reply_text(
            "用法: /set [参数名] [值]\n例如: <code>/set fee_rate 0.00075</code>\n使用 /help 查看可用参数。",
            parse_mode=ParseMode.HTML,
        )
        return

    param_name = args[0].lower()
    param_value_str = args[1]
    global config

    original_value = None  # 用于记录旧值以便比较
    param_updated = False
    update_message = ""

    try:
        if param_name == "fee_rate":
            original_value = config["taker_fee_rate"]
            new_val = Decimal(param_value_str)
            if 0 <= new_val < 0.05:  # 费率通常不会超过 5%
                config["taker_fee_rate"] = new_val
                param_updated = True
            else:
                update_message = "❌ 费率应 >= 0 且 < 0.05"
        elif param_name == "min_profit":  # 更新为全路径模拟的阈值
            original_value = config["min_profit_full_sim_percent"]
            new_val = Decimal(param_value_str)
            if 0 < new_val < 10:  # 利润阈值通常不会太高
                config["min_profit_full_sim_percent"] = new_val
                param_updated = True
            else:
                update_message = "❌ 利润阈值应 > 0 且 < 10"
        elif param_name == "interval":
            original_value = config["run_interval_seconds"]
            new_val = float(param_value_str)
            if 0.05 <= new_val <= 60:  # 允许 50ms 到 1 分钟
                config["run_interval_seconds"] = new_val
                param_updated = True
            else:
                update_message = "❌ 计算间隔应在 0.05 到 60 秒之间"
        elif param_name == "depth":
            original_value = config["max_arbitrage_depth"]
            new_val = int(param_value_str)
            if 2 <= new_val <= 7:  # 深度限制
                config["max_arbitrage_depth"] = new_val
                param_updated = True
            else:
                update_message = "❌ 最大深度应在 2 到 7 之间"
        elif param_name == "volume_filter":
            original_value = config["min_24h_quote_volume"]
            new_val = Decimal(param_value_str)
            if new_val >= 0:
                config["min_24h_quote_volume"] = new_val
                param_updated = True
                # 注意：这个修改不会自动重新筛选 WS 列表，需要重启或手动触发
                update_message = (
                    "✅ 流动性阈值已更新。注意：需重启机器人以重新筛选监听列表。"
                )
            else:
                update_message = "❌ 成交额阈值必须 >= 0"
        elif param_name == "use_cpp":
            original_value = config["use_cpp_bf"]
            val = param_value_str.lower()
            if val in ["on", "1", "true", "yes"]:
                if cpp_lib_loaded:
                    config["use_cpp_bf"] = True
                    param_updated = True
                else:
                    update_message = "⚠️ C++ DLL 未加载，无法启用。"
            elif val in ["off", "0", "false", "no"]:
                config["use_cpp_bf"] = False
                param_updated = True
            else:
                update_message = "❌ use_cpp 值应为 on 或 off"
        elif param_name == "balance_interval":
            original_value = config["balance_update_interval_seconds"]
            new_val = int(param_value_str)
            if 10 <= new_val <= 3600:  # 10秒到1小时
                config["balance_update_interval_seconds"] = new_val
                param_updated = True
                # 注意：修改间隔需要重启余额更新任务才能生效
                update_message = (
                    "✅ 余额更新间隔已设置。注意：需重启机器人以应用新间隔。"
                )
            else:
                update_message = "❌ 余额更新间隔应在 10 到 3600 秒之间"
        else:
            update_message = (
                f"❌ 未知参数: <code>{param_name}</code>. 使用 /help 查看可用参数。"
            )
            await update.message.reply_text(update_message, parse_mode=ParseMode.HTML)
            return

        # 如果参数成功更新，发送确认消息
        if param_updated:
            new_value = config[param_name]
            # 对 Decimal 和 bool 类型特殊处理显示
            if isinstance(new_value, Decimal):
                new_value_str = f"{new_value:.8f}".rstrip("0").rstrip(".")
            elif isinstance(new_value, bool):
                new_value_str = "on" if new_value else "off"
            else:
                new_value_str = str(new_value)

            if isinstance(original_value, Decimal):
                original_value_str = f"{original_value:.8f}".rstrip("0").rstrip(".")
            elif isinstance(original_value, bool):
                original_value_str = "on" if original_value else "off"
            else:
                original_value_str = str(original_value)

            # 如果 update_message 为空 (表示没有特殊提示)，则使用标准成功消息
            if not update_message:
                update_message = f"✅ 参数 <code>{param_name}</code> 已从 <code>{original_value_str}</code> 更新为 <code>{new_value_str}</code>"
            # 如果 update_message 已有内容 (例如重启提示)，则在其后附加参数变更信息
            elif update_message.startswith("✅"):  # 如果是成功提示
                update_message += f"\n(参数 <code>{param_name}</code> 从 <code>{original_value_str}</code> 更新为 <code>{new_value_str}</code>)"

            await update.message.reply_text(update_message, parse_mode=ParseMode.HTML)
            logger.info(
                f"配置更新 via TG: {param_name} -> {new_value_str} (原: {original_value_str})"
            )
        elif update_message:  # 如果没有更新成功，但有错误消息
            await update.message.reply_text(update_message, parse_mode=ParseMode.HTML)

    except (ValueError, DecimalInvalidOperation) as e:
        await update.message.reply_text(
            f"❌ 无效值格式: '{param_value_str}' for {param_name}. 错误: {e}",
            parse_mode=ParseMode.HTML,
        )
    except KeyError:
        await update.message.reply_text(
            f"❌ 参数名错误: <code>{param_name}</code>", parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await update.message.reply_text(f"❌ 处理出错: {e}", parse_mode=ParseMode.HTML)
        logger.error(f"处理 /set 命令时出错: {e}", exc_info=True)


async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID:
        return
    global config
    if config["running"]:
        config["running"] = False
        await update.message.reply_text(
            "⏸️ 套利计算循环已暂停。(WebSocket 和余额更新仍在后台运行)"
        )
        logger.info("套利计算已被用户暂停。")
    else:
        await update.message.reply_text("计算循环已处于暂停状态。")


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID:
        return
    global config
    if not config["running"]:
        config["running"] = True
        await update.message.reply_text("▶️ 套利计算循环已恢复。")
        logger.info("套利计算已被用户恢复。")
    else:
        await update.message.reply_text("计算循环正在运行中。")


async def toggle_cpp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID:
        return
    global config, cpp_lib_loaded
    if not cpp_lib_loaded:
        await update.message.reply_text(
            "⚠️ C++ 库未加载，无法切换。当前强制使用 Python。"
        )
        config["use_cpp_bf"] = False
        return

    original_state = config["use_cpp_bf"]
    config["use_cpp_bf"] = not original_state
    new_mode = "C++ DLL" if config["use_cpp_bf"] else "Python"
    logger.info(f"配置更新 via TG: Bellman-Ford 实现 -> {new_mode}")
    await update.message.reply_text(
        f"✅ Bellman-Ford 实现已切换为: <b>{new_mode}</b>。", parse_mode=ParseMode.HTML
    )


# --- 切换自动交易命令 ---
async def trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /trade [on|off] 命令"""
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID:
        return
    global config
    args = context.args
    current_state = config["auto_trade_enabled"]

    if not args:  # 如果没有参数，显示当前状态和警告
        status_msg = (
            '<font color="red">已启用 ✅</font> - <b><u>警告: 机器人会自动执行真实交易!</u></b>'
            if current_state
            else '<font color="green">已禁用 ❌</font> - 机器人不会执行真实交易。'
        )
        await update.message.reply_text(
            f"当前自动交易状态: {status_msg}\n使用 <code>/trade on</code> 或 <code>/trade off</code> 进行切换。",
            parse_mode=ParseMode.HTML,
        )
        return

    command = args[0].lower()
    if command == "on":
        if not current_state:
            # --- !! 增加二次确认 !! ---
            keyboard = [
                [
                    InlineKeyboardButton(
                        "确认启用自动交易", callback_data="confirm_trade_on"
                    )
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "⚠️ <b><u>警告!</u></b> 您确定要启用自动交易吗?\n"
                "启用后，机器人将在检测到盈利机会并通过风险评估后<b>自动执行真实市价单交易</b>，这可能导致资金损失。\n"
                "请确保您已仔细检查配置 (特别是手续费率) 并完全理解风险。",
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML,
            )
            # config['auto_trade_enabled'] = True # 不在这里直接启用
            # logger.info("用户请求启用自动交易，等待确认...") # 日志放回调里
            # await update.message.reply_text("✅ 自动交易已启用。\n<b><u>请务必注意风险!</u></b>", parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text("自动交易已处于启用状态。")
    elif command == "off":
        if current_state:
            config["auto_trade_enabled"] = False
            logger.info("自动交易已由用户禁用。")
            await update.message.reply_text(
                "❌ 自动交易已禁用。机器人将不再执行真实交易。",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text("自动交易已处于禁用状态。")
    else:
        await update.message.reply_text(
            "用法: <code>/trade on</code> 或 <code>/trade off</code>",
            parse_mode=ParseMode.HTML,
        )


# --- 回调查询处理 (用于确认启用自动交易) ---
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理内联按钮点击"""
    query = update.callback_query
    await query.answer()  # 告诉 TG 已收到回调
    callback_data = query.data
    logger.info(f"收到按钮回调数据: {callback_data}")
    global config

    try:
        if callback_data == "confirm_trade_on":
            if not config["auto_trade_enabled"]:  # 再次检查，防止重复启用
                config["auto_trade_enabled"] = True
                logger.info("用户已确认，自动交易已启用。")
                await query.edit_message_text(
                    text="✅ 自动交易已确认启用。\n<b><u>请密切监控机器人活动和账户资金!</u></b>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=None,  # 移除按钮
                )
            else:  # 如果已经是启用状态
                await query.edit_message_text(
                    text="自动交易已经被启用了。", reply_markup=None
                )
        # 可以扩展处理其他按钮回调
        # elif callback_data == "...":
        #     pass

    except (RetryAfter, TimedOut, TelegramNetworkError) as e:
        logger.warning(f"编辑 TG 消息时遇到临时网络或速率限制问题: {e}")
    except Exception as e:
        logger.error(f"处理按钮回调时出错: {e}", exc_info=True)
        try:
            # 尝试用新消息回复错误，因为编辑可能失败
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="❌ 处理您的确认请求时发生内部错误。",
            )
            # 尝试编辑原始消息以移除按钮
            await query.edit_message_text(
                text=query.message.text + "\n\n(处理出错)", reply_markup=None
            )
        except Exception:
            pass  # 如果连发送错误消息都失败


# --- 查看余额命令 ---
async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /balance 命令，显示非零余额及估值"""
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID:
        return
    global global_balances, global_tickers, config

    if not global_balances:
        await update.message.reply_text("余额信息尚不可用或未获取成功。请稍后再试。")
        return

    balance_text = "<b>当前可用余额 (非零):</b>\n<pre>"  # 使用 pre 标签保证对齐
    sorted_balances = sorted(
        global_balances.items(), key=lambda item: item[0]
    )  # 按字母排序
    total_usd_value = Decimal("0")
    lines = []
    max_curr_len = 0
    max_amt_len = 0

    # 第一次遍历：计算最大长度和总价值
    temp_balance_data = []
    for currency, amount in sorted_balances:
        if amount > Decimal("1e-12"):  # 过滤掉几乎为零的余额
            usd_value = Decimal("0")
            price_source = ""
            # 尝试估算 USD 价值
            if currency == "USDT":
                usd_value = amount
                price_source = "Stablecoin"
            elif (
                currency in config.get("stablecoin_preference", [])[1:]
            ):  # 其他稳定币近似 1 USD
                usd_value = amount
                price_source = "Stablecoin(Approx)"
            else:
                # 尝试从 Ticker 快照获取价格
                ticker_usdt = f"{currency}/USDT"
                ticker_usdt_rev = f"USDT/{currency}"
                price = None
                try:
                    ticker_data_fwd = global_tickers.get(ticker_usdt)
                    ticker_data_rev = global_tickers.get(ticker_usdt_rev)
                    if (
                        ticker_data_fwd
                        and ticker_data_fwd.get("bid")
                        and Decimal(str(ticker_data_fwd["bid"])) > 0
                    ):
                        price = Decimal(str(ticker_data_fwd["bid"]))
                        price_source = f"{ticker_usdt} bid"
                    elif (
                        ticker_data_rev
                        and ticker_data_rev.get("ask")
                        and Decimal(str(ticker_data_rev["ask"])) > 0
                    ):
                        ask_rev = Decimal(str(ticker_data_rev["ask"]))
                        price = Decimal("1.0") / ask_rev
                        price_source = f"{ticker_usdt_rev} ask"

                    if price:
                        usd_value = amount * price
                except (ValueError, DecimalInvalidOperation, TypeError):
                    price = None  # 价格转换失败
                    price_source = "Error"

            total_usd_value += usd_value
            amount_str = format_decimal(amount, 8)
            usd_value_str = (
                f" (~${format_decimal(usd_value, 2)})" if usd_value > 0.01 else ""
            )  # 只显示有意义的估值

            max_curr_len = max(max_curr_len, len(currency))
            max_amt_len = max(max_amt_len, len(amount_str))
            temp_balance_data.append(
                {"curr": currency, "amt_str": amount_str, "usd_str": usd_value_str}
            )

    # 第二次遍历：格式化输出
    for item in temp_balance_data:
        line = f"{item['curr']:<{max_curr_len}} : {item['amt_str']:>{max_amt_len}}{item['usd_str']}"
        lines.append(line)

    balance_text += "\n".join(lines)
    balance_text += "</pre>"  # 结束 pre 标签
    balance_text += f"\n<b>总估值:</b> ~${format_decimal(total_usd_value, 2)}"

    # 获取最后更新时间
    update_time_str = "未知"
    if balance_update_task and hasattr(balance_update_task, "last_update_time"):
        delta = time.time() - balance_update_task.last_update_time
        if delta < 5:
            update_time_str = "刚刚"
        elif delta < 120:
            update_time_str = f"{int(delta)}秒前"
        else:
            update_time_str = f"{int(delta/60)}分钟前"
    balance_text += f"\n(余额最后更新: {update_time_str})"

    await update.message.reply_text(balance_text, parse_mode=ParseMode.HTML)


# --- Telegram 错误处理 ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """记录错误并尽可能通知用户。"""
    logger.error("处理 Telegram 更新时发生异常:", exc_info=context.error)

    # 尝试获取更具体的错误信息
    tb_list = traceback.format_exception(
        None, context.error, context.error.__traceback__
    )
    tb_string = "".join(tb_list)
    error_message_short = f"{context.error.__class__.__name__}: {context.error}"

    # 准备给用户的消息
    user_message = (
        f"处理您的请求时发生内部错误。\n"
        f"错误类型: <pre>{error_message_short}</pre>\n"
        f"如果您反复看到此消息，请联系管理员或检查日志。"
    )
    # 准备给管理员（如果设置了）的详细消息
    admin_message = (
        f"处理更新时发生异常:\n"
        f"错误: <pre>{error_message_short}</pre>\n"
        f"Update: <pre>{update}</pre>\n"
        # f"Traceback (结尾部分):\n<pre>{traceback_preview}</pre>" # Traceback 可能过长
    )

    # 尝试通知用户
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=user_message,
                parse_mode=ParseMode.HTML,
            )
        except Exception as send_error:
            logger.error(f"无法向用户发送错误通知: {send_error}")

    # 如果是授权用户遇到的错误，也发送详细信息
    if user_chat_id:  # 使用全局 user_chat_id 发送详细错误
        try:
            # 分割长消息
            max_len = 4000
            # 发送基本信息
            parts = [
                admin_message[i : i + max_len]
                for i in range(0, len(admin_message), max_len)
            ]
            for part in parts:
                await context.bot.send_message(
                    chat_id=user_chat_id, text=part, parse_mode=ParseMode.HTML
                )
            # 单独发送 Traceback 摘要
            traceback_preview = tb_string[-(max_len - 100) :]  # 留下空间给提示信息
            await context.bot.send_message(
                chat_id=user_chat_id,
                text=f"Traceback (结尾部分):\n<pre>{traceback_preview}</pre>",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"无法向授权用户发送详细错误追溯信息: {e}")
            logger.error(f"原始错误追溯:\n{tb_string}")  # 仍在日志中记录完整追溯


# --- 主套利循环 (使用全路径模拟验证，并触发带价格捕获的执行器) ---
# --- 主套利循环 (使用全路径模拟验证，并触发带价格捕获的执行器) ---
async def main_arbitrage_loop(
    application: Application, exchange: ccxtpro.Exchange, markets: dict
):
    """主要的套利计算循环 (异步, 使用全局 Tickers) - 增加了详细的内部计时"""
    global config, last_verified_opportunities, user_chat_id, global_tickers
    global websocket_connection_status, last_cycle_duration, is_trading_active
    global current_execution_task
    # --- 新增: 声明要修改的全局耗时变量 ---
    global snap_copy_duration_g, graph_build_duration_g, bf_call_duration_g
    global verification_duration_g, other_duration_g
    global cpp_graph_lib_loaded, cpp_bf_lib_loaded  # <--- 添加 DLL 加载状态
    global cycle_count_total
    logger.info("主套利计算循环启动...")
    last_seen_update_time = 0  # <--- 新增: 用于跟踪上次处理的 Ticker 更新时间戳
    loop_counter = 0  # 添加计数器
    while True:
        loop_counter += 1
        logger.debug(f"--- 开始套利循环第 {loop_counter} 次迭代 ---")  # <--- 添加日志
        loop_start_time = time.time()  # 记录循环开始时间

        # --- 检查运行状态和 WS 连接 ---
        if not config["running"]:
            await asyncio.sleep(1)  # 暂停时短暂休眠
            continue

        is_any_ws_connected = any(status for status in websocket_connection_status)
        current_update_time = last_ticker_update_time  # 读取当前的全局更新时间

        if not is_any_ws_connected or current_update_time == last_seen_update_time:
            # 如果 WS 未连接，或者当前的全局更新时间戳与上次处理的时间戳相同
            if loop_counter % 100 == 1:  # 降低日志频率
                if not is_any_ws_connected:
                    active_chunks = sum(1 for s in websocket_connection_status if s)
                    total_chunks = len(websocket_connection_status)
                    logger.debug(
                        f"等待 WebSocket ({active_chunks}/{total_chunks} 活跃) 连接..."
                    )
                elif current_update_time == 0:
                    logger.debug("等待首次 Ticker 数据...")
                else:
                    logger.debug("等待新的 Ticker 更新...")  # 表示上次循环后没有新数据

            await asyncio.sleep(0.01)  # <--- 短暂休眠，避免空转 CPU，快速响应更新
            continue  # 重新检查状态和时间戳

        # --- 如果有新 Ticker 更新，则开始处理 ---
        logger.debug(
            f"--- 检测到新 Ticker 数据 (时间戳: {current_update_time:.3f})，开始套利计算周期 {loop_counter} ---"
        )
        last_seen_update_time = current_update_time  # <--- 更新看到的最新时间戳
        loop_start_time = time.time()  # 记录计算周期开始时间

        # --- 计时点 1: 获取当前 Tickers 快照 ---
        snap_copy_start = time.time()
        current_tickers_snapshot = global_tickers.copy()  # 在确认有新数据后再复制
        snap_copy_end = time.time()
        snap_copy_duration = snap_copy_end - snap_copy_start
        snap_copy_duration_g = snap_copy_duration
        logger.debug(
            f"[{datetime.now():%H:%M:%S.%f}] Ticker 快照复制耗时: {snap_copy_duration*1000:.2f} ms"
        )

        if not current_tickers_snapshot:
            logger.warning(
                "获取到的 Ticker 快照为空 (可能在等待和复制之间数据被清空?)，跳过本轮计算。"
            )
            continue  # 跳过这轮计算，等待下一轮 ticker 更新

        # 初始化本轮计时变量
        graph_build_duration = 0
        bf_call_duration = 0
        verification_duration = 0
        negative_cycles = []  # 初始化为空列表

        try:
            # --- 计时点 2: 构建图 (放入 Executor 或直接异步?) ---
            loop = asyncio.get_running_loop()
            graph_build_start = time.time()
            build_result = None  # 初始化 build_result

            # --- !! 修改点: 根据配置选择图构建方式 !! ---
            if config.get("use_cpp_graph_build", False) and cpp_graph_lib_loaded:
                # logger.debug("使用 C++ DLL 构建图...") # 可以取消注释
                build_result = await loop.run_in_executor(
                    None,
                    build_graph_cpp_wrapper,  # <--- 调用 C++ 包装器
                    markets,
                    current_tickers_snapshot,
                    config,
                )
            else:
                if not config.get("use_cpp_graph_build", False):
                    pass  # logger.debug("使用 Python 构建图 (配置禁用 C++)。")
                elif not cpp_graph_lib_loaded:
                    logger.debug("使用 Python 构建图 (图构建 C++ 库未加载或加载失败)。")

                build_result = await loop.run_in_executor(
                    None,
                    build_arbitrage_graph,  # <--- 调用原始 Python 函数
                    markets,
                    current_tickers_snapshot,
                )
            graph_build_end = time.time()  # <--- 修改: 构建图结束计时
            graph_build_duration = (
                graph_build_end - graph_build_start
            )  # <--- 修改: 计算构建图耗时
            graph_build_duration_g = graph_build_duration  # <--- 更新全局变量
            # logger.debug(f"[{datetime.now():%H:%M:%S.%f}] 图构建完成，耗时: {graph_build_duration*1000:.2f} ms") # <--- 新增计时日志

            graph_edges, index_to_currency, currency_to_index = (
                build_result if build_result else (None, None, None)
            )

            if graph_edges and index_to_currency and currency_to_index:
                # --- 计时点 3: 查找负环 ---
                bf_call_start = time.time()
                negative_cycles = await loop.run_in_executor(
                    None,
                    find_negative_cycles_bellman_ford,
                    graph_edges,
                    index_to_currency,
                    currency_to_index,
                    markets,
                )
                bf_call_end = time.time()
                bf_call_duration = bf_call_end - bf_call_start
                bf_call_duration_g = bf_call_duration  # <--- 更新全局变量
                # logger.debug(f"[{datetime.now():%H:%M:%S.%f}] Bellman-Ford 调用完成，耗时: {bf_call_duration*1000:.2f} ms. 找到 {len(negative_cycles) if negative_cycles is not None else 'None'} 个候选环路.")
            else:
                logger.debug(
                    f"[{datetime.now():%H:%M:%S.%f}] 图构建失败或结果为空，跳过 Bellman-Ford。"
                )
                bf_call_duration_g = 0  # 明确设为 0

            # --- 计时点 4: 验证并处理机会 (使用全路径模拟) ---
            verification_start = time.time()  # <--- 新增: 验证处理开始计时
            current_verified_summary = []  # 存储本轮验证通过的机会摘要

            if negative_cycles:
                # logger.info(f"发现 {len(negative_cycles)} 个候选负环. 开始全路径模拟验证...") # <--- INFO 级别，保持

                verification_tasks = []
                for cycle in negative_cycles:
                    # --- 全路径模拟设置 ---
                    actual_start_curr = "USDT"
                    sim_start_amt_usdt = config["simulation_start_amount"]
                    end_with_usdt_sim = True
                    simulation_func = (
                        simulate_full_cpp_wrapper
                        if config.get("use_cpp_full_simulation", False)
                        and cpp_ops_lib_loaded
                        else simulate_full_execution_profit
                    )
                    # 创建模拟任务
                    task = asyncio.create_task(
                        simulation_func(  # <--- 调用选定的函数
                            cycle_info=cycle,
                            actual_start_currency=actual_start_curr,
                            actual_start_amount=sim_start_amt_usdt,
                            end_with_usdt=end_with_usdt_sim,
                            current_tickers=current_tickers_snapshot,  # 使用快照
                            markets=markets,
                            config=config,
                        ),
                        name=f"SimFull-{cycle.get('nodes',['?'])[0]}-{int(time.time()*1000)}",
                    )
                verification_tasks.append((task, cycle))

                # --- 等待所有模拟任务完成 ---
                if verification_tasks:  # 仅当有任务时才 gather
                    return_exceptions = True
                    results = await asyncio.gather(
                        *(task for task, _ in verification_tasks),
                        return_exceptions=True,
                    )
                else:
                    results = []

                # --- 处理模拟结果 ---
                for i, result_or_exception in enumerate(results):
                    original_cycle = verification_tasks[i][1]
                    # 安全地获取路径字符串
                    path_nodes = original_cycle.get("nodes", ["未知路径"])
                    path_str_log = " -> ".join(path_nodes)

                    if isinstance(result_or_exception, Exception):
                        logger.error(
                            f"模拟完整路径 {path_str_log} 时任务异常: {result_or_exception}",
                            exc_info=result_or_exception,
                        )
                        tb_list = traceback.format_exception(
                            None, result_or_exception, result_or_exception.__traceback__
                        )
                        logger.error("详细 Traceback:\n" + "".join(tb_list))
                        continue  # 处理下一个结果

                    # --- 获取模拟结果字典 ---
                    verification_result = result_or_exception
                    # 增加对 None 或非字典结果的检查 (防御性编程)
                    if not isinstance(verification_result, dict):
                        logger.error(
                            f"模拟任务 {path_str_log} 返回了非字典类型的结果: {type(verification_result)} - {verification_result}"
                        )
                        continue

                    if verification_result.get(
                        "error"
                    ):  # 检查 C++ 或包装器返回的 'error' 字段
                        logger.error(
                            f"模拟完整路径 {path_str_log} 时 C++/包装器返回错误: {verification_result.get('reason', verification_result.get('error'))}"
                        )
                        continue
                    if verification_result and verification_result.get("verified"):
                        # 模拟验证通过
                        profit_perc = verification_result["profit_percent"]
                        profit_amount = verification_result["profit_amount"]
                        final_currency = verification_result["final_currency"]

                        # 记录通过验证的摘要
                        cycle_summary = {
                            "nodes_str": path_str_log,
                            "profit_percent": profit_perc,
                            "profit_amount": profit_amount,  # 记录模拟利润额
                            "final_currency": final_currency,  # 记录模拟最终货币
                        }
                        current_verified_summary.append(cycle_summary)

                        logger.info(
                            f"✅ 全路径模拟验证成功: {path_str_log} (模拟利润: {profit_perc:.4f}% vs USDT)"
                        )

                        # --- 检查是否启用自动交易 ---
                        if config["auto_trade_enabled"]:
                            # --- 风险评估 (仍然需要) ---
                            risk_assessment_func = (
                                assess_risk_cpp_wrapper
                                if config.get("use_cpp_risk_assessment", False)
                                and cpp_ops_lib_loaded
                                else assess_arbitrage_risk
                            )
                            logger.debug(
                                f"使用 {'C++' if risk_assessment_func == assess_risk_cpp_wrapper else 'Python'} 进行风险评估 for {path_str_log}"
                            )
                            risk_assessment_start_currency = (
                                path_nodes[0] if path_nodes else "?"
                            )  # 安全获取起始货币
                            risk_assessment_start_amount_est = config[
                                "simulation_start_amount"
                            ]  # 假设从配置的USDT开始
                            if (
                                risk_assessment_start_currency != "USDT"
                                and risk_assessment_start_currency != "?"
                            ):
                                # 模拟初始闪兑以估算风险评估的起始量
                                initial_swap_sim_risk = await simulate_swap_order(
                                    None,
                                    "USDT",
                                    risk_assessment_start_currency,
                                    risk_assessment_start_amount_est,
                                    markets,
                                    current_tickers_snapshot,
                                )
                                if (
                                    initial_swap_sim_risk
                                    and initial_swap_sim_risk.get(
                                        "estimated_to_amount", 0
                                    )
                                    > 0
                                ):
                                    risk_assessment_start_amount_est = (
                                        initial_swap_sim_risk["estimated_to_amount"]
                                    )
                                else:
                                    logger.warning(
                                        f"风险评估: 无法模拟初始闪兑 USDT -> {risk_assessment_start_currency}。跳过风险评估和执行。"
                                    )
                                    continue  # 决定跳过此机会

                            logger.info(
                                f"自动交易启用，评估路径风险: {path_str_log} (估算起始: {format_decimal(risk_assessment_start_amount_est)} {risk_assessment_start_currency})"
                            )
                            # --- 单独计时风险评估 (可选, 如果需要可以加入) ---
                            risk_assess_start = time.time()
                            risk_assessment = await risk_assessment_func(  # <--- 调用选定的函数
                                original_cycle,
                                risk_assessment_start_amount_est,  # 使用估算的起始金额
                                exchange,  # C++ wrapper 也需要 exchange 来获取订单簿
                                markets,
                                current_tickers_snapshot,  # C++ wrapper 也需要 Ticker
                                config,
                            )

                            if risk_assessment and risk_assessment.get(
                                "error"
                            ):  # 检查 C++ 或包装器错误
                                logger.error(
                                    f"风险评估 {path_str_log} 时 C++/包装器返回错误: {risk_assessment.get('reason', risk_assessment.get('error'))}"
                                )
                                continue  # 跳过此机会
                            risk_assess_end = time.time()
                            logger.info(
                                f"风险评估耗时: {(risk_assess_end - risk_assess_start)*1000:.2f} ms for {path_str_log}"
                            )

                            if risk_assessment["is_viable"]:
                                profit_after_slippage_est = risk_assessment[
                                    "estimated_profit_percent_after_slippage"
                                ]
                                logger.info(
                                    f"风险评估通过 (滑点后预估利润: {profit_after_slippage_est:.4f}% vs {config['min_profit_after_slippage_percent']}% 要求). 尝试执行..."
                                )

                                # --- 检查是否有正在执行的任务 ---
                                if (
                                    current_execution_task
                                    and not current_execution_task.done()
                                ):
                                    logger.info(
                                        f"已有交易执行任务 '{current_execution_task.get_name()}' 正在运行，跳过本次机会: {path_str_log}"
                                    )
                                else:
                                    # --- 触发真实交易 ---
                                    logger.info(
                                        f"*** 准备创建并启动真实交易任务: {path_str_log} ***"
                                    )
                                    if user_chat_id:  # <-- 只在这里发送执行通知
                                        try:
                                            await application.bot.send_message(
                                                chat_id=user_chat_id,
                                                text=f"🤖 检测到机会 (模拟利润 {profit_perc:.4f}% vs USDT)，自动交易开启，风险评估通过。\n<b>开始执行真实交易...</b>\n路径: `{path_str_log}`",
                                                parse_mode=ParseMode.HTML,
                                            )
                                        except Exception as tg_e:
                                            logger.warning(f"发送执行通知失败: {tg_e}")

                                    # 创建并存储执行任务引用
                                    current_execution_task = asyncio.create_task(
                                        execute_arbitrage_path(
                                            exchange,
                                            original_cycle,
                                            markets,
                                            application,
                                        ),
                                        name=f"ArbitrageExec-{int(time.time())}",
                                    )
                                    logger.info(
                                        f"交易任务已启动，结束本轮机会查找以避免冲突。"
                                    )
                                    # 跳出内层 for result_or_exception in results 循环
                                    break  # <--- 结束本轮机会查找

                            else:  # 风险评估未通过
                                logger.warning(
                                    f"风险评估未通过，取消执行路径: {path_str_log}"
                                )
                                reasons_str = (
                                    "; ".join(risk_assessment.get("reasons", []))
                                    if risk_assessment.get("reasons")
                                    else "未知原因"
                                )
                                logger.warning(f"  - 原因: {reasons_str}")

                        # --- 自动交易禁用时 ---
                        elif user_chat_id:  # 自动交易禁用，只记录日志，不通知用户
                            logger.info(
                                f"发现已验证机会 (自动交易禁用，不通知用户): {path_str_log}, 模拟利润: {profit_perc:.4f}% vs USDT"
                            )
                        else:  # 自动交易禁用且无用户
                            logger.info(
                                f"发现已验证机会: {path_str_log}, 模拟利润: {profit_perc:.4f}% vs USDT (自动交易禁用)"
                            )

                    elif verification_result:  # 模拟完成但未验证通过 (verified=False)
                        reason = verification_result.get("reason", "未知原因")
                        logger.debug(f"全路径模拟未通过: {path_str_log} ({reason})")

            # --- 验证处理结束 ---
            verification_end = time.time()  # <--- 新增: 验证处理结束计时
            verification_duration = (
                verification_end - verification_start
            )  # <--- 新增: 计算验证处理耗时
            verification_duration_g = verification_duration  # <--- 更新全局变量
            logger.debug(
                f"[{datetime.now():%H:%M:%S.%f}] 验证处理完成，耗时: {verification_duration*1000:.2f} ms"
            )  # <--- 新增计时日志

            # --- 更新全局的最近机会列表 (现在存储的是摘要) ---
            last_verified_opportunities = current_verified_summary
            if not current_verified_summary and negative_cycles:
                logger.debug("所有候选环路均未通过完整路径模拟验证。")
            cycle_count_total += 1  # 在计算成功完成后增加计数
        # --- 异常处理 ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
            logger.warning(f"计算循环中网络或交易所错误: {type(e).__name__} - {e}。")
            await asyncio.sleep(
                max(config["run_interval_seconds"], 1.0)
            )  # 短暂等待后重试
            continue  # 跳过本轮剩余部分
        except Exception as e:
            logger.error(f"主计算循环发生意外错误: {e}", exc_info=True)
            await asyncio.sleep(
                config["run_interval_seconds"] * 5
            )  # 发生错误时等待更久
            continue  # 跳过本轮剩余部分

        # --- 循环结束，计算等待时间 ---
        loop_end_time = time.time()
        loop_duration = loop_end_time - loop_start_time
        last_cycle_duration = loop_duration  # 更新全局循环耗时

        # --- 计算并更新 other_duration_g ---
        # 使用刚刚更新的全局变量来计算
        other_duration = loop_duration - (
            (snap_copy_duration_g or 0)
            + (graph_build_duration_g or 0)
            + (bf_call_duration_g or 0)
            + (verification_duration_g or 0)
        )
        other_duration_g = other_duration  # <--- 更新全局变量

        # --- 打印详细耗时日志 (Debug 级别) ---
        logger.debug(
            f"循环耗时分解 (总计 {loop_duration*1000:.1f} ms): "
            f"快照复制={(snap_copy_duration_g or 0)*1000:.1f}ms, "
            f"图构建={(graph_build_duration_g or 0)*1000:.1f}ms, "
            f"BF调用={(bf_call_duration_g or 0)*1000:.1f}ms, "
            f"验证处理={(verification_duration_g or 0)*1000:.1f}ms, "
            f"其他={(other_duration_g or 0)*1000:.1f}ms"
        )
        """
        wait_time = max(0, config["run_interval_seconds"] - loop_duration)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        """


# --- 主函数入口 (增强版) ---
async def main():
    """设置并运行 Telegram Bot, WebSocket 监控, 余额更新和套利循环"""
    global cpp_lib_loaded, config, websocket_symbols, user_chat_id, markets, exchange
    global websocket_connection_status, ticker_watch_tasks, balance_update_task, AUTHORIZED_USER_ID
    global cpp_bf_lib_loaded, cpp_graph_lib_loaded, config  # 添加 DLL 加载状态
    global cycle_count_total, stats_reporting_start_time
    cycle_count_total = 0
    stats_reporting_start_time = time.time()  # 记录机器人启动（或统计开始）的时间
    if not cpp_bf_lib_loaded and config["use_cpp_bf"]:
        logger.warning(
            "警告：配置使用 Bellman-Ford C++ 但加载失败，将自动切换到 Python 实现。"
        )
        config["use_cpp_bf"] = False
    if not cpp_graph_lib_loaded and config["use_cpp_graph_build"]:
        logger.warning(
            "警告：配置使用图构建 C++ 但加载失败，将自动切换到 Python 实现。"
        )
        config["use_cpp_graph_build"] = False
    if not cpp_ops_lib_loaded and config["use_cpp_risk_assessment"]:
        logger.warning(
            "警告：配置使用 C++ 风险评估但加载失败，将自动切换到 Python 实现。"
        )
        config["use_cpp_risk_assessment"] = False
    if not cpp_ops_lib_loaded and config["use_cpp_full_simulation"]:
        logger.warning(
            "警告：配置使用 C++ 全路径模拟但加载失败，将自动切换到 Python 实现。"
        )
        config["use_cpp_full_simulation"] = False
    # --- 基本配置检查 ---
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
        logger.critical("错误：必须设置 TELEGRAM_BOT_TOKEN！程序退出。")
        return
    if (
        not API_KEY
        or not API_SECRET
        or API_KEY == "YOUR_API_KEY"
        or API_SECRET == "YOUR_SECRET_KEY"
    ):
        logger.critical("错误：必须设置有效的币安 API_KEY 和 API_SECRET！程序退出。")
        return
    # 强制将 AUTHORIZED_USER_ID 转为整数
    try:
        AUTH_ID_INT = int(AUTHORIZED_USER_ID)
        AUTHORIZED_USER_ID = AUTH_ID_INT  # 更新全局变量
    except ValueError:
        logger.critical(
            f"错误：AUTHORIZED_USER_ID ('{AUTHORIZED_USER_ID}') 必须是一个有效的整数！程序退出。"
        )
        return

    if AUTHORIZED_USER_ID == 0:
        logger.warning(
            "警告：AUTHORIZED_USER_ID 设置为 0，意味着 *任何人* 都可以与此机器人交互并可能控制交易！强烈建议设置为您自己的 Telegram 用户 ID。"
        )
    else:
        logger.info(f"授权用户 ID: {AUTHORIZED_USER_ID}")

    if config["auto_trade_enabled"]:
        logger.warning(
            "!! 警告：自动交易 (auto_trade_enabled) 在启动时配置为 True！请确认 /trade 命令的状态并了解风险。!!"
        )

    if not cpp_bf_lib_loaded and config["use_cpp_bf"]:
        logger.warning("警告：配置使用 C++ DLL 但加载失败，将自动切换到 Python 实现。")
        config["use_cpp_bf"] = False

    # --- 连接交易所并加载市场 ---
    exchange, markets = await connect_xt_pro()
    if exchange is None or markets is None:
        logger.critical("无法连接币安或加载市场，程序退出。")
        return

    # --- 筛选用于监听的交易对 (使用分批获取 Ticker) ---
    logger.info("开始筛选用于 WebSocket 监听的交易对...")
    all_spot_symbols = [
        s
        for s, m in markets.items()
        if m.get("spot") and m.get("active") and "/" in s and not m.get("prediction")
    ]
    logger.info(f"交易所共有 {len(all_spot_symbols)} 个活跃现货交易对。")

    # --- 步骤 2: 分批获取 Ticker 数据用于流动性过滤 ---
    all_tickers = {}
    batch_size = config.get("ticker_batch_size", 500)  # 从配置获取批次大小
    logger.info(f"开始分批获取 Ticker 数据用于流动性过滤 (每批最多 {batch_size} 个)...")
    start_fetch_time = time.time()
    fetch_success = True  # 标记整体获取是否成功

    for i in range(0, len(all_spot_symbols), batch_size):
        batch_symbols = all_spot_symbols[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = math.ceil(len(all_spot_symbols) / batch_size)
        logger.debug(
            f"获取流动性 Ticker 批次 {batch_num}/{total_batches} ({len(batch_symbols)} 对)..."
        )
        try:
            tickers_batch = await exchange.fetch_tickers(batch_symbols)
            if isinstance(tickers_batch, dict):
                all_tickers.update(tickers_batch)
            else:
                logger.error(
                    f"获取 Ticker 批次 {batch_num} 时返回了非字典类型: {type(tickers_batch)}。"
                )
                fetch_success = False
                break  # 获取失败则中止

        except ccxt.RateLimitExceeded as e:
            logger.warning(
                f"流动性 Ticker 批次 {batch_num} 速率限制: {e}。等待后重试..."
            )
            try:
                await asyncio.sleep(
                    exchange.rateLimit / 1000 * 1.5 + random.uniform(0.1, 0.5)
                )  # 等待并加入随机
                tickers_batch = await exchange.fetch_tickers(batch_symbols)
                if isinstance(tickers_batch, dict):
                    all_tickers.update(tickers_batch)
                    logger.info(f"批次 {batch_num} 重试成功。")
                else:
                    logger.error(f"重试批次 {batch_num} 时返回了非字典类型。")
                    fetch_success = False
                    break
            except Exception as retry_e:
                logger.error(f"流动性 Ticker 批次 {batch_num} 重试失败: {retry_e}")
                fetch_success = False
                break
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
            logger.error(
                f"流动性 Ticker 批次 {batch_num} 网络或交易所错误: {type(e).__name__} - {e}。中止获取。"
            )
            fetch_success = False
            break
        except ccxt.ExchangeError as e:  # 包括可能的认证或权限问题
            logger.error(f"流动性 Ticker 批次 {batch_num} 交易所错误: {e}。中止获取。")
            fetch_success = False
            break
        except Exception as e:
            logger.error(f"流动性 Ticker 批次 {batch_num} 意外错误: {e}", exc_info=True)
            fetch_success = False
            break

    end_fetch_time = time.time()
    fetch_duration = end_fetch_time - start_fetch_time

    if not all_tickers or not fetch_success:
        logger.error("获取 Ticker 数据失败或不完整，无法进行流动性过滤，程序退出。")
        await exchange.close()
        return
    else:
        logger.info(
            f"分批获取 {len(all_tickers)} 个 Ticker 数据完成，耗时 {fetch_duration:.2f} 秒。"
        )

    # --- 执行流动性过滤 ---
    # 优先考虑 USDT, BUSD (如果还存在), USDC, TUSD 等常见稳定币作为计价货币
    quote_priority = [
        "USDT",
        "USDC",
    ]
    quote_currencies_filtered = {q for q in quote_priority if q in exchange.currencies}
    # 可以添加其他高流动性计价货币如 BTC, ETH, BNB
    quote_currencies_filtered.update(["BTC", "ETH", "BNB"])
    logger.info(
        f"将使用以下计价货币进行筛选: {', '.join(sorted(list(quote_currencies_filtered)))}"
    )

    symbols_quote_filtered = [
        s
        for s in all_spot_symbols
        if markets[s].get("quote") in quote_currencies_filtered
    ]
    logger.info(f"按计价货币筛选后剩余 {len(symbols_quote_filtered)} 对。")

    # 应用 24h 成交额过滤
    min_volume = config.get("min_24h_quote_volume", Decimal("0"))
    logger.info(f"开始流动性过滤 (阈值 >= {min_volume} 计价货币)...")
    liquid_symbols = []
    missing_vol, low_vol, vol_error = 0, 0, 0
    for symbol in symbols_quote_filtered:
        ticker = all_tickers.get(symbol)
        if ticker and "quoteVolume" in ticker and ticker["quoteVolume"] is not None:
            try:
                quote_vol = Decimal(str(ticker["quoteVolume"]))
                if quote_vol >= min_volume:
                    liquid_symbols.append(symbol)
                else:
                    low_vol += 1
            except (ValueError, DecimalInvalidOperation):
                vol_error += 1  # 无法转换成交额
        else:
            missing_vol += 1  # Ticker 中缺少成交额信息

    logger.info(
        f"流动性过滤完成: {len(liquid_symbols)} 对通过, {low_vol} 对过低, {missing_vol} 对无数据, {vol_error} 对数据错误。"
    )
    websocket_symbols_final = liquid_symbols

    if not websocket_symbols_final:
        logger.critical(
            "错误：经过所有筛选后没有剩余的交易对可供监听。请检查过滤条件或市场状态。程序退出。"
        )
        await exchange.close()
        return

    # --- 随机打乱顺序（可选，但可能有助于分散 WS 连接负载） ---
    random.shuffle(websocket_symbols_final)

    websocket_symbols = websocket_symbols_final  # 更新全局变量
    logger.info(f"最终确定目标监听 {len(websocket_symbols)} 个交易对。")

    # --- 设置 Telegram Bot ---
    # defaults = Defaults(parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    defaults = Defaults(
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .defaults(defaults)
        .rate_limiter(AIORateLimiter())  # 使用内置速率限制器
        .build()
    )
    # 存储共享对象到 bot_data
    application.bot_data["exchange"] = exchange
    application.bot_data["markets"] = markets

    # 注册命令和回调处理程序
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("set", set_command))
    application.add_handler(CommandHandler("pause", pause_command))
    application.add_handler(CommandHandler("resume", resume_command))
    application.add_handler(CommandHandler("toggle_cpp", toggle_cpp_command))
    application.add_handler(CommandHandler("trade", trade_command))
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CallbackQueryHandler(button_callback))  # 处理按钮回调
    application.add_handler(
        MessageHandler(filters.COMMAND & (~filters.UpdateType.EDITED), help_command)
    )  # 处理未知命令
    application.add_error_handler(error_handler)

    # --- 准备并创建后台任务 ---
    # 1. Ticker 监听任务 (分块)
    chunk_size = config["websocket_chunk_size"]
    symbol_chunks = [
        websocket_symbols[i : i + chunk_size]
        for i in range(0, len(websocket_symbols), chunk_size)
    ]
    num_chunks = len(symbol_chunks)
    logger.info(
        f"将 {len(websocket_symbols)} 个交易对分成 {num_chunks} 个块进行监听 (每块最多 {chunk_size} 对)。"
    )
    websocket_connection_status = [False] * num_chunks  # 初始化状态列表
    ticker_watch_tasks.clear()  # 清空旧任务列表
    for i, chunk in enumerate(symbol_chunks):
        task = asyncio.create_task(
            watch_ticker_chunk_task(exchange, chunk, i, websocket_connection_status),
            name=f"WSChunk-{i+1}",  # 任务名从 1 开始
        )
        ticker_watch_tasks.append(task)

    # 2. 余额更新任务
    balance_update_interval = config["balance_update_interval_seconds"]
    balance_update_task = asyncio.create_task(
        update_balance_task(exchange, balance_update_interval), name="BalanceUpdater"
    )
    balance_update_task.last_update_time = 0  # 初始化上次更新时间

    # 3. 主套利计算循环任务
    arbitrage_loop_task = asyncio.create_task(
        main_arbitrage_loop(application, exchange, markets), name="ArbitrageLoop"
    )

    # --- 启动 Telegram Bot Polling ---
    logger.info("启动 Telegram Bot Polling...")
    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Telegram Bot 已成功启动。后台任务正在运行。按 Ctrl+C 退出。")

        # 等待任一关键任务结束 (正常应一直运行)
        all_tasks_to_wait = ticker_watch_tasks + [
            balance_update_task,
            arbitrage_loop_task,
        ]
        done, pending = await asyncio.wait(
            all_tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
        )

        # 如果有任务结束，记录日志并准备关闭
        for task in done:
            try:
                await task  # 获取可能的异常
                logger.warning(f"后台任务 {task.get_name()} 意外结束。")
            except asyncio.CancelledError:
                logger.info(f"后台任务 {task.get_name()} 被正常取消。")
            except Exception as e:
                logger.error(f"后台任务 {task.get_name()} 异常退出: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"启动或运行 Bot 时发生严重错误: {e}", exc_info=True)
    finally:
        # --- 优雅关闭 ---
        logger.info("--- 开始关闭程序 ---")
        config["running"] = False  # 停止计算循环

        # 1. 停止 Bot Polling
        logger.info("正在停止 Telegram Bot...")
        if application.updater and application.updater.running:
            await application.updater.stop()
        if application.running:
            await application.stop()
        # 等待 application 完全关闭
        await application.shutdown()
        logger.info("Telegram Bot 已停止。")

        # 2. 取消所有后台任务
        logger.info(
            f"正在取消 {len(ticker_watch_tasks)} 个 WS 任务, 余额任务, 套利循环任务..."
        )
        tasks_to_cancel = ticker_watch_tasks + [
            balance_update_task,
            arbitrage_loop_task,
        ]
        cancelled_count = 0
        for task in tasks_to_cancel:
            if task and not task.done():
                if task.cancel():
                    cancelled_count += 1
                else:
                    logger.warning(
                        f"任务 {task.get_name()} 可能无法取消 (已完成或状态不允许)。"
                    )
        logger.info(f"发出了 {cancelled_count} 个任务取消信号。等待任务结束...")
        # 等待取消完成 (设置超时)
        await asyncio.gather(
            *[t for t in tasks_to_cancel if t and t.cancelled()], return_exceptions=True
        )
        logger.info("后台任务已取消或结束。")

        # 3. 关闭 Exchange 连接
        logger.info("正在关闭 ccxt.pro 连接...")
        if exchange and hasattr(exchange, "close"):
            try:
                await exchange.close()
                logger.info("ccxt.pro 连接已关闭。")
            except Exception as e:
                logger.error(f"关闭 ccxt.pro 连接时出错: {e}", exc_info=True)

        logger.info("--- 程序关闭完成 ---")


# --- 程序入口 ---
if __name__ == "__main__":
    # 添加文件日志处理器 (可选)
    # log_filename = f"arbitrage_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    # file_handler.setFormatter(log_formatter)
    # logger.addHandler(file_handler)
    # logger.info(f"日志将记录到文件: {log_filename}")

    try:
        # Windows 下的事件循环策略 (如果需要)
        if sys.platform == "win32":
            logger.info(
                "检测到 Windows 系统，设置事件循环策略为 WindowsSelectorEventLoopPolicy"
            )
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # 运行主异步函数
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C 信号，正在关闭...")
    except Exception as e:
        logger.critical(f"程序顶层发生致命错误，无法启动: {e}", exc_info=True)
