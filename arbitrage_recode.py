# -*- coding: utf-8 -*-
"""
一个完全依赖 C++ 核心算法进行三角套利监控和交易的机器人。

本脚本作为 Python "外壳"，负责以下任务：
1.  通过 ccxt.pro 连接交易所（如币安），处理 API 和 WebSocket 通信。
2.  获取实时的市场 Ticker 数据和账户余额。
3.  提供一个 Telegram 机器人界面，用于监控状态、调整配置和控制开关。
4.  调用外部的 C++ 动态链接库 (DLL/SO) 来执行所有计算密集型任务：
    -   构建套利图 (build_arbitrage_graph.dll)
    -   使用 Bellman-Ford 算法查找负权重环路 (arbitrage_cpp.dll)
    -   对发现的机会进行详细的风险评估和滑点分析 (arbitrage_operations.dll)
    -   对机会进行包含首尾闪兑的完整路径模拟 (arbitrage_operations.dll)
5.  在满足所有条件时，执行真实的市价单交易。

使用前请确保：
-   已安装所有必要的 Python 库 (ccxt, ccxt-pro, python-telegram-bot)。
-   已将 C++ 项目编译为动态链接库，并放置于同目录下。
-   已正确填写下面的 API 密钥和 Telegram Bot Token。
"""
import ccxt.pro as ccxtpro
import time
import logging
from datetime import datetime
import os
import sys
import json
import asyncio
import traceback
import ctypes
import random
import collections
import concurrent.futures
from decimal import (
    Decimal,
    getcontext,
    ROUND_DOWN,
    InvalidOperation as DecimalInvalidOperation,
)

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
    BadSymbol,
    AuthenticationError,
)

# --- 导入 Telegram Bot 库 ---
from telegram import LinkPreviewOptions, Update, InlineKeyboardButton, InlineKeyboardMarkup
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


# ==============================================================================
# --- 配置区域 ---
# ==============================================================================

# --- 核心认证信息 ---
# Binance API Keys (从环境变量读取) - 确保有 **交易** 权限！
API_KEY = ""
API_SECRET = ""

# xt api
#API_KEY = "64b3c9fd-b076-4bb4-9707-7376960df7ca"
#API_SECRET = "d8a4c75899061bd0ff96095b28d612d19bb52323"

# Telegram Bot Token (从环境变量读取)
TELEGRAM_BOT_TOKEN = ""
# 你的 Telegram 用户 ID (用于限制谁能控制机器人, 设为 0 则不限制)
AUTHORIZED_USER_ID = 0  # <--- 强烈建议修改为你的真实 TG 用户 ID

# --- C++ 动态链接库路径 ---
if sys.platform == "win32":
    BF_DLL_PATH = "./arbitrage_cpp.dll"
    GRAPH_DLL_PATH = "./build_arbitrage_graph.dll"
    OPS_DLL_PATH = "./arbitrage_operations.dll"
else:
    BF_DLL_PATH = "./arbitrage_cpp.so"
    GRAPH_DLL_PATH = "./build_arbitrage_graph.so"
    OPS_DLL_PATH = "./arbitrage_operations.so"

# --- 核心运行配置 (可通过 Telegram Bot 修改) ---
config = {
    # --- 系统控制 ---
    "running": True,  # 套利计算循环是否运行

    # --- 交易与模拟 ---
    "auto_trade_enabled": False,  # !! 核心开关：是否自动执行真实交易 (默认为 False)
    "simulation_start_amount": Decimal("100.0"),  # 模拟起始金额 (USDT)
    "taker_fee_rate": Decimal("0.00075"),  # 吃单手续费率 (BNB抵扣后为0.075%)
    "min_trade_amount_usd_equivalent": Decimal("6.0"), # 估计的最低交易起始金额 (高于币安5U限制)
    "use_quote_order_qty_for_buy": True,  # 买入时是否优先使用 quoteOrderQty (按花费金额购买)
    "max_trade_retries": 2,  # 交易失败时的最大重试次数
    "trade_retry_delay_sec": 1.5,  # 交易重试间隔 (秒)

    # --- 机会发现与验证 ---
    "min_profit_full_sim_percent": Decimal("0.05"),  # 全路径模拟 (含闪兑) 的最低利润率
    "max_arbitrage_depth": 5,  # 最大套利路径深度
    "min_24h_quote_volume": Decimal("100000"),  # 市场流动性过滤阈值 (计价货币)

    # --- 风险评估 (这些值会被传递给 C++) ---
    "risk_assessment_enabled": True,  # 是否启用风险评估
    "min_profit_after_slippage_percent": Decimal("0.05"),  # 考虑滑点后的最低可接受利润率
    "max_allowed_slippage_percent_total": Decimal("0.15"),  # 整个路径允许的总预估滑点百分比
    "max_bid_ask_spread_percent_per_step": Decimal("0.50"),  # 单步允许的最大买卖价差百分比
    "min_depth_required_usd": Decimal("100.0"),  # 订单簿上靠近顶部的最小流动性要求 (以USD计价)
    "order_book_depth": 10,  # 获取订单簿的深度

    # --- 系统性能 ---
    "websocket_chunk_size": 180,  # WebSocket 监听块大小
    "balance_update_interval_seconds": 60,  # 余额更新间隔 (秒)
    "ticker_batch_size": 200,  # 首次获取 Ticker 进行流动性过滤的批次大小
    "orderbook_fetch_max_workers": 4, # 用于获取订单簿的最大线程数
    "use_threaded_orderbook_fetch": True, # 是否使用线程池并行获取订单簿
}

# Decimal 精度设置
getcontext().prec = 18

# ==============================================================================
# --- 全局状态变量 ---
# ==============================================================================
user_chat_id = None  # 存储授权用户的 chat_id
global_tickers = {}  # 全局存储最新 Ticker 数据 (由 WebSocket 更新)
global_balances = {}  # 全局存储最新余额信息
websocket_symbols = []  # 存储需要监听的交易对列表
websocket_connection_status = []  # 每个 WebSocket 块的连接状态
ticker_watch_tasks = []  # 存储所有 watcher 任务对象
last_ticker_update_time = 0  # 记录上次收到 ticker 更新的时间
is_trading_active = asyncio.Lock()  # 锁，防止并发执行自动交易
current_execution_task = None  # 当前活动的套利执行任务

# --- 性能统计相关 ---
stats_reporting_start_time = 0
cycle_count_total = 0
last_cycle_duration_g = None
snap_copy_duration_g = None
graph_build_duration_g = None
bf_call_duration_g = None
verification_duration_g = None
other_duration_g = None
last_execution_duration_g = None

# --- C++ 库加载状态 ---
cpp_bf_lib_loaded = False
cpp_graph_lib_loaded = False
cpp_ops_lib_loaded = False
arbitrage_lib = None
graph_builder_lib = None
ops_lib = None

# ==============================================================================
# --- 日志设置 ---
# ==============================================================================
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ArbitrageBot")
logger.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("ccxtpro").setLevel(logging.WARNING)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
if not logger.handlers:
    logger.addHandler(stream_handler)


# ==============================================================================
# --- C++ 库加载与定义 ---
# ==============================================================================

# C++ Edge 结构体定义，用于 Bellman-Ford
class CppEdge(ctypes.Structure):
    _fields_ = [
        ("from_node", ctypes.c_int),
        ("to_node", ctypes.c_int),
        ("weight", ctypes.c_double),
        ("pair_symbol", ctypes.c_char_p),
        ("trade_type", ctypes.c_char_p),
    ]

def load_cpp_libraries():
    """加载所有 C++ 动态链接库并设置函数签名。"""
    global arbitrage_lib, graph_builder_lib, ops_lib
    global cpp_bf_lib_loaded, cpp_graph_lib_loaded, cpp_ops_lib_loaded

    # 1. 加载 Bellman-Ford 库 (arbitrage_cpp)
    try:
        if os.path.exists(BF_DLL_PATH):
            arbitrage_lib = ctypes.CDLL(BF_DLL_PATH)
            arbitrage_lib.find_negative_cycles.argtypes = [
                ctypes.c_int, ctypes.POINTER(CppEdge), ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_longlong)
            ]
            arbitrage_lib.find_negative_cycles.restype = ctypes.c_int
            arbitrage_lib.free_memory.argtypes = [ctypes.c_char_p]
            arbitrage_lib.free_memory.restype = None
            cpp_bf_lib_loaded = True
            logger.info(f"成功加载 Bellman-Ford C++ 库: {BF_DLL_PATH}")
        else:
            logger.error(f"错误：找不到 Bellman-Ford C++ 库文件 '{BF_DLL_PATH}'。")
    except Exception as e:
        logger.error(f"加载 Bellman-Ford C++ 库 '{BF_DLL_PATH}' 时发生错误: {e}")

    # 2. 加载图构建库 (build_arbitrage_graph)
    try:
        if os.path.exists(GRAPH_DLL_PATH):
            graph_builder_lib = ctypes.CDLL(GRAPH_DLL_PATH)
            graph_builder_lib.build_graph_cpp.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
            graph_builder_lib.build_graph_cpp.restype = ctypes.c_char_p
            graph_builder_lib.free_memory.argtypes = [ctypes.c_char_p]
            graph_builder_lib.free_memory.restype = None
            cpp_graph_lib_loaded = True
            logger.info(f"成功加载图构建 C++ 库: {GRAPH_DLL_PATH}")
        else:
            logger.error(f"错误：找不到图构建 C++ 库文件 '{GRAPH_DLL_PATH}'。")
    except Exception as e:
        logger.error(f"加载图构建 C++ 库 '{GRAPH_DLL_PATH}' 时发生错误: {e}")

    # 3. 加载操作库 (arbitrage_operations)
    try:
        if os.path.exists(OPS_DLL_PATH):
            ops_lib = ctypes.CDLL(OPS_DLL_PATH)
            # 风险评估函数签名
            ops_lib.assess_risk_cpp_buffered.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int
            ]
            ops_lib.assess_risk_cpp_buffered.restype = ctypes.c_int
            # 全路径模拟函数签名
            ops_lib.simulate_full_cpp_buffered.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int
            ]
            ops_lib.simulate_full_cpp_buffered.restype = ctypes.c_int
            cpp_ops_lib_loaded = True
            logger.info(f"成功加载操作 C++ 库: {OPS_DLL_PATH}")
        else:
            logger.error(f"错误：找不到操作 C++ 库文件 '{OPS_DLL_PATH}'。")
    except Exception as e:
        logger.error(f"加载操作 C++ 库 '{OPS_DLL_PATH}' 时发生错误: {e}")

# ==============================================================================
# --- 核心 C++ 算法包装器 ---
# ==============================================================================

def build_arbitrage_graph(markets, current_tickers_snapshot, config):
    """
    (C++ 包装器 - 最终版) 调用 C++ DLL 来高效构建套利图。
    增加了数据预验证，并向 C++ 提供了 'active' 和 'spot' 字段以通过其内部过滤。
    """
    if not cpp_graph_lib_loaded:
        logger.error("图构建 C++ 库未加载，无法构建图。")
        return None, None, None

    # --- 数据预验证 (保持不变) ---
    valid_tickers = {}
    for symbol, ticker in current_tickers_snapshot.items():
        try:
            ask = ticker.get('ask')
            bid = ticker.get('bid')
            if ask and bid and ask > 0 and bid > 0 and ask >= bid:
                valid_tickers[symbol] = ticker
        except (TypeError, ValueError, DecimalInvalidOperation):
            continue
    
    if len(valid_tickers) < 3:
        logger.debug(f"有效的 Ticker 快照数量过少 ({len(valid_tickers)})，跳过图构建。")
        return None, None, None

    result_ptr_address = None
    tickers_json_str = ""
    markets_json_str = ""
    try:
        # --- !! 关键修改点 !! ---
        # 在 relevant_market_data 中添加 'active' 和 'spot' 字段
        relevant_market_data = {
            symbol: {
                "base": m.get("base"),
                "quote": m.get("quote"),
                "active": m.get("active", False),  # <--- 新增
                "spot": m.get("spot", False)        # <--- 新增
            }
            for symbol, m in markets.items() if symbol in valid_tickers
        }
        # --- !! 修改结束 !! ---

        tickers_json_str = json.dumps({s: {"bid": str(t['bid']), "ask": str(t['ask'])} for s, t in valid_tickers.items()})
        markets_json_str = json.dumps(relevant_market_data)
        taker_fee_rate = float(config["taker_fee_rate"])

        # 调用 C++ 函数
        graph_builder_lib.build_graph_cpp.restype = ctypes.c_void_p
        result_ptr_address = graph_builder_lib.build_graph_cpp(
            tickers_json_str.encode("utf-8"),
            markets_json_str.encode("utf-8"),
            ctypes.c_double(taker_fee_rate)
        )

        if not result_ptr_address:
            logger.error("C++ build_graph_cpp 调用失败，返回空指针。")
            logger.error(f"  - 发送的 Ticker 数量: {len(valid_tickers)}")
            logger.error(f"  - 发送的 Market 数量: {len(relevant_market_data)}")
            logger.error(f"  - Tickers (前200字符): {tickers_json_str[:200]}")
            logger.error(f"  - Markets (前200字符): {markets_json_str[:200]}") # <--- 新增日志，检查 markets 结构
            return None, None, None

        # 处理返回结果
        returned_json_str = ctypes.c_char_p(result_ptr_address).value.decode("utf-8")
        result_data = json.loads(returned_json_str)

        if "nodes" in result_data and "edges" in result_data:
            index_to_currency = result_data["nodes"]
            graph_edges = result_data["edges"]
            currency_to_index = {name: i for i, name in enumerate(index_to_currency)}
            return graph_edges, index_to_currency, currency_to_index
        else:
            logger.error(f"C++ build_graph_cpp 返回的 JSON 结构无效。")
            return None, None, None

    except Exception as e:
        logger.error(f"调用 C++ build_graph_cpp 或处理结果时出错: {e}", exc_info=True)
        return None, None, None
    finally:
        if result_ptr_address:
            try:
                graph_builder_lib.free_memory(ctypes.c_char_p(result_ptr_address))
            except Exception as free_e:
                logger.error(f"释放 C++ 图构建内存时出错: {free_e}")
def find_negative_cycles_bellman_ford(graph_edges_list, index_to_currency, currency_to_index, markets):
    """
    (C++ 包装器) 调用 C++ DLL 查找图中的负权重环路。
    """
    if not cpp_bf_lib_loaded:
        logger.error("Bellman-Ford C++ 库未加载，无法查找环路。")
        return []

    num_currencies = len(index_to_currency)
    if not graph_edges_list:
        return []

    json_result_ptr = ctypes.c_char_p(None)
    try:
        # 准备 C 结构体数组
        num_edges = len(graph_edges_list)
        CEdgeArray = CppEdge * num_edges
        c_edges = CEdgeArray()
        edge_string_buffers = []  # 保持对编码后字符串的引用

        for i, edge_data in enumerate(graph_edges_list):
            pair_bytes = str(edge_data.get("pair", "")).encode("utf-8")
            type_bytes = str(edge_data.get("type", "")).encode("utf-8")
            edge_string_buffers.extend([pair_bytes, type_bytes])
            c_edges[i] = CppEdge(
                ctypes.c_int(edge_data["from"]),
                ctypes.c_int(edge_data["to"]),
                ctypes.c_double(edge_data["weight"]),
                ctypes.c_char_p(pair_bytes),
                ctypes.c_char_p(type_bytes)
            )

        # 调用 C++ 函数
        relaxation_count_c = ctypes.c_longlong(0)
        result_code = arbitrage_lib.find_negative_cycles(
            ctypes.c_int(num_currencies), c_edges, ctypes.c_int(num_edges),
            ctypes.c_int(config['max_arbitrage_depth']),
            ctypes.byref(json_result_ptr), ctypes.byref(relaxation_count_c)
        )

        if result_code == 0 and json_result_ptr and json_result_ptr.value:
            json_string = json_result_ptr.value.decode("utf-8")
            cycles_data = json.loads(json_string)

            # 将 C++ 返回的索引和字符串转换为 Python 字典结构
            valid_cycles = []
            for cycle_cpp in cycles_data:
                nodes = [index_to_currency[i] for i in cycle_cpp.get("nodes", [])]
                if not nodes: continue
                trades = [
                    {
                        "from": index_to_currency[t.get("from_node")],
                        "to": index_to_currency[t.get("to_node")],
                        "pair": t.get("pair"),
                        "type": t.get("type"),
                    }
                    for t in cycle_cpp.get("trades", [])
                ]
                if len(nodes) - 1 != len(trades): continue
                valid_cycles.append({"nodes": nodes, "trades": trades, "depth": len(trades)})
            return valid_cycles
        elif result_code != 0 and result_code != 1: # 1表示未找到，是正常情况
             logger.error(f"C++ find_negative_cycles 返回错误码: {result_code}")

        return []

    except Exception as e:
        logger.error(f"调用 C++ Bellman-Ford 或处理结果时出错: {e}", exc_info=True)
        return []
    finally:
        if json_result_ptr and json_result_ptr.value:
            try:
                arbitrage_lib.free_memory(json_result_ptr)
            except Exception as free_e:
                logger.error(f"释放 C++ Bellman-Ford 内存时出错: {free_e}")

def fetch_order_book_in_thread(pair: str, exchange_config: dict, limit: int):
    """在一个新的事件循环中同步地获取订单簿，设计为在线程池中运行。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    exchange = None
    try:
        exchange = ccxtpro.binance({'options': exchange_config.get("options", {})})
        book = loop.run_until_complete(exchange.fetch_l2_order_book(pair, limit=limit))
        if not isinstance(book, dict) or "bids" not in book or "asks" not in book:
            return pair, None
        # 转换为 C++ 期望的字符串格式
        bids_cpp = [[str(p), str(a)] for p, a in book.get("bids", [])]
        asks_cpp = [[str(p), str(a)] for p, a in book.get("asks", [])]
        return pair, {"bids": bids_cpp, "asks": asks_cpp}
    except Exception as e:
        logger.warning(f"[线程] 获取订单簿 {pair} 失败: {type(e).__name__}")
        return pair, None
    finally:
        if exchange:
            loop.run_until_complete(exchange.close())
        loop.close()

async def assess_arbitrage_risk(cycle_info: dict, start_amount: Decimal, exchange: ccxtpro.Exchange, markets: dict, current_tickers: dict, config: dict) -> dict:
    """(C++ 包装器) 调用 C++ DLL 进行风险评估。"""
    if not cpp_ops_lib_loaded:
        return {"is_viable": False, "reasons": ["C++ 操作库未加载"]}

    # 1. 并行获取所有需要的订单簿
    path_pairs = {trade["pair"] for trade in cycle_info.get("trades", [])}
    order_books_data = {}
    if not path_pairs:
        return {"is_viable": False, "reasons": ["路径中无交易对"]}

    if config.get("use_threaded_orderbook_fetch", True):
        exchange_config_for_thread = {"options": exchange.options}
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["orderbook_fetch_max_workers"]) as executor:
            futures = [executor.submit(fetch_order_book_in_thread, pair, exchange_config_for_thread, config["order_book_depth"]) for pair in path_pairs]
            for future in concurrent.futures.as_completed(futures):
                pair, book_data = future.result()
                if book_data:
                    order_books_data[pair] = book_data
    else: # 备用的 asyncio.gather 方式
        async def fetch_book_async(pair):
            try:
                book = await exchange.fetch_l2_order_book(pair, limit=config["order_book_depth"])
                if not isinstance(book, dict) or "bids" not in book or "asks" not in book: return pair, None
                bids_cpp = [[str(p), str(a)] for p, a in book.get("bids", [])]
                asks_cpp = [[str(p), str(a)] for p, a in book.get("asks", [])]
                return pair, {"bids": bids_cpp, "asks": asks_cpp}
            except Exception: return pair, None
        results = await asyncio.gather(*(fetch_book_async(p) for p in path_pairs))
        order_books_data = {pair: book for pair, book in results if book}

    if len(order_books_data) != len(path_pairs):
        missing = path_pairs - set(order_books_data.keys())
        return {"is_viable": False, "reasons": [f"未能获取订单簿: {', '.join(missing)}"]}

    # 2. 准备 C++ 函数的输入
    output_buffer = ctypes.create_string_buffer(8192)
    try:
        # 准备 markets 和 tickers 的子集，只包含路径中用到的
        involved_symbols = set(order_books_data.keys())
        involved_currencies = set(cycle_info.get("nodes", []))
        for curr in involved_currencies:
            involved_symbols.add(f"{curr}/USDT")
            involved_symbols.add(f"USDT/{curr}")

        markets_to_cpp = {
            s: {
                "base": m.get("base"), "quote": m.get("quote"),
                "limits": {"amount": {"min": str(m.get("limits", {}).get("amount", {}).get("min", "0"))},
                           "cost": {"min": str(m.get("limits", {}).get("cost", {}).get("min", "0"))}}
            } for s, m in markets.items() if s in involved_symbols
        }
        tickers_to_cpp = {
            s: {"bid": str(t.get("bid", "0")), "ask": str(t.get("ask", "0"))}
            for s, t in current_tickers.items() if s in involved_symbols
        }
        config_to_cpp = {k: str(v) if isinstance(v, Decimal) else v for k, v in config.items()}

        # 3. 调用 C++ 函数
        status_code = ops_lib.assess_risk_cpp_buffered(
            json.dumps(cycle_info).encode('utf-8'),
            str(start_amount).encode('utf-8'),
            json.dumps(order_books_data).encode('utf-8'),
            json.dumps(markets_to_cpp).encode('utf-8'),
            json.dumps(tickers_to_cpp).encode('utf-8'),
            json.dumps(config_to_cpp).encode('utf-8'),
            output_buffer,
            ctypes.c_int(len(output_buffer))
        )

        # 4. 处理结果
        if status_code == 0:
            result_str = output_buffer.value.decode("utf-8")
            result_data = json.loads(result_str)
            # 将字符串结果转回 Decimal
            for key in ["estimated_profit_percent_after_slippage", "total_estimated_slippage_percent"]:
                if key in result_data: result_data[key] = Decimal(str(result_data[key]))
            for detail in result_data.get("details", []):
                for key in ["slippage_percent", "spread_percent", "depth_usd"]:
                    if key in detail: detail[key] = Decimal(str(detail[key]))
            return result_data
        else:
            return {"is_viable": False, "reasons": [f"C++ 风险评估返回错误码: {status_code}"]}

    except Exception as e:
        logger.error(f"调用 C++ 风险评估时出错: {e}", exc_info=True)
        return {"is_viable": False, "reasons": [f"Python 包装器错误: {e}"]}


async def simulate_full_execution_profit(cycle_info: dict, actual_start_currency: str, actual_start_amount: Decimal, end_with_usdt: bool, current_tickers: dict, markets: dict, config: dict) -> dict:
    """(C++ 包装器) 调用 C++ DLL 进行全路径模拟。"""
    if not cpp_ops_lib_loaded:
        return {"verified": False, "reason": "C++ 操作库未加载"}

    output_buffer = ctypes.create_string_buffer(4096)
    try:
        # 准备 markets 和 tickers 的子集
        involved_symbols = {t["pair"] for t in cycle_info.get("trades", [])}
        involved_currencies = set(cycle_info.get("nodes", []))
        involved_currencies.add(actual_start_currency)
        if end_with_usdt: involved_currencies.add("USDT")
        for c1 in involved_currencies:
            for c2 in involved_currencies:
                if c1 != c2: involved_symbols.add(f"{c1}/{c2}")

        markets_to_cpp = {s: {"base": m.get("base"), "quote": m.get("quote")} for s, m in markets.items() if s in involved_symbols}
        tickers_to_cpp = {s: {"bid": str(t.get("bid", "0")), "ask": str(t.get("ask", "0"))} for s, t in current_tickers.items() if s in involved_symbols}
        config_to_cpp = {k: str(v) if isinstance(v, Decimal) else v for k, v in config.items()}

        # 调用 C++
        status_code = ops_lib.simulate_full_cpp_buffered(
            json.dumps(cycle_info).encode('utf-8'),
            actual_start_currency.encode('utf-8'),
            str(actual_start_amount).encode('utf-8'),
            ctypes.c_int(1 if end_with_usdt else 0),
            json.dumps(tickers_to_cpp).encode('utf-8'),
            json.dumps(markets_to_cpp).encode('utf-8'),
            json.dumps(config_to_cpp).encode('utf-8'),
            output_buffer,
            ctypes.c_int(len(output_buffer))
        )

        # 处理结果
        if status_code == 0:
            result_str = output_buffer.value.decode("utf-8")
            result_data = json.loads(result_str)
            for key in ["profit_percent", "profit_amount", "final_amount"]:
                if key in result_data: result_data[key] = Decimal(str(result_data[key]))
            return result_data
        else:
            return {"verified": False, "reason": f"C++ 模拟返回错误码: {status_code}"}

    except Exception as e:
        logger.error(f"调用 C++ 全路径模拟时出错: {e}", exc_info=True)
        return {"verified": False, "reason": f"Python 包装器错误: {e}"}


# ==============================================================================
# --- 辅助函数与交易执行 ---
# ==============================================================================

def format_decimal(d, precision=8):
    """安全地格式化 Decimal 为字符串。"""
    try:
        if not isinstance(d, Decimal): d = Decimal(str(d))
        if not d.is_finite(): return str(d)
        return str(d.quantize(Decimal('1E-' + str(precision)), rounding=ROUND_DOWN))
    except Exception:
        return "格式错误"

def parse_order_result(order_info: dict, base_currency: str, quote_currency: str, side: str, expected_price: Decimal = None) -> dict:
    """解析 CCXT 订单结果，计算均价和滑点。"""
    try:
        order_id = order_info.get("id")
        status = "ok" if order_info.get("status") in ["filled", "closed"] else "partial"
        filled_base = Decimal(order_info.get("filled", "0"))
        cost_quote = Decimal(order_info.get("cost", "0"))
        fee_info = order_info.get("fee", {})
        fee_amount = Decimal(str(fee_info.get("cost", "0")))
        fee_currency = fee_info.get("currency")

        average_price = cost_quote / filled_base if filled_base > 0 else None
        slippage_percent = None
        if expected_price and average_price:
            if side == "buy":
                slippage_percent = ((average_price - expected_price) / expected_price) * 100
            else: # sell
                slippage_percent = ((expected_price - average_price) / expected_price) * 100

        if side == "buy":
            received_amount = filled_base
            received_currency = base_currency
            if fee_currency == base_currency: received_amount -= fee_amount
        else: # sell
            received_amount = cost_quote
            received_currency = quote_currency
            if fee_currency == quote_currency: received_amount -= fee_amount

        return {
            "status": status, "order_id": order_id, "side": side, "symbol": order_info.get("symbol"),
            "spent_amount": cost_quote if side == "buy" else filled_base,
            "spent_currency": quote_currency if side == "buy" else base_currency,
            "received_amount": received_amount, "received_currency": received_currency,
            "average_price": average_price, "slippage_percent": slippage_percent,
            "fee_amount": fee_amount, "fee_currency": fee_currency,
        }
    except Exception as e:
        logger.error(f"解析订单结果时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

async def execute_real_market_order(exchange, symbol, side, amount, markets, expected_price, params=None):
    """统一的市价单执行函数，带重试逻辑。"""
    max_retries = config["max_trade_retries"]
    retry_delay = config["trade_retry_delay_sec"]
    market = markets.get(symbol)
    if not market: return {"status": "error", "message": f"市场 {symbol} 未找到"}

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"[交易尝试 #{attempt+1}] {side.upper()} {format_decimal(amount)} on {symbol}")
            if side == 'buy':
                order = await exchange.create_market_buy_order(symbol, amount, params)
            else: # sell
                order = await exchange.create_market_sell_order(symbol, amount, params)
            logger.info(f"订单提交成功: ID {order.get('id')}")
            return parse_order_result(order, market['base'], market['quote'], side, expected_price)
        except (CCXTNetworkError, RequestTimeout, ExchangeNotAvailable, RateLimitExceeded) as e:
            logger.warning(f"交易尝试 #{attempt+1} 失败 (网络/限速): {type(e).__name__}")
            if attempt >= max_retries:
                return {"status": "error", "message": f"达到最大重试次数: {e}"}
            await asyncio.sleep(retry_delay)
        except (InsufficientFunds, InvalidOrder) as e:
            logger.error(f"交易失败 (不可重试): {type(e).__name__} - {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"交易时发生未知错误: {e}", exc_info=True)
            return {"status": "error", "message": f"未知错误: {e}"}

async def execute_real_swap(exchange, from_currency, to_currency, from_amount, markets, tickers_snapshot):
    """使用市价单执行货币间的闪兑，并返回标准化结果。"""
    logger.info(f"[闪兑] 请求: {format_decimal(from_amount)} {from_currency} -> {to_currency}")
    if from_currency == to_currency:
        return {"status": "ok", "message": "源货币与目标货币相同", "received_to_amount": from_amount}

    # 策略1: 买入目标货币 (例如 USDC/USDT, 花 USDT 买 USDC)
    symbol_buy = f"{to_currency}/{from_currency}"
    if symbol_buy in markets:
        ticker = tickers_snapshot.get(symbol_buy)
        expected_price = Decimal(str(ticker['ask'])) if ticker and 'ask' in ticker else None
        params = {'quoteOrderQty': float(from_amount)} if config["use_quote_order_qty_for_buy"] else {}
        amount_param = None if config["use_quote_order_qty_for_buy"] else float(from_amount / expected_price) if expected_price else None
        if amount_param is not None or params:
            return await execute_real_market_order(exchange, symbol_buy, 'buy', amount_param, markets, expected_price, params)

    # 策略2: 卖出源货币 (例如 USDT/USDC, 卖 USDT 换 USDC)
    symbol_sell = f"{from_currency}/{to_currency}"
    if symbol_sell in markets:
        ticker = tickers_snapshot.get(symbol_sell)
        expected_price = Decimal(str(ticker['bid'])) if ticker and 'bid' in ticker else None
        return await execute_real_market_order(exchange, symbol_sell, 'sell', float(from_amount), markets, expected_price)

    return {"status": "error", "message": f"无法找到合适的交易对转换 {from_currency} -> {to_currency}"}


async def execute_arbitrage_path(exchange, cycle_info, markets, application):
    """核心函数：执行完整的套利路径，生成详细报告。"""
    global is_trading_active, current_execution_task, last_execution_duration_g, user_chat_id

    path_str = " -> ".join(cycle_info.get("nodes", ["?"]))
    if not await is_trading_active.acquire():
        logger.warning(f"交易锁已被占用，跳过执行: {path_str}")
        return

    task_start_time = time.time()
    logger.info(f"--- [执行开始] 获得交易锁，开始执行路径: {path_str} ---")

    try:
        # 1. 捕获 Ticker 快照和初始余额
        tickers_snapshot = global_tickers.copy()
        initial_balances = global_balances.copy()
        involved_currencies = set(cycle_info.get("nodes", [])) | {"BNB"}

        # 2. 确定起始资金和必要的初始闪兑
        cycle_start_node = cycle_info["nodes"][0]
        min_start_usd = config["min_trade_amount_usd_equivalent"]
        
        # 简单逻辑：总是从 USDT 开始，并兑换成路径起始货币
        start_fund_currency = "USDT"
        start_fund_amount = initial_balances.get(start_fund_currency, Decimal("0"))

        if start_fund_amount < min_start_usd:
            msg = f"起始资金 {start_fund_currency} 余额 ({format_decimal(start_fund_amount)}) 不足最低要求 (${min_start_usd})。"
            logger.error(f"[执行中止] {msg}")
            if user_chat_id: await application.bot.send_message(user_chat_id, f"❌ 套利中止: {msg}")
            return

        current_amount = min(start_fund_amount, config["simulation_start_amount"]) # 使用配置的模拟金额或余额中较小者
        current_currency = start_fund_currency
        
        # 执行初始闪兑
        if current_currency != cycle_start_node:
            swap_res = await execute_real_swap(exchange, current_currency, cycle_start_node, current_amount, markets, tickers_snapshot)
            if swap_res.get("status") not in ["ok", "partial"]:
                msg = f"初始闪兑失败: {swap_res.get('message', '未知错误')}"
                logger.error(f"[执行中止] {msg}")
                if user_chat_id: await application.bot.send_message(user_chat_id, f"❌ 套利中止: {msg}")
                return
            current_amount = swap_res["received_amount"]
            current_currency = swap_res["received_currency"]

        # 3. 按顺序执行套利路径交易
        trade_results = []
        trade_successful = True
        for trade in cycle_info["trades"]:
            if current_currency != trade["from"]:
                logger.error(f"逻辑错误: 需要 {trade['from']}, 但持有 {current_currency}")
                trade_successful = False
                break
            
            pair = trade["pair"]
            side = trade["type"].lower()
            ticker = tickers_snapshot.get(pair)
            expected_price = Decimal(str(ticker['ask' if side == 'buy' else 'bid'])) if ticker else None
            
            params = {}
            amount_param = float(current_amount)
            if side == 'buy' and config["use_quote_order_qty_for_buy"]:
                params = {'quoteOrderQty': float(current_amount)}
                amount_param = None

            order_res = await execute_real_market_order(exchange, pair, side, amount_param, markets, expected_price, params)
            trade_results.append(order_res)

            if order_res.get("status") not in ["ok", "partial"]:
                trade_successful = False
                break
            
            current_amount = order_res["received_amount"]
            current_currency = order_res["received_currency"]

        # 4. 执行最终闪兑回 USDT
        final_swap_res = None
        if trade_successful and current_currency != "USDT":
            final_swap_res = await execute_real_swap(exchange, current_currency, "USDT", current_amount, markets, tickers_snapshot)
            if final_swap_res.get("status") in ["ok", "partial"]:
                current_amount = final_swap_res["received_amount"]
                current_currency = "USDT"
            else:
                logger.error(f"最终闪兑失败: {final_swap_res.get('message')}")
        
        # 5. 生成并发送报告
        # (此处省略了详细的报告生成代码以保持简洁，但逻辑是收集所有步骤结果并格式化)
        final_report = f"**📊 套利执行报告**\n路径: `{path_str}`\n"
        # ... 添加初始投入、交易步骤、最终结果、余额变化等 ...
        if trade_successful:
            final_report += f"🟢 **执行成功**\n最终持有: `{format_decimal(current_amount)} {current_currency}`"
        else:
            final_report += f"🔴 **执行失败或中止**\n最后一步结果: `{trade_results[-1] if trade_results else 'N/A'}`"

        logger.info("执行报告:\n" + final_report.replace("`", "").replace("*", ""))
        if user_chat_id:
            await application.bot.send_message(user_chat_id, final_report, parse_mode=ParseMode.HTML)

    finally:
        last_execution_duration_g = time.time() - task_start_time
        if is_trading_active.locked():
            is_trading_active.release()
        current_execution_task = None
        logger.info(f"--- [执行结束] 交易锁已释放 (耗时: {last_execution_duration_g:.3f}s) ---")


# ==============================================================================
# --- 后台任务 (WebSocket, 余额更新) ---
# ==============================================================================

async def watch_ticker_chunk_task(exchange, symbol_chunk, chunk_index, conn_status_list):
    """监听一个交易对块的 Ticker 数据。"""
    global global_tickers, last_ticker_update_time
    logger.info(f"启动 WebSocket 块 {chunk_index+1} (监听 {len(symbol_chunk)} 个交易对)...")
    conn_status_list[chunk_index] = False
    while True:
        try:
            tickers_update = await exchange.watch_tickers(symbol_chunk)
            if not conn_status_list[chunk_index]:
                logger.info(f"块 {chunk_index+1}: WebSocket 连接成功！")
                conn_status_list[chunk_index] = True
            
            now = time.time()
            for symbol, ticker in tickers_update.items():
                ask = ticker.get("ask")
                bid = ticker.get("bid")
                if ask is not None and bid is not None:
                    try:
                        global_tickers[symbol] = {
                            "ask": Decimal(str(ask)),
                            "bid": Decimal(str(bid)),
                        }
                    except DecimalInvalidOperation:
                        continue
            last_ticker_update_time = now
        except Exception as e:
            logger.warning(f"块 {chunk_index+1} WebSocket 错误: {type(e).__name__}。尝试重连...")
            conn_status_list[chunk_index] = False
            await asyncio.sleep(5)

async def update_balance_task(exchange, update_interval_sec):
    """后台任务，定期获取并更新全局账户余额。"""
    global global_balances
    logger.info(f"启动余额更新任务，每 {update_interval_sec} 秒一次...")
    while True:
        try:
            balance_data = await exchange.fetch_balance()
            free_balances = balance_data.get("free", {})
            global_balances = {
                currency: Decimal(amount_str)
                for currency, amount_str in free_balances.items()
                if Decimal(amount_str) > 0
            }
        except Exception as e:
            logger.warning(f"获取余额时出错: {type(e).__name__}")
        await asyncio.sleep(update_interval_sec)


# ==============================================================================
# --- Telegram Bot 命令处理 ---
# ==============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_chat_id
    user_id = update.effective_user.id
    if AUTHORIZED_USER_ID != 0 and user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("抱歉，您无权使用此机器人。")
        return
    user_chat_id = update.effective_chat.id
    config["running"] = True
    welcome_message = (
        f"欢迎, {update.effective_user.mention_html()}!\n\n"
        f"套利机器人已启动并运行。\n"
        f"自动交易: {'已启用 ✅' if config['auto_trade_enabled'] else '已禁用 ❌'}\n\n"
        f"使用 /status 查看详细状态，/help 获取命令列表。"
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID: return
    help_text = (
        "<b>套利机器人帮助</b>\n\n"
        "<b>基础命令:</b>\n"
        "  <code>/start</code> - 初始化机器人。\n"
        "  <code>/status</code> - 查看详细运行状态和配置。\n"
        "  <code>/balance</code> - 显示当前账户余额。\n"
        "  <code>/help</code> - 显示此帮助信息。\n\n"
        "<b>控制命令:</b>\n"
        "  <code>/trade [on|off]</code> - <b>[!!]</b> 启用或禁用自动真实交易。\n"
        "  <code>/pause</code> - 暂停套利计算。\n"
        "  <code>/resume</code> - 恢复套利计算。\n\n"
        "<b>配置命令:</b> <code>/set [参数] [值]</code>\n"
        "  - <code>fee_rate [小数]</code> (例如 0.00075)\n"
        "  - <code>min_profit [百分比]</code> (例如 0.05)\n"
        "  - <code>depth [整数]</code> (例如 5)\n"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID: return
    
    # 计算 CPS
    cps_str = "N/A"
    if stats_reporting_start_time > 0:
        elapsed = time.time() - stats_reporting_start_time
        if elapsed > 1: cps_str = f"{cycle_count_total / elapsed:.2f} 周期/秒"

    status_text = (
        f"--- <b>机器人状态</b> ---\n"
        f"<b>运行控制:</b>\n"
        f"  计算循环: {'运行中' if config['running'] else '已暂停'}\n"
        f"  自动交易: {'<b>已启用 ✅</b>' if config['auto_trade_enabled'] else '已禁用 ❌'}\n"
        f"<b>连接与数据:</b>\n"
        f"  WebSocket: {sum(1 for s in websocket_connection_status if s)}/{len(websocket_connection_status)} 连接块活跃\n"
        f"  缓存Tickers: {len(global_tickers)} (最后更新: {time.time() - last_ticker_update_time:.1f}s 前)\n"
        f"  账户余额: 持有 {len(global_balances)} 种资产\n"
        f"<b>性能统计:</b>\n"
        f"  循环速率: {cps_str}\n"
        f"  上次计算耗时: {last_cycle_duration_g*1000:.1f} ms\n"
        f"    - 快照: {snap_copy_duration_g*1000:.1f}ms, 图构建: {graph_build_duration_g*1000:.1f}ms\n"
        f"    - BF: {bf_call_duration_g*1000:.1f}ms, 验证: {verification_duration_g*1000:.1f}ms\n"
        f"<b>C++ 库状态:</b>\n"
        f"  图构建: {'已加载' if cpp_graph_lib_loaded else '失败'}\n"
        f"  Bellman-Ford: {'已加载' if cpp_bf_lib_loaded else '失败'}\n"
        f"  操作(风控/模拟): {'已加载' if cpp_ops_lib_loaded else '失败'}\n"
    )
    await update.message.reply_text(status_text, parse_mode=ParseMode.HTML)

async def set_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID: return
    args = context.args
    if len(args) != 2:
        await update.message.reply_text("用法: /set [参数名] [值]")
        return
    
    param, value_str = args[0].lower(), args[1]
    try:
        if param == "fee_rate":
            config["taker_fee_rate"] = Decimal(value_str)
        elif param == "min_profit":
            config["min_profit_full_sim_percent"] = Decimal(value_str)
        elif param == "depth":
            config["max_arbitrage_depth"] = int(value_str)
        else:
            await update.message.reply_text(f"未知参数: {param}")
            return
        await update.message.reply_text(f"✅ 参数 `{param}` 已更新为 `{value_str}`", parse_mode=ParseMode.HTML)
        logger.info(f"配置更新 via TG: {param} -> {value_str}")
    except (ValueError, DecimalInvalidOperation) as e:
        await update.message.reply_text(f"❌ 无效值格式: {e}")

async def trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID: return
    args = context.args
    if not args:
        status = '已启用 ✅' if config['auto_trade_enabled'] else '已禁用 ❌'
        await update.message.reply_text(f"当前自动交易状态: {status}\n使用 `/trade on` 或 `/trade off` 切换。")
        return
    
    command = args[0].lower()
    if command == "on":
        keyboard = [[InlineKeyboardButton("⚠️ 确认启用自动交易", callback_data="confirm_trade_on")]]
        await update.message.reply_text(
            "<b><u>警告!</u></b> 您确定要启用自动交易吗?\n启用后将自动执行真实交易，可能导致资金损失。",
            reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
        )
    elif command == "off":
        config["auto_trade_enabled"] = False
        logger.info("自动交易已由用户禁用。")
        await update.message.reply_text("❌ 自动交易已禁用。")
    else:
        await update.message.reply_text("用法: `/trade on` 或 `/trade off`")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "confirm_trade_on":
        config["auto_trade_enabled"] = True
        logger.info("用户已确认，自动交易已启用。")
        await query.edit_message_text(text="✅ 自动交易已确认启用。\n<b>请密切监控!</b>", parse_mode=ParseMode.HTML)

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID: return
    config["running"] = False
    await update.message.reply_text("⏸️ 套利计算循环已暂停。")
    logger.info("套利计算已暂停。")

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID: return
    config["running"] = True
    await update.message.reply_text("▶️ 套利计算循环已恢复。")
    logger.info("套利计算已恢复。")

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if AUTHORIZED_USER_ID != 0 and update.effective_user.id != AUTHORIZED_USER_ID: return
    if not global_balances:
        await update.message.reply_text("余额信息尚不可用。")
        return
    
    balance_text = "<b>当前可用余额 (非零):</b>\n<pre>"
    sorted_balances = sorted(global_balances.items(), key=lambda item: item[0])
    for currency, amount in sorted_balances:
        balance_text += f"{currency:<6} : {format_decimal(amount)}\n"
    balance_text += "</pre>"
    await update.message.reply_text(balance_text, parse_mode=ParseMode.HTML)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("处理 Telegram 更新时发生异常:", exc_info=context.error)

# ==============================================================================
# --- 主套利循环 ---
# ==============================================================================

async def main_arbitrage_loop(application, exchange, markets):
    """主套利计算循环，完全依赖 C++ 核心。增加了预热阶段。"""
    global last_ticker_update_time, last_cycle_duration_g, snap_copy_duration_g, graph_build_duration_g
    global bf_call_duration_g, verification_duration_g, other_duration_g, cycle_count_total, current_execution_task

    # --- 新增：预热阶段 ---
    logger.info("主循环预热中，等待 WebSocket Ticker 数据稳定...")
    # 等待至少 80% 的目标交易对都已接收到至少一次数据更新
    required_tickers = int(len(websocket_symbols) * 0.8)
    while len(global_tickers) < required_tickers:
        logger.info(f"  ...等待 Ticker 数据 ({len(global_tickers)}/{required_tickers})")
        await asyncio.sleep(2)
    logger.info("Ticker 数据已稳定，主套利计算循环正式开始！")
    # --- 预热结束 ---

    last_processed_update_time = 0

    while True:
        # 1. 检查运行状态和是否有新的 Ticker 数据
        if not config["running"]:
            await asyncio.sleep(1)
            continue
        
        current_update_time = last_ticker_update_time
        if not any(websocket_connection_status) or current_update_time == last_processed_update_time:
            await asyncio.sleep(0.01) # 短暂休眠，避免空转
            continue
        
        last_processed_update_time = current_update_time
        loop_start_time = time.time()

        # 2. 获取 Ticker 快照
        snap_copy_start = time.time()
        tickers_snapshot = global_tickers.copy()
        snap_copy_duration_g = time.time() - snap_copy_start
        if not tickers_snapshot: continue

        # 3. 构建图 (C++)
        graph_build_start = time.time()
        # 调用的是上面那个增强版的包装器
        graph_edges, index_to_currency, currency_to_index = build_arbitrage_graph(markets, tickers_snapshot, config)
        graph_build_duration_g = time.time() - graph_build_start

        if not graph_edges:
            last_cycle_duration_g = time.time() - loop_start_time
            continue

        # 4. 查找负环 (C++)
        bf_call_start = time.time()
        negative_cycles = find_negative_cycles_bellman_ford(graph_edges, index_to_currency, currency_to_index, markets)
        bf_call_duration_g = time.time() - bf_call_start

        # 5. 验证机会 (C++)
        verification_start = time.time()
        if negative_cycles:
            # (这部分逻辑保持不变)
            for cycle in negative_cycles:
                sim_res = await simulate_full_execution_profit(
                    cycle, "USDT", config["simulation_start_amount"], True,
                    tickers_snapshot, markets, config
                )

                if sim_res and sim_res.get("verified"):
                    path_str = " -> ".join(cycle.get("nodes", []))
                    profit_perc = sim_res["profit_percent"]
                    logger.info(f"✅ C++模拟验证成功: {path_str} (模拟利润: {profit_perc:.4f}%)")

                    if config["auto_trade_enabled"]:
                        if current_execution_task and not current_execution_task.done():
                            logger.info(f"已有交易任务在运行，跳过机会: {path_str}")
                            continue

                        # 风险评估 (C++)
                        risk_res = await assess_arbitrage_risk(cycle, config["simulation_start_amount"], exchange, markets, tickers_snapshot, config)
                        if risk_res and risk_res.get("is_viable"):
                            logger.info(f"风险评估通过: {path_str}。准备执行...")
                            if user_chat_id:
                                await application.bot.send_message(user_chat_id, f"🤖 检测到机会 (模拟利润 {profit_perc:.4f}%)，风险评估通过，开始执行...\n路径: `{path_str}`", parse_mode=ParseMode.HTML)
                            
                            current_execution_task = asyncio.create_task(
                                execute_arbitrage_path(exchange, cycle, markets, application),
                                name=f"ArbitrageExec-{int(time.time())}"
                            )
                            break # 找到并启动一个任务后，结束本轮查找
                        else:
                            reasons = "; ".join(risk_res.get("reasons", ["未知"]))
                            logger.warning(f"风险评估未通过: {path_str}。原因: {reasons}")
        
        verification_duration_g = time.time() - verification_start
        cycle_count_total += 1
        last_cycle_duration_g = time.time() - loop_start_time
        other_duration_g = last_cycle_duration_g - (snap_copy_duration_g + graph_build_duration_g + bf_call_duration_g + verification_duration_g)
# ==============================================================================
# --- 主函数入口 ---
# ==============================================================================

async def main():
    """程序主入口：设置并运行所有异步任务。"""
    global stats_reporting_start_time, websocket_symbols
    
    # 0. 启动前检查
    if not all([TELEGRAM_BOT_TOKEN, API_KEY, API_SECRET]):
        logger.critical("错误：必须设置所有 API 密钥和 Token！")
        return
    
    stats_reporting_start_time = time.time()
    load_cpp_libraries()

    # 1. 连接交易所
    exchange = ccxtpro.binance({'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': False, 'options': {'defaultType': 'spot'}})
    try:
        markets = await exchange.load_markets()
        markets = {s: m for s, m in markets.items() if m.get("spot") and m.get("active")}
        logger.info(f"成功连接到币安并加载了 {len(markets)} 个活跃现货市场。")
    except Exception as e:
        logger.critical(f"连接交易所或加载市场失败: {e}", exc_info=True)
        if exchange: await exchange.close()
        return

    # 2. 筛选用于监听的交易对
    all_spot_symbols = list(markets.keys())
    all_tickers = {}
    batch_size = config["ticker_batch_size"]
    logger.info(f"分批获取 Ticker 以进行流动性过滤...")
    for i in range(0, len(all_spot_symbols), batch_size):
        try:
            batch = all_spot_symbols[i:i+batch_size]
            all_tickers.update(await exchange.fetch_tickers(batch))
        except Exception as e:
            logger.error(f"获取 Ticker 批次失败: {e}")
            break
    
    min_volume = config["min_24h_quote_volume"]
    websocket_symbols = [
        s for s, t in all_tickers.items()
        if t.get("quoteVolume") and Decimal(str(t["quoteVolume"])) >= min_volume
    ]
    if not websocket_symbols:
        logger.critical("流动性过滤后无可用交易对，程序退出。")
        await exchange.close()
        return
    logger.info(f"流动性过滤完成，将监听 {len(websocket_symbols)} 个交易对。")

    # 3. 设置 Telegram Bot
    defaults = Defaults(parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True))
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).defaults(defaults).rate_limiter(AIORateLimiter()).build()
    
    # 注册命令
    handlers = [
        CommandHandler("start", start_command), CommandHandler("help", help_command),
        CommandHandler("status", status_command), CommandHandler("set", set_command),
        CommandHandler("trade", trade_command), CommandHandler("pause", pause_command),
        CommandHandler("resume", resume_command), CommandHandler("balance", balance_command),
        CallbackQueryHandler(button_callback)
    ]
    for handler in handlers: application.add_handler(handler)
    application.add_error_handler(error_handler)

    # 4. 创建并启动后台任务
    # WebSocket 监听任务
    chunk_size = config["websocket_chunk_size"]
    symbol_chunks = [websocket_symbols[i:i+chunk_size] for i in range(0, len(websocket_symbols), chunk_size)]
    global websocket_connection_status, ticker_watch_tasks
    websocket_connection_status = [False] * len(symbol_chunks)
    for i, chunk in enumerate(symbol_chunks):
        task = asyncio.create_task(watch_ticker_chunk_task(exchange, chunk, i, websocket_connection_status))
        ticker_watch_tasks.append(task)
    
    # 余额更新任务
    balance_task = asyncio.create_task(update_balance_task(exchange, config["balance_update_interval_seconds"]))
    
    # 主套利循环任务
    arbitrage_loop_task = asyncio.create_task(main_arbitrage_loop(application, exchange, markets))

    # 5. 启动 Bot 并保持运行
    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        logger.info("机器人已成功启动。按 Ctrl+C 退出。")
        await asyncio.gather(balance_task, arbitrage_loop_task, *ticker_watch_tasks)
    except Exception as e:
        logger.critical(f"程序顶层发生错误: {e}", exc_info=True)
    finally:
        logger.info("--- 开始关闭程序 ---")
        if application.updater.running: await application.updater.stop()
        if application.running: await application.stop()
        await application.shutdown()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task(): task.cancel()
        await asyncio.gather(*[t for t in asyncio.all_tasks() if t is not asyncio.current_task()], return_exceptions=True)
        await exchange.close()
        logger.info("--- 程序关闭完成 ---")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C 信号，正在关闭...")