#pragma once
#ifndef ARBITRAGE_OPERATIONS_H
#define ARBITRAGE_OPERATIONS_H

#include <string> // 为了在注释中使用 std::string

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
    /**
     * @brief (C++ DLL) 评估套利机会的风险。
     * @param cycle_info_json 套利路径信息的 JSON 字符串 (包含 nodes, trades)。
     * @param start_amount_str 起始金额的字符串表示 (将在 C++ 中转为 double)。
     * @param order_books_json 路径中所有交易对的订单簿数据的 JSON 字符串。
     *                         格式: {"PAIR1": {"bids": [[price_str, amount_str], ...], "asks": [...]}, "PAIR2": ...}
     * @param markets_json 市场数据的 JSON 字符串 (包含 limits, precision, base, quote)。
     * @param tickers_json 当前 Ticker 快照的 JSON 字符串 (包含 bid, ask)。
     * @param config_json 配置信息的 JSON 字符串 (包含 risk_assessment_enabled, taker_fee_rate, order_book_depth, min_profit_after_slippage_percent, max_allowed_slippage_percent_total, min_depth_required_usd, max_bid_ask_spread_percent_per_step, stablecoin_preference)。
     * @return 风险评估结果的 JSON 字符串。调用者需要使用 free_memory_ops 释放此指针。
     *         返回的 JSON 包含: is_viable (bool), estimated_profit_percent_after_slippage (double), total_estimated_slippage_percent (double), reasons (array[string]), details (array[object])。
     *         如果发生内部错误，返回包含 "error" 字段的 JSON。
     */
    DLL_EXPORT int assess_risk_cpp_buffered(const char* cycle_info_json,
        const char* start_amount_str,
        const char* order_books_json,
        const char* markets_json,
        const char* tickers_json,
        const char* config_json,
        char* output_buffer, // <-- 新增
        int buffer_size);    // <-- 新增);

    /**
     * @brief (C++ DLL) 模拟完整的潜在执行路径，包括必要的闪兑。
     * @param cycle_info_json 套利路径信息的 JSON 字符串。
     * @param actual_start_currency_str 实际起始货币 (例如 "USDT")。
     * @param actual_start_amount_str 实际起始金额的字符串表示。
     * @param end_with_usdt 是否强制以 USDT 结束 (1 表示 true, 0 表示 false)。
     * @param tickers_json 当前 Ticker 快照的 JSON 字符串。
     * @param markets_json 市场数据的 JSON 字符串。
     * @param config_json 配置信息的 JSON 字符串 (包含 taker_fee_rate, min_profit_full_sim_percent)。
     * @return 全路径模拟结果的 JSON 字符串。调用者需要使用 free_memory_ops 释放此指针。
     *         返回的 JSON 包含: verified (bool), profit_percent (double, 可能为特殊负值), profit_amount (double), final_amount (double), final_currency (string), log_messages (array[string]), reason (string)。
     *         如果发生内部错误，返回包含 "error" 字段的 JSON。
     */
    DLL_EXPORT int simulate_full_cpp_buffered(const char* cycle_info_json,
        const char* actual_start_currency_str,
        const char* actual_start_amount_str,
        int end_with_usdt,
        const char* tickers_json,
        const char* markets_json,
        const char* config_json,
        char* output_buffer, // <-- 新增
        int buffer_size);    // <-- 新增);


}

#endif // ARBITRAGE_OPERATIONS_H