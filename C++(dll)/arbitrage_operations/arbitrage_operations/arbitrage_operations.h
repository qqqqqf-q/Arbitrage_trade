#pragma once
#ifndef ARBITRAGE_OPERATIONS_H
#define ARBITRAGE_OPERATIONS_H

#include <string> // Ϊ����ע����ʹ�� std::string

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
    /**
     * @brief (C++ DLL) ������������ķ��ա�
     * @param cycle_info_json ����·����Ϣ�� JSON �ַ��� (���� nodes, trades)��
     * @param start_amount_str ��ʼ�����ַ�����ʾ (���� C++ ��תΪ double)��
     * @param order_books_json ·�������н��׶ԵĶ��������ݵ� JSON �ַ�����
     *                         ��ʽ: {"PAIR1": {"bids": [[price_str, amount_str], ...], "asks": [...]}, "PAIR2": ...}
     * @param markets_json �г����ݵ� JSON �ַ��� (���� limits, precision, base, quote)��
     * @param tickers_json ��ǰ Ticker ���յ� JSON �ַ��� (���� bid, ask)��
     * @param config_json ������Ϣ�� JSON �ַ��� (���� risk_assessment_enabled, taker_fee_rate, order_book_depth, min_profit_after_slippage_percent, max_allowed_slippage_percent_total, min_depth_required_usd, max_bid_ask_spread_percent_per_step, stablecoin_preference)��
     * @return ������������� JSON �ַ�������������Ҫʹ�� free_memory_ops �ͷŴ�ָ�롣
     *         ���ص� JSON ����: is_viable (bool), estimated_profit_percent_after_slippage (double), total_estimated_slippage_percent (double), reasons (array[string]), details (array[object])��
     *         ��������ڲ����󣬷��ذ��� "error" �ֶε� JSON��
     */
    DLL_EXPORT int assess_risk_cpp_buffered(const char* cycle_info_json,
        const char* start_amount_str,
        const char* order_books_json,
        const char* markets_json,
        const char* tickers_json,
        const char* config_json,
        char* output_buffer, // <-- ����
        int buffer_size);    // <-- ����);

    /**
     * @brief (C++ DLL) ģ��������Ǳ��ִ��·����������Ҫ�����ҡ�
     * @param cycle_info_json ����·����Ϣ�� JSON �ַ�����
     * @param actual_start_currency_str ʵ����ʼ���� (���� "USDT")��
     * @param actual_start_amount_str ʵ����ʼ�����ַ�����ʾ��
     * @param end_with_usdt �Ƿ�ǿ���� USDT ���� (1 ��ʾ true, 0 ��ʾ false)��
     * @param tickers_json ��ǰ Ticker ���յ� JSON �ַ�����
     * @param markets_json �г����ݵ� JSON �ַ�����
     * @param config_json ������Ϣ�� JSON �ַ��� (���� taker_fee_rate, min_profit_full_sim_percent)��
     * @return ȫ·��ģ������ JSON �ַ�������������Ҫʹ�� free_memory_ops �ͷŴ�ָ�롣
     *         ���ص� JSON ����: verified (bool), profit_percent (double, ����Ϊ���⸺ֵ), profit_amount (double), final_amount (double), final_currency (string), log_messages (array[string]), reason (string)��
     *         ��������ڲ����󣬷��ذ��� "error" �ֶε� JSON��
     */
    DLL_EXPORT int simulate_full_cpp_buffered(const char* cycle_info_json,
        const char* actual_start_currency_str,
        const char* actual_start_amount_str,
        int end_with_usdt,
        const char* tickers_json,
        const char* markets_json,
        const char* config_json,
        char* output_buffer, // <-- ����
        int buffer_size);    // <-- ����);


}

#endif // ARBITRAGE_OPERATIONS_H