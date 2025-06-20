#define _CRT_SECURE_NO_WARNINGS // 禁用 strcpy 等函数的安全警告
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <cmath> // For std::log
#include <stdexcept>
#include <limits>
#include <algorithm> // For std::sort
#include <iomanip>   // For std::setprecision (debugging)
#include <cstring>   // For std::strlen, std::strcpy

// --- 包含 nlohmann/json.hpp ---
// 确保此文件在你的包含路径中，或者提供正确路径
#include "json.hpp" // 或者 #include <nlohmann/json.hpp>
// 使用 nlohmann::json 的命名空间别名
using json = nlohmann::json;

// --- Bellman-Ford 相关结构和函数 (保持不变) ---
struct Edge {
    int from_node;
    int to_node;
    double weight;
    std::string pair_symbol; // 使用 std::string
    std::string trade_type;  // 使用 std::string
};

// 假设 find_negative_cycles 函数在这里定义或在别处定义并链接

// --- 构建图的 C++ 函数 ---

// 定义导出宏
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT char* build_graph_cpp(
    const char* tickers_json_str,
    const char* markets_json_str,
    double taker_fee_rate)
{
    try {
        // 1. 解析输入的 JSON 字符串
        json tickers_json = json::parse(tickers_json_str);
        json markets_json = json::parse(markets_json_str);

        if (!tickers_json.is_object() || !markets_json.is_object()) {
            std::cerr << "[C++] Error: Input JSON is not an object." << std::endl;
            return nullptr;
        }

        // 2. 预处理和收集货币
        std::set<std::string> unique_currencies;
        std::vector<std::string> valid_symbols_for_edges;

        // --- 修改: 使用传统迭代器遍历 tickers_json ---
        for (auto it = tickers_json.begin(); it != tickers_json.end(); ++it) {
            const std::string& symbol = it.key(); // 获取键 (symbol)
            const json& ticker = it.value();    // 获取值 (ticker 对象)
            // --- 结束修改 ---

                // 检查 ticker 是否有效且包含必要字段
            if (!ticker.is_object() || !ticker.contains("ask") || !ticker.contains("bid")) {
                continue;
            }
            // 检查 market 是否存在且有效
            if (!markets_json.contains(symbol)) {
                continue;
            }
            const auto& market = markets_json[symbol];
            if (!market.is_object() || !market.value("active", false) || !market.value("spot", false) ||
                !market.contains("base") || !market.contains("quote")) {
                continue;
            }

            // 检查价格有效性 (必须是数字且 > 0)
            double ask_price = 0.0, bid_price = 0.0;
            bool ask_valid = false, bid_valid = false;

            // --- 修改: 分开获取字符串和转换 ---
            if (ticker["ask"].is_number()) {
                ask_price = ticker["ask"].get<double>();
                ask_valid = ask_price > 0;
            }
            else if (ticker["ask"].is_string()) {
                try {
                    std::string ask_str = ticker["ask"].get<std::string>(); // 先获取
                    ask_price = std::stod(ask_str);                         // 再转换
                    ask_valid = ask_price > 0;
                }
                catch (...) { /* 转换失败 */ }
            }

            if (ticker["bid"].is_number()) {
                bid_price = ticker["bid"].get<double>();
                bid_valid = bid_price > 0;
            }
            else if (ticker["bid"].is_string()) {
                try {
                    std::string bid_str = ticker["bid"].get<std::string>(); // 先获取
                    bid_price = std::stod(bid_str);                         // 再转换
                    bid_valid = bid_price > 0;
                }
                catch (...) { /* 转换失败 */ }
            }
            // --- 结束修改 ---

            if (!ask_valid || !bid_valid) {
                continue;
            }

            // 如果所有检查通过，添加货币并记录有效符号
            // --- 修改: 确保 market["base"] 和 market["quote"] 能正确获取字符串 ---
            const std::string& base_currency = market["base"].get_ref<const std::string&>();
            const std::string& quote_currency = market["quote"].get_ref<const std::string&>();
            // --- 结束修改 ---
            unique_currencies.insert(base_currency);
            unique_currencies.insert(quote_currency);
            valid_symbols_for_edges.push_back(symbol);
        }

        if (unique_currencies.empty() || valid_symbols_for_edges.empty()) {
            std::cerr << "[C++] Error: No valid currencies or symbols found after filtering." << std::endl;
            return nullptr;
        }

        // 3. 创建货币索引映射 (保持不变)
        std::vector<std::string> index_to_currency_cpp(unique_currencies.begin(), unique_currencies.end());
        std::sort(index_to_currency_cpp.begin(), index_to_currency_cpp.end()); // 排序以获得确定性

        std::unordered_map<std::string, int> currency_to_index_cpp;
        for (int i = 0; i < index_to_currency_cpp.size(); ++i) {
            currency_to_index_cpp[index_to_currency_cpp[i]] = i;
        }
        int num_currencies = index_to_currency_cpp.size();

        // 4. 构建边列表 (使用 double 计算权重)
        std::vector<json> edges_json_array;
        double fee_multiplier = 1.0 - taker_fee_rate;

        for (const std::string& symbol : valid_symbols_for_edges) {
            const auto& ticker = tickers_json[symbol];
            const auto& market = markets_json[symbol];
            // --- 修改: 确保 market["base"] 和 market["quote"] 能正确获取字符串 ---
            const std::string& base = market["base"].get_ref<const std::string&>();
            const std::string& quote = market["quote"].get_ref<const std::string&>();
            // --- 结束修改 ---

            // 获取已经验证过的价格 (使用修改后的安全转换逻辑)
            double ask_price = 0.0, bid_price = 0.0;
            if (ticker["ask"].is_number()) {
                ask_price = ticker["ask"].get<double>();
            }
            else if (ticker["ask"].is_string()) {
                try { ask_price = std::stod(ticker["ask"].get<std::string>()); }
                catch (...) {}
            }
            if (ticker["bid"].is_number()) {
                bid_price = ticker["bid"].get<double>();
            }
            else if (ticker["bid"].is_string()) {
                try { bid_price = std::stod(ticker["bid"].get<std::string>()); }
                catch (...) {}
            }


            int base_idx = currency_to_index_cpp[base];
            int quote_idx = currency_to_index_cpp[quote];

            // a) 边：Quote -> Base (买入 Base, 花费 Quote) - 使用 Ask Price
            if (ask_price > 0) {
                double net_rate_q_to_b = (1.0 / ask_price) * fee_multiplier;
                if (net_rate_q_to_b > 0) {
                    double weight = -std::log(net_rate_q_to_b);
                    if (std::isfinite(weight)) {
                        edges_json_array.push_back({
                            {"from", quote_idx},
                            {"to", base_idx},
                            {"weight", weight},
                            {"pair", symbol},
                            {"type", "BUY"}
                            });
                    }
                }
            }

            // b) 边：Base -> Quote (卖出 Base, 收到 Quote) - 使用 Bid Price
            if (bid_price > 0) {
                double net_rate_b_to_q = bid_price * fee_multiplier;
                if (net_rate_b_to_q > 0) {
                    double weight = -std::log(net_rate_b_to_q);
                    if (std::isfinite(weight)) {
                        edges_json_array.push_back({
                            {"from", base_idx},
                            {"to", quote_idx},
                            {"weight", weight},
                            {"pair", symbol},
                            {"type", "SELL"}
                            });
                    }
                }
            }
        }

        if (edges_json_array.empty()) {
            std::cerr << "[C++] Error: No valid edges generated." << std::endl;
            return nullptr;
        }

        // 5. 构建最终的输出 JSON (保持不变)
        json output_json;
        output_json["nodes"] = index_to_currency_cpp; // ["BTC", "ETH", "USDT", ...]
        output_json["edges"] = edges_json_array;      // [{"from": 0, "to": 1, ...}, ...]

        // 6. 序列化 JSON 并分配内存返回给 Python (保持不变)
        std::string output_str = output_json.dump();
        char* result_c_str = new char[output_str.length() + 1];
        std::strcpy(result_c_str, output_str.c_str());
        return result_c_str;

    }
    catch (const json::parse_error& e) {
        std::cerr << "[C++] JSON parsing error in build_graph_cpp: " << e.what() << std::endl;
        return nullptr;
    }
    catch (const std::exception& e) {
        std::cerr << "[C++] Standard exception in build_graph_cpp: " << e.what() << std::endl;
        return nullptr;
    }
    catch (...) {
        std::cerr << "[C++] Unknown error occurred in build_graph_cpp." << std::endl;
        return nullptr;
    }
}


// --- 内存释放函数 (保持不变) ---
extern "C" DLLEXPORT void free_memory(char* ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
    }
}

// --- 如果需要，添加 find_negative_cycles 的定义或 extern 声明 ---
// extern "C" DLLEXPORT int find_negative_cycles(...) { ... }