#define _CRT_SECURE_NO_WARNINGS
#include "arbitrage_operations.h" // 包含函数声明
#include <iostream>               // 用于 cerr 输出
#include <vector>                 // 用于 std::vector
#include <string>                 // 用于 std::string
#include <map>                    // 如果需要 std::map (虽然当前代码没直接用)
#include <cmath>                  // 用于 std::isinf, std::isnan, std::fabs, std::min
#include <stdexcept>              // 用于 std::runtime_error 等异常
#include <limits>                 // 用于 std::numeric_limits
#include <sstream>                // 用于 std::stringstream (格式化 double)
#include <iomanip>                // 用于 std::setprecision, std::fixed
#include <algorithm>              // 用于 std::transform, std::all_of, (std::min 在 cmath 或 algorithm 中)
#include <cstring>                // 需要包含 <cstring> for strlen, strncpy, strcpy_s (Windows)
#include "json.hpp"      // 确保 JSON 库头文件可用

using json = nlohmann::json;

// --- 辅助函数 ---

// 将字符串安全转换为 double
double string_to_double(const std::string& s, double default_value = 0.0) {
    if (s.empty()) return default_value;
    try {
        if (s == "inf" || s == "+inf" || s == "Infinity") return std::numeric_limits<double>::infinity();
        if (s == "-inf" || s == "-Infinity") return -std::numeric_limits<double>::infinity();
        if (s == "nan" || s == "NaN") return std::numeric_limits<double>::quiet_NaN();
        size_t processed_chars = 0;
        double val = std::stod(s, &processed_chars);
        // 允许尾随空格，但检查是否有其他非数字字符
        while (processed_chars < s.length() && std::isspace(static_cast<unsigned char>(s[processed_chars]))) {
            processed_chars++;
        }
        if (processed_chars != s.length()) {
            std::cerr << u8"警告: 字符串 '" << s << u8"' 包含无效字符，仅部分转换。" << std::endl;
        }
        return val;
    }
    catch (const std::invalid_argument& ia) {
        std::cerr << u8"警告: 无法将字符串 '" << s << u8"' 转换为 double (invalid_argument)。返回默认值 " << default_value << std::endl;
        return default_value;
    }
    catch (const std::out_of_range& oor) {
        std::cerr << u8"警告: 字符串 '" << s << u8"' 转换为 double 超出范围。返回默认值 " << default_value << std::endl;
        return default_value;
    }
    catch (const std::exception& e) {
        std::cerr << u8"警告: 转换字符串 '" << s << u8"' 为 double 时发生未知错误: " << e.what() << u8"。返回默认值 " << default_value << std::endl;
        return default_value;
    }
}

// 格式化 double 为字符串
std::string format_double(double d, int precision = 8) {
    if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
    if (std::isnan(d)) return "NaN";
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << d;
    std::string s = ss.str();
    // 移除末尾多余的零
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    // 如果移除后末尾是小数点，也移除小数点
    if (!s.empty() && s.back() == '.') { s.pop_back(); }
    // 修正 "-0" 或 "-0." 的情况
    if (s == "-0" || s == "-0.") { return "0"; }
    // 修正极小的负数显示问题
    if (s.length() > 1 && s[0] == '-' && std::all_of(s.begin() + 1, s.end(), [](char c) { return c == '0' || c == '.'; })) { return "0"; }
    return s;
}

// 获取 JSON 值 (带默认值和类型检查)
template <typename T> T get_json_value(const json& j, const std::string& key, const T& default_value) {
    if (j.contains(key) && !j.at(key).is_null()) {
        try { return j.at(key).get<T>(); }
        catch (const std::exception& e) {
            std::cerr << u8"警告: 获取 JSON 键 '" << key << u8"' (期望类型 " << typeid(T).name() << u8") 时出错: " << e.what() << u8"。 使用默认值。" << std::endl;
            return default_value;
        }
    } return default_value;
}
// 从字符串或数字获取 double
double get_json_double_from_string(const json& j, const std::string& key, double default_value) {
    if (j.contains(key)) {
        const auto& val = j.at(key); // 使用 const auto& 避免复制
        if (val.is_string()) return string_to_double(val.get<std::string>(), default_value);
        if (val.is_number()) return val.get<double>();
    } return default_value;
}
// 获取 bool (支持多种形式)
bool get_json_bool(const json& j, const std::string& key, bool default_value) {
    if (j.contains(key)) {
        const auto& val = j.at(key);
        if (val.is_boolean()) return val.get<bool>();
        if (val.is_string()) { std::string s = val.get<std::string>(); std::transform(s.begin(), s.end(), s.begin(), ::tolower); if (s == "true" || s == "1" || s == "on" || s == "yes") return true; if (s == "false" || s == "0" || s == "off" || s == "no") return false; }
        if (val.is_number()) return val.get<double>() != 0.0;
    } return default_value;
}

// 模拟闪兑 (内部使用) - 保持不变
json simulate_swap_order_cpp(const std::string& from_currency, const std::string& to_currency, double from_amount, const json& markets, const json& tickers, double fee_rate) {
    json result;
    result["estimated_to_amount"] = 0.0;
    result["method"] = nullptr;
    std::string direct_symbol_forward = to_currency + "/" + from_currency;
    std::string direct_symbol_backward = from_currency + "/" + to_currency;
    std::string intermediate_currency = "USDT";
    std::string symbol_from_intermediate = from_currency + "/" + intermediate_currency;
    std::string symbol_to_intermediate = to_currency + "/" + intermediate_currency;
    double estimated_to_amount = 0.0;

    if (tickers.contains(direct_symbol_backward) && tickers[direct_symbol_backward].is_object()) {
        const auto& ticker_back = tickers[direct_symbol_backward];
        if (ticker_back.contains("bid") && ticker_back["bid"].is_string()) {
            double bid_price = string_to_double(ticker_back["bid"].get<std::string>());
            if (bid_price > 0) { estimated_to_amount = (from_amount * bid_price) * (1.0 - fee_rate); result["method"] = "sell"; }
        }
    }
    if (estimated_to_amount <= 0 && tickers.contains(direct_symbol_forward) && tickers[direct_symbol_forward].is_object()) {
        const auto& ticker_fwd = tickers[direct_symbol_forward];
        if (ticker_fwd.contains("ask") && ticker_fwd["ask"].is_string()) {
            double ask_price = string_to_double(ticker_fwd["ask"].get<std::string>());
            if (ask_price > 0) { estimated_to_amount = (from_amount / ask_price) * (1.0 - fee_rate); result["method"] = "buy_cost"; }
        }
    }
    if (estimated_to_amount <= 0) {
        bool from_int_ok = false; double bid_price_from = 0.0;
        if (tickers.contains(symbol_from_intermediate) && tickers[symbol_from_intermediate].is_object() && tickers[symbol_from_intermediate].contains("bid") && tickers[symbol_from_intermediate]["bid"].is_string()) {
            bid_price_from = string_to_double(tickers[symbol_from_intermediate]["bid"].get<std::string>()); if (bid_price_from > 0) from_int_ok = true;
        }
        bool to_int_ok = false; double ask_price_to = 0.0;
        if (tickers.contains(symbol_to_intermediate) && tickers[symbol_to_intermediate].is_object() && tickers[symbol_to_intermediate].contains("ask") && tickers[symbol_to_intermediate]["ask"].is_string()) {
            ask_price_to = string_to_double(tickers[symbol_to_intermediate]["ask"].get<std::string>()); if (ask_price_to > 0) to_int_ok = true;
        }
        if (from_int_ok && to_int_ok) {
            double intermediate_amount_net = (from_amount * bid_price_from) * (1.0 - fee_rate);
            if (intermediate_amount_net > 1e-12) { estimated_to_amount = (intermediate_amount_net / ask_price_to) * (1.0 - fee_rate); result["method"] = "intermediate"; }
            else { estimated_to_amount = 0.0; }
        }
    }
    result["estimated_to_amount"] = (estimated_to_amount > 0) ? estimated_to_amount : 0.0;
    return result;
}


// --- 核心实现: 风险评估 (缓冲区版本 - 移除 goto) ---
DLL_EXPORT int assess_risk_cpp_buffered(const char* cycle_info_json,
    const char* start_amount_str,
    const char* order_books_json,
    const char* markets_json,
    const char* tickers_json,
    const char* config_json,
    char* output_buffer,
    int buffer_size)
{
    json result_json;
    int return_status = 0; // 默认成功

    if (output_buffer == nullptr || buffer_size <= 1) { std::cerr << "[C++] assess_risk_cpp_buffered: 无效的输出缓冲区。" << std::endl; return -5; }
    output_buffer[0] = '\0'; // 初始化

    try {
        // 1. 解析输入
        json cycle_info = json::parse(cycle_info_json);
        json order_books = json::parse(order_books_json);
        json markets = json::parse(markets_json);
        json tickers = json::parse(tickers_json);
        json config = json::parse(config_json);
        double start_amount = string_to_double(start_amount_str);

        // 2. 获取配置 & 处理风险评估未启用情况
        bool risk_assessment_enabled = get_json_bool(config, "risk_assessment_enabled", false);
        if (!risk_assessment_enabled) {
            result_json["is_viable"] = true; result_json["reasons"] = { u8"风险评估未启用" };
            // 尝试获取 full_simulation_profit_percent，如果 cycle_info_json 中有的话
            if (cycle_info.contains("full_simulation_profit_percent")) {
                result_json["estimated_profit_percent_after_slippage"] = get_json_double_from_string(cycle_info, "full_simulation_profit_percent", -1.0);
            }
            else {
                result_json["estimated_profit_percent_after_slippage"] = -1.0; // 或者其他默认值
            }
            result_json["total_estimated_slippage_percent"] = 0.0; result_json["details"] = json::array();
            // 注意：函数会继续执行到末尾的输出部分
        }
        else {
            // --- 只有启用了风险评估才执行核心逻辑 ---
            double taker_fee_rate = get_json_double_from_string(config, "taker_fee_rate", 0.001);
            double min_profit_req = get_json_double_from_string(config, "min_profit_after_slippage_percent", 0.05);
            double max_slip_req = get_json_double_from_string(config, "max_allowed_slippage_percent_total", 0.15);
            double min_depth_usd_req = get_json_double_from_string(config, "min_depth_required_usd", 100.0);
            double max_spread_req = get_json_double_from_string(config, "max_bid_ask_spread_percent_per_step", 0.50);
            std::vector<std::string> stablecoin_prefs = get_json_value<std::vector<std::string>>(config, "stablecoin_preference", { "USDT", "USDC" });

            // 3. 初始化结果
            result_json["is_viable"] = true; result_json["estimated_profit_percent_after_slippage"] = -999.0;
            result_json["total_estimated_slippage_percent"] = -999.0; result_json["reasons"] = json::array();
            result_json["details"] = json::array(); double total_slippage_cost_usd = 0.0; double intermediate_amount = start_amount;

            if (!cycle_info.contains("nodes") || !cycle_info["nodes"].is_array() || cycle_info["nodes"].empty()) throw std::runtime_error("C++ assess_risk: cycle_info 缺少有效 'nodes' 数组");
            if (!cycle_info.contains("trades") || !cycle_info["trades"].is_array()) throw std::runtime_error("C++ assess_risk: cycle_info 缺少有效 'trades' 数组");
            if (cycle_info["nodes"][0].is_null() || !cycle_info["nodes"][0].is_string()) throw std::runtime_error("C++ assess_risk: cycle_info 'nodes'[0] 无效");
            std::string path_start_currency = cycle_info["nodes"][0].get<std::string>();
            std::string current_currency = path_start_currency; const json& trades = cycle_info["trades"]; // 使用 const&

            // 4. 估算初始 USD 价值
            double start_value_usd_est = 0.0;
            if (current_currency == "USDT") { start_value_usd_est = start_amount; }
            else {
                std::string ticker_fwd_sym = current_currency + "/USDT", ticker_rev_sym = "USDT/" + current_currency; double price = 0.0;
                if (tickers.contains(ticker_fwd_sym) && tickers[ticker_fwd_sym].is_object() && tickers[ticker_fwd_sym].contains("bid") && tickers[ticker_fwd_sym]["bid"].is_string()) price = string_to_double(tickers[ticker_fwd_sym]["bid"].get<std::string>());
                if (price <= 0 && tickers.contains(ticker_rev_sym) && tickers[ticker_rev_sym].is_object() && tickers[ticker_rev_sym].contains("ask") && tickers[ticker_rev_sym]["ask"].is_string()) { double ask_rev = string_to_double(tickers[ticker_rev_sym]["ask"].get<std::string>()); if (ask_rev > 0) price = 1.0 / ask_rev; }
                if (price > 0) { start_value_usd_est = start_amount * price; }
                else { bool is_other_stable = false; for (const auto& stable : stablecoin_prefs) if (current_currency == stable && current_currency != "USDT") { is_other_stable = true; break; } if (is_other_stable) start_value_usd_est = start_amount; else result_json["reasons"].push_back(u8"无法估算起始 USD 价值 for " + current_currency); }
            }

            // 5. 遍历交易步骤
            for (size_t i = 0; i < trades.size(); ++i) {
                if (!trades[i].is_object() || !trades[i].contains("pair") || !trades[i]["pair"].is_string() || !trades[i].contains("type") || !trades[i]["type"].is_string() || !trades[i].contains("from") || !trades[i]["from"].is_string() || !trades[i].contains("to") || !trades[i]["to"].is_string()) throw std::runtime_error("C++ assess_risk: trades[" + std::to_string(i) + "] 结构无效");
                const auto& trade = trades[i]; std::string pair = trade["pair"].get<std::string>(); std::string trade_type = trade["type"].get<std::string>();
                std::string from_currency = trade["from"].get<std::string>(); std::string to_currency = trade["to"].get<std::string>(); int step_num = i + 1;

                json step_detail; step_detail["step"] = step_num; step_detail["pair"] = pair; step_detail["type"] = trade_type; step_detail["slippage_percent"] = std::numeric_limits<double>::quiet_NaN(); step_detail["spread_percent"] = std::numeric_limits<double>::quiet_NaN(); step_detail["depth_ok"] = false; step_detail["depth_usd"] = 0.0; step_detail["limits_ok"] = true; step_detail["message"] = "";

                if (intermediate_amount <= 0) { std::string msg = u8"步骤 " + std::to_string(step_num) + u8": 上一步金额无效 (" + format_double(intermediate_amount) + ")"; result_json["reasons"].push_back(msg); step_detail["message"] = msg; result_json["details"].push_back(step_detail); intermediate_amount = -1.0; break; }
                if (current_currency != from_currency) { std::string msg = u8"逻辑错误：步骤 " + std::to_string(step_num) + u8" 需发 " + from_currency + u8", 持有 " + current_currency; result_json["reasons"].push_back(msg); step_detail["message"] = msg; result_json["details"].push_back(step_detail); intermediate_amount = -1.0; break; }
                if (!markets.contains(pair) || !markets[pair].is_object() || !markets[pair].contains("base") || !markets[pair].contains("quote") || !markets[pair].contains("limits")) { std::string msg = u8"步骤 " + std::to_string(step_num) + u8": 市场数据无效 " + pair; result_json["reasons"].push_back(msg); step_detail["message"] = msg; result_json["details"].push_back(step_detail); intermediate_amount = -1.0; continue; } // continue might be better than break here
                const auto& market = markets[pair]; std::string base_curr = market["base"].get<std::string>(); std::string quote_curr = market["quote"].get<std::string>();
                double min_amount = 0.0, min_cost = 0.0;
                if (market["limits"].is_object()) { if (market["limits"].contains("amount") && market["limits"]["amount"].is_object()) min_amount = get_json_double_from_string(market["limits"]["amount"], "min", 0.0); if (market["limits"].contains("cost") && market["limits"]["cost"].is_object()) min_cost = get_json_double_from_string(market["limits"]["cost"], "min", 0.0); }

                if (!order_books.contains(pair) || !order_books[pair].is_object()) { std::string msg = u8"步骤 " + std::to_string(step_num) + u8": 缺订单簿 " + pair; result_json["reasons"].push_back(msg + u8" (无法评估)"); step_detail["message"] = msg; result_json["details"].push_back(step_detail); intermediate_amount = -1.0; continue; }
                const auto& book = order_books[pair]; const json& bids = book.contains("bids") && book["bids"].is_array() ? book["bids"] : json::array(); const json& asks = book.contains("asks") && book["asks"].is_array() ? book["asks"] : json::array();
                if (bids.empty() || asks.empty() || !bids[0].is_array() || bids[0].size() < 2 || !asks[0].is_array() || asks[0].size() < 2) { std::string msg = u8"步骤 " + std::to_string(step_num) + u8": 订单簿不完整 " + pair; result_json["reasons"].push_back(msg + u8" (无法评估)"); step_detail["message"] = msg; result_json["details"].push_back(step_detail); intermediate_amount = -1.0; continue; }

                double top_bid = string_to_double(bids[0][0].get<std::string>()); double top_ask = string_to_double(asks[0][0].get<std::string>()); double spread_percent = std::numeric_limits<double>::infinity();
                if (top_bid > 0 && std::isfinite(top_ask) && top_ask > 0) { spread_percent = ((top_ask - top_bid) / top_ask) * 100.0; step_detail["spread_percent"] = spread_percent; if (spread_percent > max_spread_req) { std::string msg = u8"价差过高 (" + format_double(spread_percent, 3) + "%)"; result_json["reasons"].push_back(u8"步骤 " + std::to_string(step_num) + ": " + msg); step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; } }
                else { step_detail["spread_percent"] = spread_percent; std::string msg = u8"无法计算价差"; result_json["reasons"].push_back(u8"步骤 " + std::to_string(step_num) + ": " + msg); step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; }

                double estimated_executed_amount = 0.0, sim_average_price = 0.0; double slippage_percent_step = std::numeric_limits<double>::quiet_NaN();
                double accumulated_base = 0.0, cost_accumulated = 0.0; double accumulated_quote = 0.0, amount_sold = 0.0;
                double depth_usd_available = 0.0; double approx_trade_value_usd = 0.0;

                if (current_currency == "USDT") { approx_trade_value_usd = intermediate_amount; }
                else { /* ... 估算 approx_trade_value_usd ... */
                    std::string ticker_fwd_sym_usd = current_currency + "/USDT", ticker_rev_sym_usd = "USDT/" + current_currency; double price_usd = 0.0;
                    if (tickers.contains(ticker_fwd_sym_usd) && tickers[ticker_fwd_sym_usd].is_object() && tickers[ticker_fwd_sym_usd].contains("bid") && tickers[ticker_fwd_sym_usd]["bid"].is_string()) price_usd = string_to_double(tickers[ticker_fwd_sym_usd]["bid"].get<std::string>());
                    if (price_usd <= 0 && tickers.contains(ticker_rev_sym_usd) && tickers[ticker_rev_sym_usd].is_object() && tickers[ticker_rev_sym_usd].contains("ask") && tickers[ticker_rev_sym_usd]["ask"].is_string()) { double ask_rev_usd = string_to_double(tickers[ticker_rev_sym_usd]["ask"].get<std::string>()); if (ask_rev_usd > 0) price_usd = 1.0 / ask_rev_usd; }
                    if (price_usd > 0) { approx_trade_value_usd = intermediate_amount * price_usd; }
                    else { bool is_other_stable = false; for (const auto& stable : stablecoin_prefs) { if (current_currency == stable && current_currency != "USDT") { is_other_stable = true; break; } } if (is_other_stable) approx_trade_value_usd = intermediate_amount; }
                }

                if (trade_type == "BUY") {
                    double amount_to_spend = intermediate_amount;
                    if (min_cost > 0 && amount_to_spend < min_cost) { std::string msg = u8"花费 " + format_double(amount_to_spend) + u8" 低于最小成本 " + format_double(min_cost); result_json["reasons"].push_back(u8"步骤 " + std::to_string(step_num) + ": " + msg); step_detail["limits_ok"] = false; step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; }
                    for (const auto& level : asks) {
                        if (!level.is_array() || level.size() < 2 || !level[0].is_string() || !level[1].is_string()) continue;
                        double price = string_to_double(level[0].get<std::string>()); double amount = string_to_double(level[1].get<std::string>()); if (price <= 0 || amount <= 0) continue; double cost_at_level = price * amount;
                        double level_usd_value = 0.0; if (quote_curr == "USDT") { level_usd_value = cost_at_level; }
                        else { /* ... 估算 level_usd_value ... */ std::string tkr_fwd_q = quote_curr + "/USDT", tkr_rev_q = "USDT/" + quote_curr; double price_q = 0.0; if (tickers.contains(tkr_fwd_q) && tickers[tkr_fwd_q].is_object() && tickers[tkr_fwd_q].contains("bid") && tickers[tkr_fwd_q]["bid"].is_string()) price_q = string_to_double(tickers[tkr_fwd_q]["bid"].get<std::string>()); if (price_q <= 0 && tickers.contains(tkr_rev_q) && tickers[tkr_rev_q].is_object() && tickers[tkr_rev_q].contains("ask") && tickers[tkr_rev_q]["ask"].is_string()) { double ask_rev_q = string_to_double(tickers[tkr_rev_q]["ask"].get<std::string>()); if (ask_rev_q > 0) price_q = 1.0 / ask_rev_q; } if (price_q > 0) level_usd_value = cost_at_level * price_q; } depth_usd_available += level_usd_value;
                        double remaining_spend = amount_to_spend - cost_accumulated; if (remaining_spend <= 1e-9) break;
                        double buy_amount_base_at_level = (std::min)(remaining_spend / price, amount); double cost_this_level = buy_amount_base_at_level * price; accumulated_base += buy_amount_base_at_level; cost_accumulated += cost_this_level;
                    }
                    estimated_executed_amount = accumulated_base;
                    if (accumulated_base > 1e-12) {
                        sim_average_price = cost_accumulated / accumulated_base; if (std::isfinite(top_ask) && top_ask > 0) slippage_percent_step = ((sim_average_price - top_ask) / top_ask) * 100.0;
                        if (min_amount > 0 && estimated_executed_amount < min_amount) { std::string msg = u8"买入量 " + format_double(estimated_executed_amount) + u8" 低于最小 " + format_double(min_amount); result_json["reasons"].push_back(u8"步骤 " + std::to_string(step_num) + ": " + msg); step_detail["limits_ok"] = false; step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; }
                    }
                    else { sim_average_price = 0.0; slippage_percent_step = std::numeric_limits<double>::infinity(); std::string msg = u8"无法模拟买入"; result_json["reasons"].push_back(u8"步骤 " + std::to_string(step_num) + ": " + msg); step_detail["limits_ok"] = false; step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; intermediate_amount = -1.0; }
                }
                else if (trade_type == "SELL") {
                    double amount_to_sell = intermediate_amount;
                    if (min_amount > 0 && amount_to_sell < min_amount) { std::string msg = u8"卖出量 " + format_double(amount_to_sell) + u8" 低于最小 " + format_double(min_amount); result_json["reasons"].push_back(u8"步骤 " + std::to_string(step_num) + ": " + msg); step_detail["limits_ok"] = false; step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; }
                    for (const auto& level : bids) {
                        if (!level.is_array() || level.size() < 2 || !level[0].is_string() || !level[1].is_string()) continue;
                        double price = string_to_double(level[0].get<std::string>()); double amount = string_to_double(level[1].get<std::string>()); if (price <= 0 || amount <= 0) continue;
                        double level_usd_value = 0.0; if (base_curr == "USDT") { level_usd_value = amount * price; }
                        else { /* ... 估算 level_usd_value ... */ std::string tkr_fwd_b = base_curr + "/USDT", tkr_rev_b = "USDT/" + base_curr; double price_b = 0.0; if (tickers.contains(tkr_fwd_b) && tickers[tkr_fwd_b].is_object() && tickers[tkr_fwd_b].contains("bid") && tickers[tkr_fwd_b]["bid"].is_string()) price_b = string_to_double(tickers[tkr_fwd_b]["bid"].get<std::string>()); if (price_b <= 0 && tickers.contains(tkr_rev_b) && tickers[tkr_rev_b].is_object() && tickers[tkr_rev_b].contains("ask") && tickers[tkr_rev_b]["ask"].is_string()) { double ask_rev_b = string_to_double(tickers[tkr_rev_b]["ask"].get<std::string>()); if (ask_rev_b > 0) price_b = 1.0 / ask_rev_b; } if (price_b > 0) level_usd_value = amount * price_b; } depth_usd_available += level_usd_value;
                        double remaining_sell = amount_to_sell - amount_sold; if (remaining_sell <= 1e-12) break;
                        double sell_amount_base_at_level = (std::min)(remaining_sell, amount); double quote_received_this_level = sell_amount_base_at_level * price; accumulated_quote += quote_received_this_level; amount_sold += sell_amount_base_at_level;
                    }
                    estimated_executed_amount = accumulated_quote;
                    if (amount_sold > 1e-12) {
                        sim_average_price = accumulated_quote / amount_sold; if (top_bid > 0) slippage_percent_step = ((top_bid - sim_average_price) / top_bid) * 100.0;
                        if (min_cost > 0 && estimated_executed_amount < min_cost) { std::string msg = u8"卖出所得 " + format_double(estimated_executed_amount) + u8" 低于最小成本 " + format_double(min_cost); step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; }
                    }
                    else { sim_average_price = 0.0; slippage_percent_step = std::numeric_limits<double>::infinity(); std::string msg = u8"无法模拟卖出"; result_json["reasons"].push_back(u8"步骤 " + std::to_string(step_num) + ": " + msg); step_detail["limits_ok"] = false; step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; intermediate_amount = -1.0; }
                }

                step_detail["depth_usd"] = depth_usd_available;
                if (depth_usd_available >= min_depth_usd_req) { step_detail["depth_ok"] = true; }
                else { std::string msg = u8"深度不足 (仅约 $" + format_double(depth_usd_available, 2) + ")"; result_json["reasons"].push_back(u8"步骤 " + std::to_string(step_num) + ": " + msg); step_detail["depth_ok"] = false; step_detail["message"] = step_detail["message"].get<std::string>() + msg + "; "; }

                if (intermediate_amount >= 0) {
                    step_detail["slippage_percent"] = slippage_percent_step; double slippage_cost_step_usd = 0.0;
                    if (std::isfinite(slippage_percent_step) && approx_trade_value_usd > 0 && slippage_percent_step > 0) slippage_cost_step_usd = approx_trade_value_usd * (slippage_percent_step / 100.0);
                    total_slippage_cost_usd += slippage_cost_step_usd;
                    double next_intermediate_amount = estimated_executed_amount * (1.0 - taker_fee_rate);
                    current_currency = to_currency; intermediate_amount = next_intermediate_amount;
                    if (intermediate_amount < 0) intermediate_amount = 0;
                }
                result_json["details"].push_back(step_detail);
                if (intermediate_amount < 0) { result_json["reasons"].push_back(u8"因步骤 " + std::to_string(step_num) + u8" 错误中止评估"); break; }
            } // end for loop (trades)

            // 6. 计算最终结果
            if (intermediate_amount >= 0 && current_currency == path_start_currency) {
                double profit_amount = intermediate_amount - start_amount;
                if (start_amount > 1e-12) result_json["estimated_profit_percent_after_slippage"] = (profit_amount / start_amount) * 100.0; else result_json["estimated_profit_percent_after_slippage"] = 0.0;
                if (start_value_usd_est > 1e-9) result_json["total_estimated_slippage_percent"] = (total_slippage_cost_usd / start_value_usd_est) * 100.0; else if (total_slippage_cost_usd == 0.0) result_json["total_estimated_slippage_percent"] = 0.0; else result_json["total_estimated_slippage_percent"] = std::numeric_limits<double>::infinity();
                double final_profit_perc = result_json["estimated_profit_percent_after_slippage"].get<double>(); double final_slip_perc = result_json["total_estimated_slippage_percent"].get<double>();
                if (final_profit_perc < min_profit_req) result_json["reasons"].push_back(u8"利润率 (" + format_double(final_profit_perc, 4) + u8"%) 低于要求");
                if (std::isfinite(final_slip_perc) && final_slip_perc > max_slip_req) result_json["reasons"].push_back(u8"总滑点 (" + format_double(final_slip_perc, 4) + u8"%) 高于阈值");
                else if (std::isinf(final_slip_perc)) result_json["reasons"].push_back(u8"无法计算总滑点%");
            }
            else if (intermediate_amount < 0) { result_json["estimated_profit_percent_after_slippage"] = -998.0; }
            else if (current_currency != path_start_currency) { result_json["reasons"].push_back(u8"最终货币(" + current_currency + u8")与起始不符"); result_json["estimated_profit_percent_after_slippage"] = -997.0; }

            // 7. 判断最终可行性
            bool is_viable = true; if (intermediate_amount < 0) is_viable = false;
            // 检查 profit_check 是否有效再比较
            double profit_check = -999.0;
            if (result_json.contains("estimated_profit_percent_after_slippage") && result_json["estimated_profit_percent_after_slippage"].is_number()) {
                profit_check = result_json["estimated_profit_percent_after_slippage"].get<double>();
            }
            if (!std::isfinite(profit_check) || profit_check < min_profit_req) is_viable = false;

            // 检查 slip_check 是否有效再比较
            double slip_check = std::numeric_limits<double>::quiet_NaN();
            if (result_json.contains("total_estimated_slippage_percent") && result_json["total_estimated_slippage_percent"].is_number()) {
                slip_check = result_json["total_estimated_slippage_percent"].get<double>();
            }
            if (std::isfinite(slip_check) && slip_check > max_slip_req) is_viable = false;

            if (intermediate_amount >= 0 && current_currency != path_start_currency) is_viable = false;
            if (result_json.contains("details") && result_json["details"].is_array()) { for (const auto& detail : result_json["details"]) { if (detail.contains("depth_ok") && !detail["depth_ok"].get<bool>()) is_viable = false; if (detail.contains("limits_ok") && !detail["limits_ok"].get<bool>()) is_viable = false; } }
            result_json["is_viable"] = is_viable;

        } // end else (risk_assessment_enabled)

    // --- catch 块：捕获异常，设置错误信息和状态码 ---
    }
    catch (const json::parse_error& e) {
        std::cerr << "[C++ Exception] JSON 解析错误 (assess_risk_buffered): " << e.what() << " at byte " << e.byte << std::endl; result_json.clear(); result_json["error"] = "C++ JSON 解析错误: " + std::string(e.what()); return_status = -2;
    }
    catch (const json::type_error& e) {
        std::cerr << "[C++ Exception] JSON 类型错误 (assess_risk_buffered): " << e.what() << std::endl; result_json.clear(); result_json["error"] = "C++ JSON 类型错误: " + std::string(e.what()); return_status = -2;
    }
    catch (const std::exception& e) {
        std::cerr << "[C++ Exception] 评估风险时发生 C++ 异常: " << e.what() << std::endl; result_json.clear(); result_json["error"] = "C++ 内部错误: " + std::string(e.what()); return_status = -3;
    }
    catch (...) { std::cerr << "[C++ Exception] 评估风险时发生未知 C++ 异常。" << std::endl; result_json.clear(); result_json["error"] = u8"未知的 C++ 内部错误"; return_status = -3; }

    // --- 将最终的 result_json (可能包含错误信息) 写入 output_buffer ---
    std::string result_str;
    try { result_str = result_json.dump(-1, ' ', false, json::error_handler_t::replace); }
    catch (const std::exception& dump_e) {
        std::cerr << "[C++] JSON dump 错误 (assess_risk): " << dump_e.what() << std::endl;
        std::string err_json_str = "{\"error\":\"C++ JSON dump error\"}";
        if (static_cast<size_t>(buffer_size) > err_json_str.length()) { strncpy(output_buffer, err_json_str.c_str(), buffer_size - 1); output_buffer[err_json_str.length()] = '\0'; }
        else { output_buffer[0] = '\0'; }
        return (return_status == 0) ? -2 : return_status;
    }
    size_t required_size = result_str.length() + 1;
    if (static_cast<size_t>(buffer_size) < required_size) {
        std::cerr << "[C++] assess_risk_buffered: 输出缓冲区太小。需要 " << required_size << ", 提供 " << buffer_size << std::endl;
        output_buffer[0] = '\0'; return -1;
    }
#ifdef _WIN32
    errno_t err = strcpy_s(output_buffer, buffer_size, result_str.c_str()); if (err != 0) { std::cerr << "[C++] strcpy_s 错误 (assess_risk): " << err << std::endl; output_buffer[0] = '\0'; return -3; }
#else
    strncpy(output_buffer, result_str.c_str(), buffer_size - 1); output_buffer[buffer_size - 1] = '\0'; if (result_str.length() >= static_cast<size_t>(buffer_size - 1)) { std::cerr << "[C++ Warning] assess_risk_buffered: 输出可能被截断。" << std::endl; }
#endif
    return return_status;
}


// --- 核心实现: 全路径模拟 (缓冲区版本 - 移除 goto) ---
DLL_EXPORT int simulate_full_cpp_buffered(const char* cycle_info_json,
    const char* actual_start_currency_str,
    const char* actual_start_amount_str,
    int end_with_usdt_int,
    const char* tickers_json,
    const char* markets_json,
    const char* config_json,
    char* output_buffer,
    int buffer_size)
{
    json result_json;
    int return_status = 0;

    if (output_buffer == nullptr || buffer_size <= 1) { std::cerr << "[C++] simulate_full_cpp_buffered: 无效的输出缓冲区。" << std::endl; return -5; }
    output_buffer[0] = '\0';

    try {
        // 1. 解析输入
        json cycle_info = json::parse(cycle_info_json);
        json tickers = json::parse(tickers_json);
        json markets = json::parse(markets_json); // 假设需要
        json config = json::parse(config_json);

        // 2. 获取配置和初始状态
        std::string actual_start_currency = actual_start_currency_str;
        double actual_start_amount = string_to_double(actual_start_amount_str);
        bool end_with_usdt = (end_with_usdt_int == 1);
        double fee_rate = get_json_double_from_string(config, "taker_fee_rate", 0.001);
        double min_profit_req = get_json_double_from_string(config, "min_profit_full_sim_percent", 0.05);
        std::string sim_current_currency = actual_start_currency;
        double sim_current_amount = actual_start_amount;

        // 3. 检查输入结构
        if (!cycle_info.contains("nodes") || !cycle_info["nodes"].is_array() || cycle_info["nodes"].empty()) throw std::runtime_error("C++ simulate_full_cpp: cycle_info 缺少有效 'nodes' 数组");
        if (!cycle_info.contains("trades") || !cycle_info["trades"].is_array()) throw std::runtime_error("C++ simulate_full_cpp: cycle_info 缺少有效 'trades' 数组");
        if (cycle_info["nodes"][0].is_null() || !cycle_info["nodes"][0].is_string()) throw std::runtime_error("C++ simulate_full_cpp: cycle_info 'nodes'[0] 无效");
        std::string cycle_start_currency = cycle_info["nodes"][0].get<std::string>();
        const json& trades = cycle_info["trades"]; // 使用 const&

        // 4. 检查是否跳过第一步
        size_t start_trade_index = 0;
        bool skip_initial_swap_and_first_step = false;
        if (!trades.empty() && trades.size() > 1) {
            if (!trades[0].is_object() || !trades[0].contains("from") || !trades[0].contains("to") || !trades[0]["from"].is_string() || !trades[0]["to"].is_string()) throw std::runtime_error("C++ simulate_full_cpp: trades[0] 结构无效");
            const auto& first_trade = trades[0];
            if (actual_start_currency != cycle_start_currency && first_trade["from"].get<std::string>() == cycle_start_currency && first_trade["to"].get<std::string>() == actual_start_currency) { skip_initial_swap_and_first_step = true; start_trade_index = 1; }
        }

        // 5. 模拟初始闪兑
        if (!skip_initial_swap_and_first_step && sim_current_currency != cycle_start_currency) {
            json swap_result = simulate_swap_order_cpp(sim_current_currency, cycle_start_currency, sim_current_amount, markets, tickers, fee_rate);
            double received_amount = get_json_value(swap_result, "estimated_to_amount", 0.0); // 使用 get_json_value
            if (received_amount > 1e-12) { sim_current_amount = received_amount; sim_current_currency = cycle_start_currency; }
            else {
                std::string reason = u8"模拟初始闪兑 (" + sim_current_currency + " -> " + cycle_start_currency + u8") 失败或无路径";
                result_json["verified"] = false; result_json["reason"] = reason; result_json["profit_percent"] = -999.0;
                result_json["final_amount"] = 0.0; result_json["final_currency"] = sim_current_currency;
                // 直接跳到函数末尾的输出部分
            }
        }

        // 6. 模拟核心套利环路
        if (start_trade_index < trades.size()) {
            for (size_t i = start_trade_index; i < trades.size(); ++i) {
                if (!trades[i].is_object() || !trades[i].contains("pair") || !trades[i]["pair"].is_string() || !trades[i].contains("type") || !trades[i]["type"].is_string() || !trades[i].contains("from") || !trades[i]["from"].is_string() || !trades[i].contains("to") || !trades[i]["to"].is_string()) throw std::runtime_error("C++ simulate_full_cpp: trades[" + std::to_string(i) + "] 结构无效");
                const auto& trade = trades[i]; std::string pair = trade["pair"].get<std::string>(); std::string trade_type = trade["type"].get<std::string>(); std::string from_curr = trade["from"].get<std::string>(); std::string to_curr = trade["to"].get<std::string>();
                if (!markets.contains(pair) || !markets[pair].is_object() || !markets[pair].contains("base") || !markets[pair].contains("quote")) throw std::runtime_error("模拟时缺少核心市场数据 for " + pair);
                const auto& market = markets[pair]; std::string base_curr = market["base"].get<std::string>(); std::string quote_curr = market["quote"].get<std::string>();
                if (!tickers.contains(pair) || !tickers[pair].is_object()) throw std::runtime_error("模拟时缺少 Ticker 数据 for " + pair);
                const auto& ticker = tickers[pair];
                double net_amount = 0.0, price = 0.0;
                if (trade_type == "BUY") {
                    if (!ticker.contains("ask") || !ticker["ask"].is_string()) throw std::runtime_error("Ticker " + pair + " 缺少有效 ask 价"); price = string_to_double(ticker["ask"].get<std::string>()); if (price <= 0) throw std::runtime_error("Ticker " + pair + " ask 价无效"); net_amount = (sim_current_amount / price) * (1.0 - fee_rate); sim_current_currency = base_curr;
                }
                else if (trade_type == "SELL") {
                    if (!ticker.contains("bid") || !ticker["bid"].is_string()) throw std::runtime_error("Ticker " + pair + " 缺少有效 bid 价"); price = string_to_double(ticker["bid"].get<std::string>()); if (price <= 0) throw std::runtime_error("Ticker " + pair + " bid 价无效"); net_amount = (sim_current_amount * price) * (1.0 - fee_rate); sim_current_currency = quote_curr;
                }
                else { throw std::runtime_error("未知交易类型: " + trade_type); }
                sim_current_amount = net_amount; if (sim_current_amount < -1e-9) throw std::runtime_error("模拟中金额变为负数"); if (sim_current_amount < 1e-12 && i < trades.size() - 1) throw std::runtime_error("模拟中金额变为零，无法继续");
                if (sim_current_amount < 0) sim_current_amount = 0;
            }
        }

        // 7. 模拟最终闪兑回 USDT
        if (end_with_usdt && sim_current_currency != "USDT") {
            json final_swap_result = simulate_swap_order_cpp(sim_current_currency, "USDT", sim_current_amount, markets, tickers, fee_rate);
            double final_received_amount = get_json_value(final_swap_result, "estimated_to_amount", 0.0); if (final_received_amount > 1e-12) { sim_current_amount = final_received_amount; sim_current_currency = "USDT"; }
        }

        // 8. 计算最终利润
        double profit_amount = std::numeric_limits<double>::quiet_NaN(); double profit_percent = std::numeric_limits<double>::quiet_NaN(); double final_amount = sim_current_amount; std::string final_currency = sim_current_currency; bool verified = false; std::string reason = u8"未达标"; std::string profit_target_currency = end_with_usdt ? "USDT" : actual_start_currency;
        if (final_currency == profit_target_currency) {
            if (actual_start_currency == profit_target_currency) { profit_amount = final_amount - actual_start_amount; if (actual_start_amount > 1e-12) profit_percent = (profit_amount / actual_start_amount) * 100.0; else profit_percent = 0.0; }
            else { profit_amount = final_amount; profit_percent = -998.0; reason = u8"起始(" + actual_start_currency + u8"), 目标(" + profit_target_currency + u8"), 无法计算利润率"; }
            if (std::isfinite(profit_percent) && profit_percent != -998.0) { if (profit_percent > min_profit_req) { verified = true; reason = u8"模拟利润率达标"; } else { reason = u8"模拟利润率未达标"; } }
        }
        else { reason = u8"最终货币与目标不符"; profit_percent = -997.0; }
        result_json["verified"] = verified; result_json["profit_percent"] = profit_percent; result_json["profit_amount"] = std::isfinite(profit_amount) ? profit_amount : 0.0; result_json["final_amount"] = final_amount; result_json["final_currency"] = final_currency; result_json["reason"] = reason;

        // 使用 goto 跳转标签，统一处理输出和错误返回
    write_output_sim_final:; // 定义跳转标签

        // --- catch 块：捕获异常，设置错误信息和状态码 ---
    }
    catch (const json::parse_error& e) {
        std::cerr << "[C++ Exception] JSON 解析错误 (simulate_full_cpp_buffered): " << e.what() << " at byte " << e.byte << std::endl; result_json.clear(); result_json["error"] = "C++ JSON 解析错误: " + std::string(e.what()); return_status = -2;
    }
    catch (const json::type_error& e) {
        std::cerr << "[C++ Exception] JSON 类型错误 (simulate_full_cpp_buffered): " << e.what() << std::endl; result_json.clear(); result_json["error"] = "C++ JSON 类型错误: " + std::string(e.what()); return_status = -2;
    }
    catch (const std::exception& e) {
        std::cerr << "[C++ Exception] 模拟完整路径时发生 C++ 异常: " << e.what() << std::endl; result_json.clear(); result_json["error"] = "C++ 内部错误: " + std::string(e.what()); return_status = -3;
    }
    catch (...) { std::cerr << "[C++ Exception] 模拟完整路径时发生未知 C++ 异常。" << std::endl; result_json.clear(); result_json["error"] = u8"未知的 C++ 内部错误"; return_status = -3; }

    // --- 将最终的 result_json (可能包含错误信息) 写入 output_buffer ---
    std::string result_str;
    try { result_str = result_json.dump(-1, ' ', false, json::error_handler_t::replace); }
    catch (const std::exception& dump_e) {
        std::cerr << "[C++] JSON dump 错误 (simulate_full): " << dump_e.what() << std::endl;
        std::string err_json_str = "{\"error\":\"C++ JSON dump error\"}";
        if (static_cast<size_t>(buffer_size) > err_json_str.length()) { strncpy(output_buffer, err_json_str.c_str(), buffer_size - 1); output_buffer[err_json_str.length()] = '\0'; }
        else { output_buffer[0] = '\0'; }
        return (return_status == 0) ? -2 : return_status; // 如果 dump 出错，优先返回 JSON 错误码
    }
    size_t required_size = result_str.length() + 1;
    if (static_cast<size_t>(buffer_size) < required_size) {
        std::cerr << "[C++] simulate_full_cpp_buffered: 输出缓冲区太小。需要 " << required_size << ", 提供 " << buffer_size << std::endl;
        output_buffer[0] = '\0'; return -1;
    }
#ifdef _WIN32
    errno_t err = strcpy_s(output_buffer, buffer_size, result_str.c_str()); if (err != 0) { std::cerr << "[C++] strcpy_s 错误 (simulate_full): " << err << std::endl; output_buffer[0] = '\0'; return -3; }
#else
    strncpy(output_buffer, result_str.c_str(), buffer_size - 1); output_buffer[buffer_size - 1] = '\0'; if (result_str.length() >= static_cast<size_t>(buffer_size - 1)) { std::cerr << "[C++ Warning] simulate_full_cpp_buffered: 输出可能被截断。" << std::endl; }
#endif

    return return_status;
}

// --- 不再需要 free_memory_ops 函数 ---