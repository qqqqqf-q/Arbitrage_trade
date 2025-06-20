#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <string>
#include <string_view> // 使用 C++17 string_view (如果需要)
#include <unordered_map>
#include <set>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <optional>   // 使用 C++17 optional
#include <utility>    // 使用 std::move
#include <charconv>   // 使用 C++17 from_chars (更安全/高效的字符串转换)

#include "json.hpp" // 假设 nlohmann/json.hpp 路径正确
using json = nlohmann::json;

// 定义导出宏 (保持不变)
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// 用于存储已验证交易对信息的结构体
struct ValidatedTradePair {
    std::string symbol;
    std::string base_currency;
    std::string quote_currency;
    double ask_price;
    double bid_price;
};

// C++17 风格的安全获取 double 的辅助函数
// 返回 std::optional<double>，表示可能成功也可能失败
// 内部尝试使用 std::from_chars (如果可用且适合)，否则回退到 stod
// 注意：std::from_chars 对 locale 不敏感，可能更符合 API 数据格式
std::optional<double> safe_parse_positive_double(const json& j) {
    double value = 0.0;
    bool success = false;

    if (j.is_number()) {
        value = j.get<double>();
        success = true;
    }
    else if (j.is_string()) {
        const std::string str = j.get<std::string>();
        // C++17 std::from_chars 提供更安全、无异常、高性能的转换
        const char* start = str.data();
        const char* end = start + str.size();
        auto result = std::from_chars(start, end, value);
        // 检查转换是否成功，并且是否消耗了整个字符串
        success = (result.ec == std::errc{} && result.ptr == end);

        // 如果 from_chars 失败或不可用，可以回退到 stod (但 stod 可能抛异常)
        // if (!success) {
        //     try {
        //         size_t processed_chars = 0;
        //         value = std::stod(str, &processed_chars);
        //         // 确保整个字符串都被转换了
        //         success = (processed_chars == str.length());
        //     } catch (...) {
        //         success = false; // 转换失败
        //     }
        // }
    }

    if (success && value > std::numeric_limits<double>::epsilon()) { // 检查是否为正数 (使用 epsilon 避免浮点精度问题)
        return value;
    }
    return std::nullopt; // 无效或非正数
}


extern "C" DLLEXPORT char* build_graph_cpp(
    const char* tickers_json_str,
    const char* markets_json_str,
    double taker_fee_rate)
{
    try {
        // 1. 解析输入 JSON
        json tickers_json = json::parse(tickers_json_str);
        json markets_json = json::parse(markets_json_str);

        if (!tickers_json.is_object() || !markets_json.is_object()) {
            std::cerr << "[C++] 错误: 输入的 JSON 不是对象。" << std::endl;
            // 可以考虑返回一个包含错误信息的 JSON 字符串，而不是 nullptr
            // 例如: return strdup("{\"error\": \"输入 JSON 不是对象\"}");
            return nullptr;
        }

        // 2. 单次遍历：验证数据、提取信息、收集货币
        std::set<std::string> unique_currencies;
        std::vector<ValidatedTradePair> validated_pairs;
        // 预估大小并预留空间，减少内存重分配
        validated_pairs.reserve(tickers_json.size());

        for (auto const& [symbol, ticker] : tickers_json.items()) { // C++17 结构化绑定遍历 map
            // --- 基本 Ticker 验证 ---
            if (!ticker.is_object() || !ticker.contains("ask") || !ticker.contains("bid")) {
                continue; // 跳过无效 ticker
            }

            // --- Market 验证 ---
            auto market_it = markets_json.find(symbol);
            if (market_it == markets_json.end()) {
                continue; // 找不到对应的 market 信息
            }
            const json& market = market_it.value();

            // 验证 market 状态和必要字段
            if (!market.is_object() || !market.value("active", false) || !market.value("spot", false) ||
                !market.contains("base") || !market.contains("quote") ||
                !market["base"].is_string() || !market["quote"].is_string()) { // 确保 base/quote 是字符串
                continue; // 跳过无效 market
            }

            // 使用 get_ref 获取引用，避免不必要的 string 拷贝
            const std::string& base_currency = market["base"].get_ref<const std::string&>();
            const std::string& quote_currency = market["quote"].get_ref<const std::string&>();

            // 检查货币名称是否为空
            if (base_currency.empty() || quote_currency.empty()) {
                continue;
            }

            // --- 价格验证 (使用 C++17 optional 和安全解析函数) ---
            std::optional<double> ask_price_opt = safe_parse_positive_double(ticker["ask"]);
            std::optional<double> bid_price_opt = safe_parse_positive_double(ticker["bid"]);

            // 只有当 ask 和 bid 价格都有效时才继续
            if (!ask_price_opt || !bid_price_opt) {
                continue;
            }

            // --- 存储已验证数据 ---
            unique_currencies.insert(base_currency);
            unique_currencies.insert(quote_currency);
            // 使用 emplace_back 直接在 vector 中构造 ValidatedTradePair 对象
            validated_pairs.emplace_back(ValidatedTradePair{
                symbol, base_currency, quote_currency, *ask_price_opt, *bid_price_opt
                });
        }

        // 检查是否有有效的数据生成
        if (unique_currencies.empty() || validated_pairs.empty()) {
            std::cerr << "[C++] 错误: 过滤后没有找到有效的货币或交易对。" << std::endl;
            return nullptr;
        }

        // 3. 创建货币到索引的映射
        std::vector<std::string> index_to_currency_cpp(unique_currencies.begin(), unique_currencies.end());
        std::sort(index_to_currency_cpp.begin(), index_to_currency_cpp.end()); // 保持确定性排序

        std::unordered_map<std::string, int> currency_to_index_cpp;
        currency_to_index_cpp.reserve(index_to_currency_cpp.size()); // 预留空间
        for (int i = 0; i < index_to_currency_cpp.size(); ++i) {
            currency_to_index_cpp[index_to_currency_cpp[i]] = i;
        }
        int num_currencies = static_cast<int>(index_to_currency_cpp.size()); // 显式转换

        // 4. 构建边列表 (基于缓存的已验证数据)
        std::vector<json> edges_json_array;
        edges_json_array.reserve(validated_pairs.size() * 2); // 预估每对产生两条边
        const double fee_multiplier = 1.0 - taker_fee_rate;
        const double log_epsilon = 1e-15; // 用于比较费率是否大于零的小正数

        for (const auto& pair : validated_pairs) {
            // 从 map 中获取索引 (此时 base/quote 肯定存在于 map 中)
            int base_idx = currency_to_index_cpp[pair.base_currency];
            int quote_idx = currency_to_index_cpp[pair.quote_currency];

            // a) 边：Quote -> Base (买入 Base, 支付 Quote) - 使用 Ask Price
            // 汇率 = (1.0 / ask_price) * fee_multiplier (表示 1 单位 Quote 能买多少 Base)
            double net_rate_q_to_b = (1.0 / pair.ask_price) * fee_multiplier;
            if (net_rate_q_to_b > log_epsilon) { // 确保汇率有效且大于 0
                double weight = -std::log(net_rate_q_to_b);
                if (std::isfinite(weight)) { // 检查 log 计算结果是否有效
                    // 使用 emplace_back 和列表初始化构造 json 对象
                    edges_json_array.emplace_back(json{
                        {"from", quote_idx},
                        {"to", base_idx},
                        {"weight", weight},
                        {"pair", pair.symbol},
                        {"type", "BUY"} // 买入基础货币 (Base)
                        });
                }
            }

            // b) 边：Base -> Quote (卖出 Base, 收到 Quote) - 使用 Bid Price
            // 汇率 = bid_price * fee_multiplier (表示 1 单位 Base 能卖多少 Quote)
            double net_rate_b_to_q = pair.bid_price * fee_multiplier;
            if (net_rate_b_to_q > log_epsilon) { // 确保汇率有效且大于 0
                double weight = -std::log(net_rate_b_to_q);
                if (std::isfinite(weight)) {
                    edges_json_array.emplace_back(json{
                        {"from", base_idx},
                        {"to", quote_idx},
                        {"weight", weight},
                        {"pair", pair.symbol},
                        {"type", "SELL"} // 卖出基础货币 (Base)
                        });
                }
            }
        }

        if (edges_json_array.empty()) {
            std::cerr << "[C++] 错误: 未能生成任何有效的图边。" << std::endl;
            return nullptr;
        }

        // 5. 构建最终输出 JSON
        json output_json;
        output_json["nodes"] = index_to_currency_cpp;
        // 使用 std::move 将 vector 的内容移动到 JSON 对象中，避免拷贝
        output_json["edges"] = std::move(edges_json_array);

        // 6. 序列化 JSON 并返回 C 风格字符串
        std::string output_str = output_json.dump();
        // C 接口要求我们分配内存，调用者负责释放
        char* result_c_str = new char[output_str.length() + 1];
        std::strcpy(result_c_str, output_str.c_str());
        return result_c_str;

    }
    catch (const json::parse_error& e) {
        std::cerr << "[C++] JSON 解析错误 (build_graph_cpp): " << e.what() << std::endl;
        return nullptr;
    }
    catch (const std::exception& e) {
        std::cerr << "[C++] 标准异常 (build_graph_cpp): " << e.what() << std::endl;
        return nullptr;
    }
    catch (...) {
        std::cerr << "[C++] 未知错误发生在 build_graph_cpp。" << std::endl;
        return nullptr;
    }
}

// --- 内存释放函数 (保持不变) ---
extern "C" DLLEXPORT void free_memory(char* ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
    }
}