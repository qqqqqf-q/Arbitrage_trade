// Define this macro BEFORE any includes to silence the C++17 std::iterator deprecation warning
// This is often necessary when third-party libraries trigger it within standard library headers.
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING 1

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <string>
#include <string_view> // C++17
#include <set>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <optional>      // C++17
#include <utility>       // std::move
#include <charconv>      // C++17
#include <thread>        // C++11/17 多线程
#include <mutex>         // C++11/17 互斥锁
#include <future>        // C++11/17 用于获取线程结果
#include <atomic>
#include <iterator>      // std::distance, std::make_move_iterator

// --- RapidJSON Headers ---
// 确保这些头文件在你的包含路径中
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h" // Include for GetParseError_En()
// --- End RapidJSON Headers ---


// 定义导出宏 (保持不变)
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// 用于存储已验证交易对信息的结构体 (保持不变)
struct ValidatedTradePair {
    std::string symbol;
    std::string base_currency;
    std::string quote_currency;
    double ask_price;
    double bid_price;
};

// 线程处理结果的结构体 (保持不变)
struct ThreadResult {
    std::set<std::string> local_currencies;
    std::vector<ValidatedTradePair> local_validated_pairs;
};

// C++17 风格的安全获取 double 的辅助函数 (适配 RapidJSON) (保持不变)
std::optional<double> safe_parse_positive_double(const rapidjson::Value& j) {
    double value = 0.0;
    bool success = false;

    if (j.IsNumber()) {
        if (j.IsDouble()) value = j.GetDouble();
        else if (j.IsInt()) value = static_cast<double>(j.GetInt());
        else if (j.IsUint()) value = static_cast<double>(j.GetUint());
        else if (j.IsInt64()) value = static_cast<double>(j.GetInt64());
        else if (j.IsUint64()) value = static_cast<double>(j.GetUint64());
        else return std::nullopt;
        success = true;
    }
    else if (j.IsString()) {
        std::string_view sv = j.GetString();
        const char* start = sv.data();
        const char* end = start + sv.size();
        auto result = std::from_chars(start, end, value);
        success = (result.ec == std::errc{} && result.ptr == end);
    }

    if (success && value > std::numeric_limits<double>::epsilon()) {
        return value;
    }
    return std::nullopt;
}


// --- 线程工作函数 --- (保持不变)
ThreadResult process_ticker_chunk(
    rapidjson::Value::ConstMemberIterator chunk_begin,
    rapidjson::Value::ConstMemberIterator chunk_end,
    const rapidjson::Document& markets_doc,
    double taker_fee_rate
) {
    ThreadResult result;
    result.local_validated_pairs.reserve(static_cast<size_t>(std::distance(chunk_begin, chunk_end)));

    for (auto it = chunk_begin; it != chunk_end; ++it) {
        const std::string symbol = it->name.GetString();
        const rapidjson::Value& ticker = it->value;

        if (!ticker.IsObject() || !ticker.HasMember("ask") || !ticker.HasMember("bid")) {
            continue;
        }

        auto market_it = markets_doc.FindMember(symbol.c_str());
        if (market_it == markets_doc.MemberEnd()) {
            continue;
        }
        const rapidjson::Value& market = market_it->value;

        if (!market.IsObject() ||
            !market.HasMember("active") || !market["active"].IsBool() || !market["active"].GetBool() ||
            !market.HasMember("spot") || !market["spot"].IsBool() || !market["spot"].GetBool() ||
            !market.HasMember("base") || !market["base"].IsString() ||
            !market.HasMember("quote") || !market["quote"].IsString()) {
            continue;
        }

        std::string base_currency = market["base"].GetString();
        std::string quote_currency = market["quote"].GetString();

        if (base_currency.empty() || quote_currency.empty()) {
            continue;
        }

        std::optional<double> ask_price_opt = safe_parse_positive_double(ticker["ask"]);
        std::optional<double> bid_price_opt = safe_parse_positive_double(ticker["bid"]);

        if (!ask_price_opt || !bid_price_opt) {
            continue;
        }

        result.local_currencies.insert(base_currency);
        result.local_currencies.insert(quote_currency);
        result.local_validated_pairs.emplace_back(ValidatedTradePair{
            symbol,
            std::move(base_currency),
            std::move(quote_currency),
            *ask_price_opt,
            *bid_price_opt
            });
    }
    return result;
}


// --- 主函数 --- (保持不变)
extern "C" DLLEXPORT char* build_graph_cpp(
    const char* tickers_json_str,
    const char* markets_json_str,
    double taker_fee_rate)
{
    try {
        // 1. 使用 RapidJSON 解析输入
        rapidjson::Document tickers_doc;
        tickers_doc.Parse(tickers_json_str);
        if (tickers_doc.HasParseError()) {
            std::cerr << "[C++] RapidJSON tickers 解析错误: " << rapidjson::GetParseError_En(tickers_doc.GetParseError())
                << " at offset " << tickers_doc.GetErrorOffset() << std::endl;
            return nullptr;
        }

        rapidjson::Document markets_doc;
        markets_doc.Parse(markets_json_str);
        if (markets_doc.HasParseError()) {
            std::cerr << "[C++] RapidJSON markets 解析错误: " << rapidjson::GetParseError_En(markets_doc.GetParseError())
                << " at offset " << markets_doc.GetErrorOffset() << std::endl;
            return nullptr;
        }

        if (!tickers_doc.IsObject() || !markets_doc.IsObject()) {
            std::cerr << "[C++] 错误: 输入的 JSON (RapidJSON) 不是对象。" << std::endl;
            return nullptr;
        }

        // 2. 准备多线程处理
        const unsigned int num_threads = std::thread::hardware_concurrency();
        const size_t total_tickers = tickers_doc.MemberCount();
        if (total_tickers == 0) {
            std::cerr << "[C++] 错误: tickers JSON 为空或没有成员。" << std::endl;
            return nullptr;
        }

        size_t chunk_size = (total_tickers + num_threads - 1) / num_threads;
        std::vector<std::future<ThreadResult>> futures;
        futures.reserve(num_threads);
        auto current_ticker_it = tickers_doc.MemberBegin();

        // 3. 启动线程
        for (unsigned int i = 0; i < num_threads && current_ticker_it != tickers_doc.MemberEnd(); ++i) {
            auto chunk_start = current_ticker_it;
            auto chunk_end = chunk_start;
            size_t count = 0;
            while (chunk_end != tickers_doc.MemberEnd() && count < chunk_size) {
                ++chunk_end;
                ++count;
            }
            current_ticker_it = chunk_end;

            futures.emplace_back(std::async(
                std::launch::async,
                process_ticker_chunk,
                chunk_start,
                chunk_end,
                std::cref(markets_doc),
                taker_fee_rate
            ));
        }

        // 4. 合并线程结果
        std::set<std::string> unique_currencies;
        std::vector<ValidatedTradePair> validated_pairs;
        validated_pairs.reserve(total_tickers);

        for (auto& fut : futures) {
            try {
                ThreadResult result = fut.get();
                for (const auto& currency : result.local_currencies) {
                    unique_currencies.insert(currency);
                }
                validated_pairs.insert(
                    validated_pairs.end(),
                    std::make_move_iterator(result.local_validated_pairs.begin()),
                    std::make_move_iterator(result.local_validated_pairs.end())
                );
            }
            catch (const std::exception& e) {
                std::cerr << "[C++] 线程执行异常: " << e.what() << std::endl;
                // return nullptr;
            }
            catch (...) {
                std::cerr << "[C++] 线程发生未知异常。" << std::endl;
                // return nullptr;
            }
        }

        if (unique_currencies.empty() || validated_pairs.empty()) {
            std::cerr << "[C++] 错误: 过滤后没有找到有效的货币或交易对 (多线程)。" << std::endl;
            return nullptr;
        }

        // 5. 创建货币到索引的映射
        std::vector<std::string> index_to_currency_cpp(unique_currencies.begin(), unique_currencies.end());
        std::sort(index_to_currency_cpp.begin(), index_to_currency_cpp.end());

        std::unordered_map<std::string, int> currency_to_index_cpp;
        currency_to_index_cpp.reserve(static_cast<size_t>(index_to_currency_cpp.size()));
        for (int i = 0; i < index_to_currency_cpp.size(); ++i) {
            currency_to_index_cpp[index_to_currency_cpp[i]] = i;
        }

        // 6. 构建边列表 (单线程, 使用 RapidJSON Document 构建)
        rapidjson::Document output_doc;
        output_doc.SetObject();
        rapidjson::Document::AllocatorType& allocator = output_doc.GetAllocator();

        rapidjson::Value edges_array(rapidjson::kArrayType);
        edges_array.Reserve(static_cast<rapidjson::SizeType>(validated_pairs.size() * 2), allocator);

        const double fee_multiplier = 1.0 - taker_fee_rate;
        const double log_epsilon = 1e-15;

        for (const auto& pair : validated_pairs) {
            int base_idx = currency_to_index_cpp[pair.base_currency];
            int quote_idx = currency_to_index_cpp[pair.quote_currency];

            // a) 边：Quote -> Base (BUY)
            double net_rate_q_to_b = (1.0 / pair.ask_price) * fee_multiplier;
            if (net_rate_q_to_b > log_epsilon) {
                double weight = -std::log(net_rate_q_to_b);
                if (std::isfinite(weight)) {
                    rapidjson::Value edge_obj(rapidjson::kObjectType);
                    edge_obj.AddMember("from", quote_idx, allocator);
                    edge_obj.AddMember("to", base_idx, allocator);
                    edge_obj.AddMember("weight", weight, allocator);
                    rapidjson::Value pair_val(pair.symbol.c_str(), static_cast<rapidjson::SizeType>(pair.symbol.length()), allocator);
                    edge_obj.AddMember("pair", pair_val, allocator);
                    edge_obj.AddMember("type", "BUY", allocator);
                    edges_array.PushBack(edge_obj, allocator);
                }
            }

            // b) 边：Base -> Quote (SELL)
            double net_rate_b_to_q = pair.bid_price * fee_multiplier;
            if (net_rate_b_to_q > log_epsilon) {
                double weight = -std::log(net_rate_b_to_q);
                if (std::isfinite(weight)) {
                    rapidjson::Value edge_obj(rapidjson::kObjectType);
                    edge_obj.AddMember("from", base_idx, allocator);
                    edge_obj.AddMember("to", quote_idx, allocator);
                    edge_obj.AddMember("weight", weight, allocator);
                    rapidjson::Value pair_val(pair.symbol.c_str(), static_cast<rapidjson::SizeType>(pair.symbol.length()), allocator);
                    edge_obj.AddMember("pair", pair_val, allocator);
                    edge_obj.AddMember("type", "SELL", allocator);
                    edges_array.PushBack(edge_obj, allocator);
                }
            }
        }

        if (edges_array.Empty()) {
            std::cerr << "[C++] 错误: 未能生成任何有效的图边 (RapidJSON)。" << std::endl;
            return nullptr;
        }

        // 7. 构建最终输出 JSON (RapidJSON)
        rapidjson::Value nodes_array(rapidjson::kArrayType);
        nodes_array.Reserve(static_cast<rapidjson::SizeType>(index_to_currency_cpp.size()), allocator);
        for (const auto& currency : index_to_currency_cpp) {
            rapidjson::Value currency_val(currency.c_str(), static_cast<rapidjson::SizeType>(currency.length()), allocator);
            nodes_array.PushBack(currency_val, allocator);
        }
        output_doc.AddMember("nodes", nodes_array, allocator);
        output_doc.AddMember("edges", edges_array, allocator);

        // 8. 序列化 JSON (使用 RapidJSON)
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        output_doc.Accept(writer);

        // 9. 返回 C 风格字符串
        const char* output_str = buffer.GetString();
        size_t output_len = buffer.GetSize();
        char* result_c_str = new char[output_len + 1];
        std::memcpy(result_c_str, output_str, output_len);
        result_c_str[output_len] = '\0';
        return result_c_str;

    }
    catch (const std::bad_alloc& e) {
        std::cerr << "[C++] 内存分配错误: " << e.what() << std::endl;
        return nullptr;
    }
    catch (const std::future_error& e) {
        std::cerr << "[C++] 线程 future 错误: " << e.what() << " (Code: " << e.code() << ")" << std::endl;
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