#define _CRT_SECURE_NO_WARNINGS // ���� strcpy �Ⱥ����İ�ȫ����
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

// --- ���� nlohmann/json.hpp ---
// ȷ�����ļ�����İ���·���У������ṩ��ȷ·��
#include "json.hpp" // ���� #include <nlohmann/json.hpp>
// ʹ�� nlohmann::json �������ռ����
using json = nlohmann::json;

// --- Bellman-Ford ��ؽṹ�ͺ��� (���ֲ���) ---
struct Edge {
    int from_node;
    int to_node;
    double weight;
    std::string pair_symbol; // ʹ�� std::string
    std::string trade_type;  // ʹ�� std::string
};

// ���� find_negative_cycles ���������ﶨ����ڱ𴦶��岢����

// --- ����ͼ�� C++ ���� ---

// ���嵼����
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
        // 1. ��������� JSON �ַ���
        json tickers_json = json::parse(tickers_json_str);
        json markets_json = json::parse(markets_json_str);

        if (!tickers_json.is_object() || !markets_json.is_object()) {
            std::cerr << "[C++] Error: Input JSON is not an object." << std::endl;
            return nullptr;
        }

        // 2. Ԥ������ռ�����
        std::set<std::string> unique_currencies;
        std::vector<std::string> valid_symbols_for_edges;

        // --- �޸�: ʹ�ô�ͳ���������� tickers_json ---
        for (auto it = tickers_json.begin(); it != tickers_json.end(); ++it) {
            const std::string& symbol = it.key(); // ��ȡ�� (symbol)
            const json& ticker = it.value();    // ��ȡֵ (ticker ����)
            // --- �����޸� ---

                // ��� ticker �Ƿ���Ч�Ұ�����Ҫ�ֶ�
            if (!ticker.is_object() || !ticker.contains("ask") || !ticker.contains("bid")) {
                continue;
            }
            // ��� market �Ƿ��������Ч
            if (!markets_json.contains(symbol)) {
                continue;
            }
            const auto& market = markets_json[symbol];
            if (!market.is_object() || !market.value("active", false) || !market.value("spot", false) ||
                !market.contains("base") || !market.contains("quote")) {
                continue;
            }

            // ���۸���Ч�� (������������ > 0)
            double ask_price = 0.0, bid_price = 0.0;
            bool ask_valid = false, bid_valid = false;

            // --- �޸�: �ֿ���ȡ�ַ�����ת�� ---
            if (ticker["ask"].is_number()) {
                ask_price = ticker["ask"].get<double>();
                ask_valid = ask_price > 0;
            }
            else if (ticker["ask"].is_string()) {
                try {
                    std::string ask_str = ticker["ask"].get<std::string>(); // �Ȼ�ȡ
                    ask_price = std::stod(ask_str);                         // ��ת��
                    ask_valid = ask_price > 0;
                }
                catch (...) { /* ת��ʧ�� */ }
            }

            if (ticker["bid"].is_number()) {
                bid_price = ticker["bid"].get<double>();
                bid_valid = bid_price > 0;
            }
            else if (ticker["bid"].is_string()) {
                try {
                    std::string bid_str = ticker["bid"].get<std::string>(); // �Ȼ�ȡ
                    bid_price = std::stod(bid_str);                         // ��ת��
                    bid_valid = bid_price > 0;
                }
                catch (...) { /* ת��ʧ�� */ }
            }
            // --- �����޸� ---

            if (!ask_valid || !bid_valid) {
                continue;
            }

            // ������м��ͨ������ӻ��Ҳ���¼��Ч����
            // --- �޸�: ȷ�� market["base"] �� market["quote"] ����ȷ��ȡ�ַ��� ---
            const std::string& base_currency = market["base"].get_ref<const std::string&>();
            const std::string& quote_currency = market["quote"].get_ref<const std::string&>();
            // --- �����޸� ---
            unique_currencies.insert(base_currency);
            unique_currencies.insert(quote_currency);
            valid_symbols_for_edges.push_back(symbol);
        }

        if (unique_currencies.empty() || valid_symbols_for_edges.empty()) {
            std::cerr << "[C++] Error: No valid currencies or symbols found after filtering." << std::endl;
            return nullptr;
        }

        // 3. ������������ӳ�� (���ֲ���)
        std::vector<std::string> index_to_currency_cpp(unique_currencies.begin(), unique_currencies.end());
        std::sort(index_to_currency_cpp.begin(), index_to_currency_cpp.end()); // �����Ի��ȷ����

        std::unordered_map<std::string, int> currency_to_index_cpp;
        for (int i = 0; i < index_to_currency_cpp.size(); ++i) {
            currency_to_index_cpp[index_to_currency_cpp[i]] = i;
        }
        int num_currencies = index_to_currency_cpp.size();

        // 4. �������б� (ʹ�� double ����Ȩ��)
        std::vector<json> edges_json_array;
        double fee_multiplier = 1.0 - taker_fee_rate;

        for (const std::string& symbol : valid_symbols_for_edges) {
            const auto& ticker = tickers_json[symbol];
            const auto& market = markets_json[symbol];
            // --- �޸�: ȷ�� market["base"] �� market["quote"] ����ȷ��ȡ�ַ��� ---
            const std::string& base = market["base"].get_ref<const std::string&>();
            const std::string& quote = market["quote"].get_ref<const std::string&>();
            // --- �����޸� ---

            // ��ȡ�Ѿ���֤���ļ۸� (ʹ���޸ĺ�İ�ȫת���߼�)
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

            // a) �ߣ�Quote -> Base (���� Base, ���� Quote) - ʹ�� Ask Price
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

            // b) �ߣ�Base -> Quote (���� Base, �յ� Quote) - ʹ�� Bid Price
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

        // 5. �������յ���� JSON (���ֲ���)
        json output_json;
        output_json["nodes"] = index_to_currency_cpp; // ["BTC", "ETH", "USDT", ...]
        output_json["edges"] = edges_json_array;      // [{"from": 0, "to": 1, ...}, ...]

        // 6. ���л� JSON �������ڴ淵�ظ� Python (���ֲ���)
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


// --- �ڴ��ͷź��� (���ֲ���) ---
extern "C" DLLEXPORT void free_memory(char* ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
    }
}

// --- �����Ҫ����� find_negative_cycles �Ķ���� extern ���� ---
// extern "C" DLLEXPORT int find_negative_cycles(...) { ... }