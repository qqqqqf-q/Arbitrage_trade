#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>        // For std::log (if needed, though not directly here) and std::abs
#include <limits>       // For numeric_limits
#include <stdexcept>
#include <sstream>
#include <iomanip>      // For setprecision
#include <vector>
#include <numeric>      // For std::accumulate (optional)
#include <algorithm>    // For std::max
#include"pch.h"
// --- Include nlohmann/json ---
// Make sure json.hpp is accessible in your include path or same directory
#include "json.hpp"

using json = nlohmann::json;

// Define platform-specific export macro
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

// --- Helper Structures ---

struct Trade {
    std::string from;
    std::string to;
    std::string pair;
    std::string type; // "BUY" or "SELL"
};

struct TickerInfo {
    double ask = 0.0;
    double bid = 0.0;
};

struct MarketInfo {
    std::string base;
    std::string quote;
};

struct Config {
    double taker_fee_rate = 0.001; // Default, will be overwritten
    double min_profit_percent_verify = 0.0; // Default
};

struct SwapResult {
    double estimated_to_amount = 0.0;
    std::vector<std::string> steps;
    bool success = false;
    std::string error_reason;
};

// --- Helper Function: Format double to string (like Python's format_decimal) ---
// Note: This provides basic formatting, not Decimal's exact rounding behavior.
std::string format_double(double d, int precision = 8) {
    if (!std::isfinite(d)) {
        return std::to_string(d); // Handles NaN, Infinity
    }
    std::stringstream ss;
    // Use fixed and setprecision for consistent decimal places
    ss << std::fixed << std::setprecision(precision) << d;
    std::string s = ss.str();
    // Optional: Remove trailing zeros and decimal point if possible (like Python's rstrip)
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (s.back() == '.') {
        s.pop_back();
    }
    return s;
}

// --- C++ Implementation of simulate_swap_order Logic ---
// This function is internal to the C++ simulation
SwapResult simulate_swap_order_cpp(
    const std::string& from_currency,
    const std::string& to_currency,
    double from_amount,
    const std::unordered_map<std::string, TickerInfo>& tickers,
    const std::unordered_map<std::string, MarketInfo>& markets,
    double fee_rate)
{
    SwapResult result;
    result.success = false; // Assume failure initially
    std::string intermediate_currency = "USDT"; // Common intermediate

    if (from_currency == to_currency) {
        result.success = true;
        result.estimated_to_amount = from_amount;
        result.steps.push_back("Source and target currency are the same.");
        return result;
    }
    if (from_amount <= 0) {
        result.error_reason = "Swap amount must be positive.";
        return result; // Keep success as false
    }


    std::string direct_symbol_forward = to_currency + "/" + from_currency;  // e.g., BNB/ETH (Buy BNB using ETH)
    std::string direct_symbol_backward = from_currency + "/" + to_currency; // e.g., ETH/BNB (Sell ETH for BNB)
    std::string symbol_from_intermediate = from_currency + "/" + intermediate_currency; // e.g., ETH/USDT
    std::string symbol_to_intermediate = to_currency + "/" + intermediate_currency;   // e.g., BNB/USDT

    // --- Case 1: Direct Backward (Sell from_currency for to_currency) ---
    if (markets.count(direct_symbol_backward) && tickers.count(direct_symbol_backward)) {
        const auto& ticker = tickers.at(direct_symbol_backward);
        if (ticker.bid > 0) {
            double bid_price = ticker.bid;
            double estimated_quote_received_gross = from_amount * bid_price;
            double fee = estimated_quote_received_gross * fee_rate;
            result.estimated_to_amount = estimated_quote_received_gross - fee;
            if (result.estimated_to_amount > 0) {
                result.success = true;
                result.steps.push_back("1. Market Sell " + format_double(from_amount) + " " + from_currency +
                    " on " + direct_symbol_backward + " @ ~" + format_double(bid_price) +
                    " -> Estimated receive " + format_double(result.estimated_to_amount) + " " + to_currency + " (Fee adjusted)");
                return result; // Found direct path
            }
            else {
                // Calculation resulted in non-positive amount, treat as failure for this path
                result.estimated_to_amount = 0; // Reset amount
            }
        }
    }

    // --- Case 2: Direct Forward (Buy to_currency using from_currency) ---
    if (markets.count(direct_symbol_forward) && tickers.count(direct_symbol_forward)) {
        const auto& ticker = tickers.at(direct_symbol_forward);
        if (ticker.ask > 0) {
            double ask_price = ticker.ask;
            double estimated_base_received_gross = from_amount / ask_price;
            double fee = estimated_base_received_gross * fee_rate;
            result.estimated_to_amount = estimated_base_received_gross - fee;
            if (result.estimated_to_amount > 0) {
                result.success = true;
                result.steps.push_back("1. Market Buy " + to_currency + " using " + format_double(from_amount) + " " + from_currency +
                    " on " + direct_symbol_forward + " @ ~" + format_double(ask_price) +
                    " -> Estimated receive " + format_double(result.estimated_to_amount) + " " + to_currency + " (Fee adjusted)");
                return result; // Found direct path
            }
            else {
                result.estimated_to_amount = 0;
            }
        }
    }

    // --- Case 3: Through Intermediate Currency (USDT) ---
    bool can_sell_from = markets.count(symbol_from_intermediate) && tickers.count(symbol_from_intermediate);
    bool can_buy_to = markets.count(symbol_to_intermediate) && tickers.count(symbol_to_intermediate);

    if (can_sell_from && can_buy_to) {
        const auto& ticker_from = tickers.at(symbol_from_intermediate);
        const auto& ticker_to = tickers.at(symbol_to_intermediate);

        if (ticker_from.bid > 0 && ticker_to.ask > 0) {
            // Step A: Sell from_currency for intermediate_currency
            double bid_price_from = ticker_from.bid;
            double intermediate_amount_gross = from_amount * bid_price_from;
            double fee1 = intermediate_amount_gross * fee_rate;
            double intermediate_amount_net = intermediate_amount_gross - fee1;

            if (intermediate_amount_net > 0) {
                result.steps.push_back("1. (Intermediate) Market Sell " + format_double(from_amount) + " " + from_currency +
                    " on " + symbol_from_intermediate + " @ ~" + format_double(bid_price_from) +
                    " -> Estimated receive " + format_double(intermediate_amount_net) + " " + intermediate_currency);

                // Step B: Buy to_currency using intermediate_currency
                double ask_price_to = ticker_to.ask;
                double estimated_to_amount_gross = intermediate_amount_net / ask_price_to;
                double fee2 = estimated_to_amount_gross * fee_rate;
                result.estimated_to_amount = estimated_to_amount_gross - fee2;

                if (result.estimated_to_amount > 0) {
                    result.success = true;
                    result.steps.push_back("2. (Intermediate) Market Buy " + to_currency + " using " + format_double(intermediate_amount_net) + " " + intermediate_currency +
                        " on " + symbol_to_intermediate + " @ ~" + format_double(ask_price_to) +
                        " -> Estimated receive " + format_double(result.estimated_to_amount) + " " + to_currency + " (Fee adjusted)");
                    return result; // Found intermediate path
                }
                else {
                    result.estimated_to_amount = 0; // Reset amount
                    result.steps.pop_back(); // Remove the log for step A as step B failed
                }
            }
        }
    }

    // If we reach here, no path was found or successful
    if (!result.success) {
        result.error_reason = "Could not find a valid simulation path for swap: " + from_currency + " -> " + to_currency;
    }
    return result;
}


// --- Main Exported C++ Simulation Function ---
extern "C" { // Use C linkage for compatibility

    /**
     * @brief Simulates a full arbitrage execution path, including potential swaps.
     *
     * @param json_input_str A JSON string containing simulation parameters:
     *   {
     *     "cycle_info": {
     *       "nodes": ["CUR1", "CUR2", ...],
     *       "trades": [ {"from": "C1", "to": "C2", "pair": "C2/C1", "type": "BUY"}, ... ]
     *     },
     *     "actual_start_currency": "USDT",
     *     "actual_start_amount": 100.0,
     *     "end_with_usdt": true,
     *     "current_tickers": { "SYM1": {"ask": 1.0, "bid": 0.9}, ... },
     *     "markets": { "SYM1": {"base": "B", "quote": "Q"}, ... },
     *     "config": { "taker_fee_rate": 0.001, "min_profit_percent_verify": 0.1 }
     *   }
     * @param json_output_str_ptr Pointer to a char pointer. The function will allocate memory
     *                            for the result JSON string and store its address here.
     *                            The caller MUST free this memory using free_simulation_memory.
     * @return int 0 on success, negative value on error (e.g., -1 for JSON parse error, -2 for simulation error).
     */
    DLLEXPORT int simulate_full_execution_profit_cpp(const char* json_input_str, char** json_output_str_ptr) {
        json result_json;
        std::vector<std::string> sim_logs;
        bool verified = false;
        double profit_percent = -999.0;
        double profit_amount = -999.0;
        double final_amount = 0.0;
        std::string final_currency = "";
        std::string reason = "Simulation did not complete successfully.";

        try {
            // 1. Parse Input JSON
            json input_data = json::parse(json_input_str);

            // 2. Extract Data (with basic validation)
            if (!input_data.contains("cycle_info") || !input_data["cycle_info"].is_object() ||
                !input_data.contains("actual_start_currency") || !input_data["actual_start_currency"].is_string() ||
                !input_data.contains("actual_start_amount") || !input_data["actual_start_amount"].is_number() ||
                !input_data.contains("end_with_usdt") || !input_data["end_with_usdt"].is_boolean() ||
                !input_data.contains("current_tickers") || !input_data["current_tickers"].is_object() ||
                !input_data.contains("markets") || !input_data["markets"].is_object() ||
                !input_data.contains("config") || !input_data["config"].is_object())
            {
                throw std::runtime_error("Input JSON missing required fields or has incorrect types.");
            }

            const json& cycle_info_json = input_data["cycle_info"];
            std::string actual_start_currency = input_data["actual_start_currency"];
            double actual_start_amount = input_data["actual_start_amount"];
            bool end_with_usdt = input_data["end_with_usdt"];
            const json& tickers_json = input_data["current_tickers"];
            const json& markets_json = input_data["markets"];
            const json& config_json = input_data["config"];

            // Extract cycle trades
            std::vector<Trade> trades;
            if (!cycle_info_json.contains("trades") || !cycle_info_json["trades"].is_array()) {
                throw std::runtime_error("cycle_info missing 'trades' array.");
            }
            for (const auto& trade_json : cycle_info_json["trades"]) {
                if (!trade_json.is_object() || !trade_json.contains("from") || !trade_json.contains("to") ||
                    !trade_json.contains("pair") || !trade_json.contains("type")) {
                    throw std::runtime_error("Invalid trade object in cycle_info.");
                }
                trades.push_back({
                    trade_json["from"], trade_json["to"],
                    trade_json["pair"], trade_json["type"]
                    });
            }
            if (!cycle_info_json.contains("nodes") || !cycle_info_json["nodes"].is_array() || cycle_info_json["nodes"].empty()) {
                throw std::runtime_error("cycle_info missing 'nodes' array or is empty.");
            }
            std::string cycle_start_currency = cycle_info_json["nodes"][0];


            // Extract config
            Config config;
            if (config_json.contains("taker_fee_rate") && config_json["taker_fee_rate"].is_number()) {
                config.taker_fee_rate = config_json["taker_fee_rate"];
            }
            if (config_json.contains("min_profit_percent_verify") && config_json["min_profit_percent_verify"].is_number()) {
                config.min_profit_percent_verify = config_json["min_profit_percent_verify"];
            }
            // Allow overriding with min_profit_full_sim_percent if present
            if (config_json.contains("min_profit_full_sim_percent") && config_json["min_profit_full_sim_percent"].is_number()) {
                config.min_profit_percent_verify = config_json["min_profit_full_sim_percent"];
            }


            // Extract tickers
            std::unordered_map<std::string, TickerInfo> tickers;
            for (auto it = tickers_json.begin(); it != tickers_json.end(); ++it) {
                const auto& ticker_data = it.value();
                if (!ticker_data.is_object() || !ticker_data.contains("ask") || !ticker_data.contains("bid") ||
                    !ticker_data["ask"].is_number() || !ticker_data["bid"].is_number()) {
                    // Log warning or skip, but don't necessarily fail the whole simulation yet
                    std::cerr << "Warning: Invalid ticker data for symbol '" << it.key() << "'" << std::endl;
                    continue;
                }
                tickers[it.key()] = { ticker_data["ask"], ticker_data["bid"] };
            }

            // Extract markets
            std::unordered_map<std::string, MarketInfo> markets;
            for (auto it = markets_json.begin(); it != markets_json.end(); ++it) {
                const auto& market_data = it.value();
                if (!market_data.is_object() || !market_data.contains("base") || !market_data.contains("quote") ||
                    !market_data["base"].is_string() || !market_data["quote"].is_string()) {
                    std::cerr << "Warning: Invalid market data for symbol '" << it.key() << "'" << std::endl;
                    continue;
                }
                markets[it.key()] = { market_data["base"], market_data["quote"] };
            }

            // --- Simulation Logic ---
            double sim_current_amount = actual_start_amount;
            std::string sim_current_currency = actual_start_currency;
            size_t start_trade_index = 0;
            bool skip_initial_swap_and_first_step = false; // Optimization logic


            // --- Optimization Check (same as Python) ---
            if (!trades.empty()) {
                const auto& first_trade = trades[0];
                if (sim_current_currency != cycle_start_currency &&
                    first_trade.from == cycle_start_currency &&
                    first_trade.to == actual_start_currency)
                {
                    skip_initial_swap_and_first_step = true;
                    start_trade_index = 1;
                    sim_logs.push_back("Optimization: Detected offsetting first step. Skipping initial swap and first trade.");
                    sim_logs.push_back("Starting simulation holding " + format_double(sim_current_amount) + " " + sim_current_currency);
                }
            }

            // --- 1. Initial Swap (if needed and not skipped) ---
            if (!skip_initial_swap_and_first_step && sim_current_currency != cycle_start_currency) {
                sim_logs.push_back("Step 0: Simulating initial swap " + sim_current_currency + " -> " + cycle_start_currency);
                SwapResult swap_res = simulate_swap_order_cpp(sim_current_currency, cycle_start_currency, sim_current_amount, tickers, markets, config.taker_fee_rate);

                if (swap_res.success && swap_res.estimated_to_amount > 0) {
                    sim_logs.push_back("  - Initial Swap Simulation (" + sim_current_currency + " -> " + cycle_start_currency + "):");
                    for (const auto& log : swap_res.steps) sim_logs.push_back("    - " + log);
                    sim_current_amount = swap_res.estimated_to_amount;
                    sim_current_currency = cycle_start_currency;
                    sim_logs.push_back("  - After initial swap, holding: " + format_double(sim_current_amount) + " " + sim_current_currency);
                }
                else {
                    reason = "Failed to simulate initial swap: " + swap_res.error_reason;
                    throw std::runtime_error(reason); // Fail simulation early
                }
            }
            else if (!skip_initial_swap_and_first_step) {
                sim_logs.push_back("Step 0: Start currency matches cycle start, no initial swap needed.");
            }


            // --- 2. Core Loop Simulation ---
            if (start_trade_index < trades.size()) {
                // Formatting header
                std::string fee_percentage_str = format_double(config.taker_fee_rate * 100.0, 4) + "%";
                std::string fee_column_label = "Fee(" + fee_percentage_str + ")";
                std::stringstream header_ss;
                header_ss << std::left // Align text to the left
                    << "  " << std::setw(4) << "Step" << " | "
                    << std::setw(14) << "Action" << " | "
                    << std::setw(12) << "Pair" << " | "
                    << std::setw(18) << "Price" << " | "
                    << std::setw(28) << "Sent Amount/Qty" << " | "
                    << std::setw(28) << fee_column_label << " | " // Adjust width as needed
                    << std::setw(28) << "Net Received Amt/Qty";
                sim_logs.push_back(header_ss.str());
                sim_logs.push_back(std::string(header_ss.str().length(), '-')); // Separator


                for (size_t i = start_trade_index; i < trades.size(); ++i) {
                    const auto& trade = trades[i];
                    size_t step_num_display = i + 1; // Actual step number

                    if (sim_current_currency != trade.from) {
                        reason = "Currency mismatch at step " + std::to_string(step_num_display) +
                            ": Expected " + trade.from + ", holding " + sim_current_currency;
                        throw std::runtime_error(reason);
                    }
                    if (!markets.count(trade.pair) || !tickers.count(trade.pair)) {
                        reason = "Missing market or ticker data for pair " + trade.pair + " at step " + std::to_string(step_num_display);
                        throw std::runtime_error(reason);
                    }

                    const auto& market = markets.at(trade.pair);
                    const auto& ticker = tickers.at(trade.pair);
                    double price = 0.0;
                    double net_amount = 0.0;
                    double fee = 0.0;
                    std::string next_currency = "";
                    std::string action_str = "";
                    std::string price_str = "N/A";
                    std::string sent_str = "";
                    std::string fee_str = "";
                    std::string received_str = "";


                    if (trade.type == "BUY") { // Buy base, spend quote
                        if (sim_current_currency != market.quote || trade.to != market.base) {
                            reason = "Logic error in BUY step " + std::to_string(step_num_display) + ": Mismatch currencies.";
                            throw std::runtime_error(reason);
                        }
                        price = ticker.ask;
                        if (price <= 0) throw std::runtime_error("Invalid ask price <= 0 for " + trade.pair);
                        price_str = format_double(price);
                        double gross_base_received = sim_current_amount / price;
                        fee = gross_base_received * config.taker_fee_rate;
                        net_amount = gross_base_received - fee;
                        next_currency = market.base;
                        action_str = "Buy " + market.base;
                        sent_str = format_double(sim_current_amount) + " " + market.quote;
                        fee_str = format_double(fee) + " " + market.base; // Fee in received currency
                        received_str = format_double(net_amount) + " " + market.base;

                    }
                    else if (trade.type == "SELL") { // Sell base, receive quote
                        if (sim_current_currency != market.base || trade.to != market.quote) {
                            reason = "Logic error in SELL step " + std::to_string(step_num_display) + ": Mismatch currencies.";
                            throw std::runtime_error(reason);
                        }
                        price = ticker.bid;
                        if (price <= 0) throw std::runtime_error("Invalid bid price <= 0 for " + trade.pair);
                        price_str = format_double(price);
                        double gross_quote_received = sim_current_amount * price;
                        fee = gross_quote_received * config.taker_fee_rate;
                        net_amount = gross_quote_received - fee;
                        next_currency = market.quote;
                        action_str = "Sell " + market.base;
                        sent_str = format_double(sim_current_amount) + " " + market.base;
                        fee_str = format_double(fee) + " " + market.quote; // Fee in received currency
                        received_str = format_double(net_amount) + " " + market.quote;

                    }
                    else {
                        throw std::runtime_error("Unknown trade type: " + trade.type);
                    }

                    if (net_amount <= 0) {
                        reason = "Amount became non-positive after step " + std::to_string(step_num_display);
                        sim_current_amount = 0; // Set to 0 before throwing
                        throw std::runtime_error(reason);
                    }
                    // --- Format log entry ---
                    std::stringstream log_ss;
                    log_ss << std::left
                        << "  " << std::setw(4) << step_num_display << " | "
                        << std::setw(14) << action_str << " | "
                        << std::setw(12) << trade.pair << " | "
                        << std::setw(18) << price_str << " | "
                        << std::setw(28) << sent_str << " | "
                        << std::setw(28) << fee_str << " | "
                        << std::setw(28) << received_str;
                    sim_logs.push_back(log_ss.str());

                    sim_current_amount = net_amount;
                    sim_current_currency = next_currency;
                }
                sim_logs.push_back("--- Core loop simulation ended, holding: " + format_double(sim_current_amount) + " " + sim_current_currency + " ---");
            }
            else {
                sim_logs.push_back("--- Core loop skipped (due to optimization or empty trades) ---");
            }


            // --- 3. Final Swap (if needed) ---
            bool final_swap_simulated = false;
            if (end_with_usdt && sim_current_currency != "USDT") {
                sim_logs.push_back("Step F: Simulating final swap " + sim_current_currency + " -> USDT");
                SwapResult final_swap_res = simulate_swap_order_cpp(sim_current_currency, "USDT", sim_current_amount, tickers, markets, config.taker_fee_rate);

                if (final_swap_res.success && final_swap_res.estimated_to_amount > 0) {
                    sim_logs.push_back("  - Final Swap Simulation (" + sim_current_currency + " -> USDT):");
                    for (const auto& log : final_swap_res.steps) sim_logs.push_back("    - " + log);
                    sim_current_amount = final_swap_res.estimated_to_amount;
                    sim_current_currency = "USDT";
                    sim_logs.push_back("  - After final swap, holding: " + format_double(sim_current_amount) + " " + sim_current_currency);
                    final_swap_simulated = true;
                }
                else {
                    // Final swap failed, log warning but proceed with calculation based on current state
                    reason = "Warning: Failed to simulate final swap to USDT: " + final_swap_res.error_reason + ". Final result is in " + sim_current_currency;
                    sim_logs.push_back("!! " + reason);
                }
            }
            else if (end_with_usdt && sim_current_currency == "USDT") {
                sim_logs.push_back("Step F: Already holding USDT, no final swap needed.");
                final_swap_simulated = true; // Considered 'done'
            }
            else {
                sim_logs.push_back("Step F: Final swap to USDT not required by config.");
            }


            // --- 4. Calculate Profit & Verify ---
            final_amount = sim_current_amount;
            final_currency = sim_current_currency;
            std::string target_currency_for_profit = end_with_usdt ? "USDT" : actual_start_currency;

            if (final_currency == target_currency_for_profit) {
                if (actual_start_amount > std::numeric_limits<double>::epsilon()) { // Check if start amount > 0
                    profit_amount = final_amount - actual_start_amount;
                    profit_percent = (profit_amount / actual_start_amount) * 100.0;
                }
                else {
                    profit_amount = final_amount; // If start is 0, profit is final amount
                    profit_percent = std::numeric_limits<double>::infinity(); // Or 0? Let's use infinity if final > 0
                    if (std::abs(final_amount) < std::numeric_limits<double>::epsilon()) profit_percent = 0.0;
                }

                sim_logs.push_back("--- Final Calculation vs " + actual_start_currency + " ---");
                sim_logs.push_back("Initial investment: " + format_double(actual_start_amount) + " " + actual_start_currency);
                sim_logs.push_back("Final holdings    : " + format_double(final_amount) + " " + final_currency);
                sim_logs.push_back("Profit            : " + format_double(profit_amount) + " " + target_currency_for_profit +
                    " (" + format_double(profit_percent, 4) + "%)");


                if (profit_percent > config.min_profit_percent_verify) {
                    verified = true;
                    reason = "Simulation successful and profit threshold met (" + format_double(profit_percent, 4) + "% > " + format_double(config.min_profit_percent_verify, 4) + "%)";
                    sim_logs.push_back("Verification: PASSED - " + reason);
                }
                else {
                    verified = false;
                    reason = "Profit threshold not met (" + format_double(profit_percent, 4) + "% <= " + format_double(config.min_profit_percent_verify, 4) + "%)";
                    sim_logs.push_back("Verification: FAILED - " + reason);
                }

            }
            else {
                // Final currency doesn't match target (likely final swap failed when end_with_usdt=true)
                verified = false;
                profit_amount = -998.0; // Indicate error state
                profit_percent = -998.0;
                reason = "Final currency (" + final_currency + ") does not match target profit currency (" + target_currency_for_profit + ").";
                sim_logs.push_back("Verification: FAILED - " + reason);
            }


        }
        catch (const json::parse_error& e) {
            std::cerr << "JSON Parse Error: " << e.what() << std::endl;
            reason = std::string("JSON Parse Error: ") + e.what();
            profit_percent = -997.1; // Specific error code
            *json_output_str_ptr = nullptr; // Ensure no garbage pointer
            return -1; // Indicate JSON error
        }
        catch (const std::runtime_error& e) {
            std::cerr << "Simulation Runtime Error: " << e.what() << std::endl;
            reason = std::string("Simulation Runtime Error: ") + e.what();
            profit_percent = -997.2; // Specific error code
            sim_logs.push_back("!! ERROR: " + reason); // Add error to log
            // Continue to generate output JSON with error state
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << std::endl;
            reason = std::string("Standard Exception: ") + e.what();
            profit_percent = -997.3; // Specific error code
            sim_logs.push_back("!! ERROR: " + reason);
        }
        catch (...) {
            std::cerr << "Unknown C++ Exception occurred during simulation." << std::endl;
            reason = "Unknown C++ Exception during simulation.";
            profit_percent = -997.4; // Specific error code
            sim_logs.push_back("!! ERROR: " + reason);
        }

        // --- 5. Construct Output JSON ---
        result_json["verified"] = verified;
        result_json["profit_percent"] = profit_percent;
        result_json["profit_amount"] = profit_amount;
        result_json["final_amount"] = final_amount;
        result_json["final_currency"] = final_currency;
        result_json["log_messages"] = sim_logs; // Add logs
        result_json["reason"] = reason;

        // 6. Allocate memory for the output string and copy JSON data
        std::string output_str = result_json.dump(2); // Pretty print with indent 2
        *json_output_str_ptr = new char[output_str.length() + 1];
        std::strcpy(*json_output_str_ptr, output_str.c_str());

        return 0; // Success
    }

    /**
     * @brief Frees memory allocated by simulate_full_execution_profit_cpp for the output JSON string.
     *
     * @param ptr The pointer previously filled by simulate_full_execution_profit_cpp.
     */
    DLLEXPORT void free_simulation_memory(char* ptr) {
        if (ptr != nullptr) {
            delete[] ptr;
        }
    }
} // extern "C"