

// Define this before any includes to suppress warnings about unsafe functions like strcpy
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>      // For cerr (fallback error output)
#include <vector>        // For std::vector
#include <string>        // For std::string
#include <unordered_map> // For std::unordered_map
#include <set>           // For std::set
#include <cmath>         // For std::log, std::isfinite
#include <stdexcept>     // For std::exception handling
#include <limits>        // For numeric limits (optional, good practice)
#include <algorithm>     // For std::sort
#include <iomanip>       // For std::put_time, std::setfill, std::setw
#include <cstring>       // For std::strcpy (used because of _CRT_SECURE_NO_WARNINGS)
#include <fstream>       // For std::ofstream (file logging)
#include <chrono>        // For std::chrono (timestamps)
#include <ctime>         // For std::time_t, std::tm, std::localtime_s/r
#include <sstream>       // For std::stringstream (formatting timestamps)

// --- Include nlohmann/json library ---
// Ensure the json.hpp file is accessible in your include paths
// You can download it from: https://github.com/nlohmann/json
#include "json.hpp" // Or adjust the path as needed, e.g., <nlohmann/json.hpp>

// Use the nlohmann::json namespace alias for convenience
using json = nlohmann::json;

// --- Optional: Edge struct (can be removed if only build_graph_cpp is in this DLL) ---
// This might be useful context if you plan to add more graph functions later
struct Edge {
    int from_node;
    int to_node;
    double weight;
    std::string pair_symbol;
    std::string trade_type;
};

// --- Platform-specific DLL export macro ---
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// --- Logging Helper Function ---
// Gets the current timestamp as a formatted string (YYYY-MM-DD HH:MM:SS.ms)
inline std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    // Get milliseconds part
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::tm now_tm;
#ifdef _WIN32
    localtime_s(&now_tm, &now_c); // Windows-specific secure function
#else
    localtime_r(&now_c, &now_tm); // POSIX thread-safe function
#endif
    std::stringstream ss;
    // Format the date and time part
    ss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    // Append milliseconds, padding with zeros if needed
    ss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
}

// --- C++ Function to Build the Graph ---
extern "C" DLLEXPORT char* build_graph_cpp(
    const char* tickers_json_str,
    const char* markets_json_str,
    double taker_fee_rate)
{
    // --- Open log file in append mode ---
    // The log file will be created in the same directory where the DLL is located
    // (or the current working directory of the Python process, depending on how DLL loading works)
    std::ofstream logFile("build_graph_cpp.log", std::ios::app);
    bool logFileOk = logFile.is_open(); // Check if file opened successfully

    if (logFileOk) {
        logFile << "[" << getCurrentTimestamp() << "] [C++] Entering build_graph_cpp..." << std::endl;
    }
    else {
        // Fallback to standard error if log file cannot be opened
        std::cerr << "[" << getCurrentTimestamp() << "] [C++] Entering build_graph_cpp... (Log file failed to open!)" << std::endl;
    }

    try {
        // 1. Parse Input JSON Strings
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Parsing tickers JSON (length: " << (tickers_json_str ? strlen(tickers_json_str) : 0) << ")..." << std::endl;
        json tickers_json = json::parse(tickers_json_str);
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Parsing markets JSON (length: " << (markets_json_str ? strlen(markets_json_str) : 0) << ")..." << std::endl;
        json markets_json = json::parse(markets_json_str);
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] JSON parsing complete." << std::endl;

        // Basic validation of JSON structure
        if (!tickers_json.is_object() || !markets_json.is_object()) {
            if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Error: Input JSON is not an object type." << std::endl;
            else std::cerr << "[C++] Error: Input JSON is not an object type." << std::endl;
            return nullptr; // Return null pointer to indicate error
        }

        // 2. Preprocess Tickers and Markets, Collect Currencies
        std::set<std::string> unique_currencies;
        std::vector<std::string> valid_symbols_for_edges; // Store symbols that pass initial checks

        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Starting currency collection and validation loop..." << std::endl;
        int ticker_count = 0;
        int valid_ticker_count = 0;

        // Iterate through tickers using iterators for better compatibility
        for (auto it = tickers_json.begin(); it != tickers_json.end(); ++it) {
            ticker_count++;
            const std::string& symbol = it.key();
            const json& ticker = it.value();

            // --- Ticker Validation ---
            if (!ticker.is_object() || !ticker.contains("ask") || !ticker.contains("bid")) {
                // Optional logging for skipped tickers due to structure
                // if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Skipping ticker " << symbol << ": Invalid structure or missing ask/bid." << std::endl;
                continue;
            }

            // --- Market Validation ---
            if (!markets_json.contains(symbol)) {
                // Optional logging for skipped tickers due to missing market info
                // if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Skipping ticker " << symbol << ": Market info not found." << std::endl;
                continue;
            }
            const json& market = markets_json[symbol];
            if (!market.is_object() || !market.value("active", false) || !market.value("spot", false) ||
                !market.contains("base") || !market.contains("quote")) {
                // Optional logging for skipped tickers due to inactive/invalid market
                // if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Skipping ticker " << symbol << ": Market inactive, not spot, or missing base/quote." << std::endl;
                continue;
            }

            // --- Price Validation (more robust) ---
            double ask_price = 0.0, bid_price = 0.0;
            bool ask_valid = false, bid_valid = false;

            // Try parsing 'ask' price (number or string)
            if (ticker["ask"].is_number()) {
                ask_price = ticker["ask"].get<double>();
                ask_valid = ask_price > 0; // Ensure positive price
            }
            else if (ticker["ask"].is_string()) {
                try {
                    std::string ask_str = ticker["ask"].get<std::string>();
                    ask_price = std::stod(ask_str);
                    ask_valid = ask_price > 0; // Ensure positive price
                }
                catch (...) { /* Conversion failed, ask_valid remains false */ }
            }

            // Try parsing 'bid' price (number or string)
            if (ticker["bid"].is_number()) {
                bid_price = ticker["bid"].get<double>();
                bid_valid = bid_price > 0; // Ensure positive price
            }
            else if (ticker["bid"].is_string()) {
                try {
                    std::string bid_str = ticker["bid"].get<std::string>();
                    bid_price = std::stod(bid_str);
                    bid_valid = bid_price > 0; // Ensure positive price
                }
                catch (...) { /* Conversion failed, bid_valid remains false */ }
            }

            // If either price is invalid or not positive, skip this symbol
            if (!ask_valid || !bid_valid) {
                // Optional logging for skipped tickers due to invalid price
                // if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Skipping ticker " << symbol << ": Invalid or non-positive ask/bid price (ask=" << ask_price << ", bid=" << bid_price << ")." << std::endl;
                continue;
            }

            // --- Passed all checks ---
            valid_ticker_count++;
            // Use .get_ref() for safer access to string members
            const std::string& base_currency = market["base"].get_ref<const std::string&>();
            const std::string& quote_currency = market["quote"].get_ref<const std::string&>();

            unique_currencies.insert(base_currency);
            unique_currencies.insert(quote_currency);
            valid_symbols_for_edges.push_back(symbol); // Add symbol to list for edge generation
        }
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Currency collection loop finished. Processed " << ticker_count << " tickers. Found " << unique_currencies.size() << " unique currencies from " << valid_symbols_for_edges.size() << " valid symbols." << std::endl;

        // Check if any valid data was found
        if (unique_currencies.empty() || valid_symbols_for_edges.empty()) {
            if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Error: No valid currencies or symbols found after filtering." << std::endl;
            else std::cerr << "[C++] Error: No valid currencies or symbols found after filtering." << std::endl;
            return nullptr;
        }

        // 3. Create Currency Index Mapping
        std::vector<std::string> index_to_currency_cpp(unique_currencies.begin(), unique_currencies.end());
        // Sort for deterministic node indexing (important!)
        std::sort(index_to_currency_cpp.begin(), index_to_currency_cpp.end());

        std::unordered_map<std::string, int> currency_to_index_cpp;
        for (int i = 0; i < index_to_currency_cpp.size(); ++i) {
            currency_to_index_cpp[index_to_currency_cpp[i]] = i;
        }
        int num_currencies = static_cast<int>(index_to_currency_cpp.size()); // Use static_cast
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Currency mapping created. Number of nodes: " << num_currencies << std::endl;

        // 4. Build Edge List with Weights
        std::vector<json> edges_json_array;
        double fee_multiplier = 1.0 - taker_fee_rate;
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Starting edge building loop (Fee Multiplier: " << fee_multiplier << ")..." << std::endl;
        int edge_count = 0;

        for (const std::string& symbol : valid_symbols_for_edges) {
            const json& ticker = tickers_json[symbol]; // Already validated
            const json& market = markets_json[symbol]; // Already validated
            const std::string& base = market["base"].get_ref<const std::string&>();
            const std::string& quote = market["quote"].get_ref<const std::string&>();

            // Retrieve validated prices again (safer than assuming they were stored)
            double ask_price = 0.0, bid_price = 0.0;
            if (ticker["ask"].is_number()) { ask_price = ticker["ask"].get<double>(); }
            else if (ticker["ask"].is_string()) { try { ask_price = std::stod(ticker["ask"].get<std::string>()); } catch (...) {} }
            if (ticker["bid"].is_number()) { bid_price = ticker["bid"].get<double>(); }
            else if (ticker["bid"].is_string()) { try { bid_price = std::stod(ticker["bid"].get<std::string>()); } catch (...) {} }

            // Check if base and quote currencies are in our map (should always be true here)
            if (currency_to_index_cpp.find(base) == currency_to_index_cpp.end() ||
                currency_to_index_cpp.find(quote) == currency_to_index_cpp.end()) {
                if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Warning: Logic error - Could not find index for base/quote in pre-validated symbol: " << symbol << std::endl;
                continue; // Skip this edge if something went wrong
            }
            int base_idx = currency_to_index_cpp[base];
            int quote_idx = currency_to_index_cpp[quote];

            // Edge a: Quote -> Base (BUY Base using Quote) - Use Ask Price
            if (ask_price > 0) { // Price validity already checked
                double net_rate_q_to_b = (1.0 / ask_price) * fee_multiplier;
                if (net_rate_q_to_b > 0) { // Check net rate before log
                    double weight = -std::log(net_rate_q_to_b);
                    if (std::isfinite(weight)) { // Check if log result is finite
                        edges_json_array.push_back({
                            {"from", quote_idx},
                            {"to", base_idx},
                            {"weight", weight},
                            {"pair", symbol},
                            {"type", "BUY"}
                            });
                        edge_count++;
                    }
                    else {
                        // Optional logging for infinite weight
                        // if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Warning: Calculated non-finite weight for BUY edge " << symbol << " (rate=" << net_rate_q_to_b << ")" << std::endl;
                    }
                } // else: Optional logging for non-positive net rate
            }

            // Edge b: Base -> Quote (SELL Base for Quote) - Use Bid Price
            if (bid_price > 0) { // Price validity already checked
                double net_rate_b_to_q = bid_price * fee_multiplier;
                if (net_rate_b_to_q > 0) { // Check net rate before log
                    double weight = -std::log(net_rate_b_to_q);
                    if (std::isfinite(weight)) { // Check if log result is finite
                        edges_json_array.push_back({
                            {"from", base_idx},
                            {"to", quote_idx},
                            {"weight", weight},
                            {"pair", symbol},
                            {"type", "SELL"}
                            });
                        edge_count++;
                    }
                    else {
                        // Optional logging for infinite weight
                        // if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Warning: Calculated non-finite weight for SELL edge " << symbol << " (rate=" << net_rate_b_to_q << ")" << std::endl;
                    }
                } // else: Optional logging for non-positive net rate
            }
        } // End of edge building loop

        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Edge building loop finished. Generated " << edges_json_array.size() << " edges (expected ~ " << edge_count << ")." << std::endl;

        // Check if any edges were actually generated
        if (edges_json_array.empty()) {
            if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Error: No valid edges generated after processing symbols." << std::endl;
            else std::cerr << "[C++] Error: No valid edges generated after processing symbols." << std::endl;
            return nullptr;
        }

        // 5. Construct Final Output JSON
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Building final output JSON..." << std::endl;
        json output_json;
        output_json["nodes"] = index_to_currency_cpp; // Array of currency strings
        output_json["edges"] = edges_json_array;      // Array of edge objects
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Output JSON constructed." << std::endl;


        // 6. Serialize JSON to String and Allocate Memory for Return
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Serializing output JSON to string..." << std::endl;
        std::string output_str = output_json.dump(); // Use default dump (no indentation for efficiency)
        size_t bufferSize = output_str.length() + 1; // +1 for the null terminator
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Allocating memory buffer (size: " << bufferSize << " bytes)..." << std::endl;
        // Allocate memory that Python (ctypes) will need to free later using free_memory
        char* result_c_str = new (std::nothrow) char[bufferSize]; // Use nothrow version
        if (!result_c_str) {
            if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Error: Failed to allocate memory for output string." << std::endl;
            else std::cerr << "[C++] Error: Failed to allocate memory for output string." << std::endl;
            return nullptr;
        }
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Memory allocated successfully at address " << static_cast<void*>(result_c_str) << "." << std::endl;

        // Copy the JSON string into the allocated buffer using the potentially "unsafe" strcpy
        // (We defined _CRT_SECURE_NO_WARNINGS at the top to allow this)
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Copying JSON string to buffer using strcpy..." << std::endl;
        std::strcpy(result_c_str, output_str.c_str());
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] String copied successfully." << std::endl;


        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] build_graph_cpp returning JSON string pointer successfully." << std::endl;
        return result_c_str; // Return the pointer to the C-style string

        // --- Exception Handling ---
    }
    catch (const json::parse_error& e) {
        // Log JSON parsing errors
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] JSON parsing error in build_graph_cpp: " << e.what() << " (byte: " << e.byte << ")" << std::endl;
        else std::cerr << "[C++] JSON parsing error in build_graph_cpp: " << e.what() << " (byte: " << e.byte << ")" << std::endl;
        return nullptr; // Indicate failure
    }
    catch (const std::exception& e) {
        // Log standard C++ exceptions
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Standard exception caught in build_graph_cpp: " << e.what() << std::endl;
        else std::cerr << "[C++] Standard exception caught in build_graph_cpp: " << e.what() << std::endl;
        return nullptr; // Indicate failure
    }
    catch (...) {
        // Log any other unknown exceptions
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Unknown exception caught in build_graph_cpp." << std::endl;
        else std::cerr << "[C++] Unknown exception caught in build_graph_cpp." << std::endl;
        return nullptr; // Indicate failure
    }
} // End of build_graph_cpp


// --- Memory Freeing Function (Exported for Python) ---
// This function MUST be called by Python to free the memory allocated by build_graph_cpp
extern "C" DLLEXPORT void free_memory(char* ptr) {
    std::ofstream logFile("build_graph_cpp.log", std::ios::app);
    bool logFileOk = logFile.is_open();

    if (logFileOk) {
        logFile << "[" << getCurrentTimestamp() << "] [C++] free_memory called";
        if (ptr != nullptr) {
            logFile << " for pointer: " << static_cast<void*>(ptr) << "." << std::endl;
        }
        else {
            logFile << " with nullptr." << std::endl;
        }
    }

    if (ptr != nullptr) {
        delete[] ptr; // Use delete[] because memory was allocated with new char[...]
        if (logFileOk) logFile << "[" << getCurrentTimestamp() << "] [C++] Memory freed successfully." << std::endl;
    }
}

