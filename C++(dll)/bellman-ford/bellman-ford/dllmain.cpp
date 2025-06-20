// ---------- START OF FILE dllmain.cpp ----------

#include <vector>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <set>
#include <cstring>
#include <iostream> // Keep for potential cerr debugging if needed
#include <memory>
#include <queue>
#include <utility>
#include <fstream>    // ** Added for file logging **
#include <chrono>     // ** Added for timestamps **
#include <iomanip>    // ** Added for formatting timestamps **
#include <sstream>    // ** Added for formatting log messages **
#include <mutex>      // ** Added for potential thread safety (currently commented out) **

// --- Conditional DLL_EXPORT Macro ---
#ifndef DLL_EXPORT
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif
#endif // DLL_EXPORT

// Define the Edge structure
struct Edge {
    int from_node;
    int to_node;
    double weight;
    const char* pair_symbol;
    const char* trade_type;
};

// Structure to hold details of a single trade step within a cycle
struct TradeStep {
    int from_node_idx;
    int to_node_idx;
    const char* pair_symbol;
    const char* trade_type;
};

// Structure to hold a complete detected arbitrage cycle
struct ArbitrageCycle {
    int depth;
    std::vector<int> node_indices; // Sequence of node indices [A, B, C, A]
    std::vector<TradeStep> trades; // Sequence of trades [A->B, B->C, C->A]
};

// --- Logging Setup ---
// NOTE: This basic file logger is NOT thread-safe if Python calls
// find_negative_cycles concurrently from multiple threads (like run_in_executor might).
// For debugging a single hang, it's likely sufficient. For production with true
// concurrency, add locking (e.g., std::mutex).
// std::mutex log_mutex; // Uncomment for thread safety
std::ofstream spfaLogFile;

// Helper to get formatted timestamp
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm timeinfo = {}; // <--- 创建一个 tm 结构体变量来接收结果
    errno_t err = localtime_s(&timeinfo, &now_c); // <--- 使用 localtime_s

    std::stringstream ss;
    if (err == 0) { // <--- 检查 localtime_s 是否成功
        // Format adjusted slightly for typical log formats
        ss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count();
    }
    else {
        // 如果 localtime_s 失败，记录一个错误时间戳
        ss << "YYYY-MM-DD HH:MM:SS.ms_localtime_s_error";
    }
    return ss.str();
}



// Helper function to safely create std::string from potentially null C-string
std::string safe_json_string(const char* c_str) {
    if (c_str == nullptr) return "";
    std::string s = c_str;
    std::string escaped_s;
    escaped_s.reserve(s.length());
    for (char c : s) {
        switch (c) {
        case '"':  escaped_s += "\\\""; break;
        case '\\': escaped_s += "\\\\"; break;
        case '/':  escaped_s += "\\/";  break;
        case '\b': escaped_s += "\\b";  break;
        case '\f': escaped_s += "\\f";  break;
        case '\n': escaped_s += "\\n";  break;
        case '\r': escaped_s += "\\r";  break;
        case '\t': escaped_s += "\\t";  break;
        default:
            // Allow safe printable ASCII and UTF-8 characters
            if ((c >= 32 && c <= 126) || (static_cast<unsigned char>(c) >= 128)) {
                escaped_s += c;
            }
            // Potentially handle other control characters if needed, currently ignored
            break;
        }
    }
    return escaped_s;
}


// Use extern "C" to prevent C++ name mangling for DLL functions
extern "C" {

    /**
     * @brief Finds negative weight cycles using the SPFA algorithm. (SPFA Version with File Logging)
     *        Logs detailed steps to "spfa_log.txt".
     * @param num_currencies Total number of unique currencies (nodes).
     * @param edges Pointer to an array of Edge structures representing the graph.
     * @param num_edges The number of edges in the 'edges' array.
     * @param max_depth The maximum cycle depth (number of trades) to report.
     * @param out_json_result Pointer to a char pointer. If cycles are found, this will
     *                        be set to point to a newly allocated JSON string containing
     *                        the cycle details. The caller (Python) is responsible for
     *                        calling free_memory() on this pointer.
     * @param out_relaxation_count Pointer to a long long. Returns the total number of
     *                             edge relaxation checks performed (may be approximate if limit hit).
     * @return int Returns 0 on success, -1 on input error, -2 on memory error,
     *         -3 on internal error, -4 unknown error, -5 on SPFA iteration limit hit.
     *         out_json_result will be nullptr if no cycles found or an error occurs.
     */
    DLL_EXPORT int find_negative_cycles(
        int num_currencies,
        const Edge* edges,
        int num_edges,
        int max_depth,
        char** out_json_result,
        long long* out_relaxation_count
    ) {
        // --- Open Log File (Append Mode) ---
        // Using static ensures it's initialized once per process load,
        // but we open in append mode each time the function is called.
        // This avoids needing explicit close/reopen logic if the DLL stays loaded.
        if (!spfaLogFile.is_open()) {
            spfaLogFile.open("spfa_log.txt", std::ios::app);
            if (spfaLogFile.is_open()) {
                spfaLogFile << "--- Log file opened at " << getCurrentTimestamp() << " ---" << std::endl;
            }
            else {
                // Optionally print to stderr if log file fails to open
                std::cerr << "Error: Failed to open spfa_log.txt for writing." << std::endl;
            }
        }

        

        // --- Input Validation ---
        *out_json_result = nullptr;
        if (out_relaxation_count) {
            *out_relaxation_count = 0;
        }
        if (num_currencies <= 0 || edges == nullptr || num_edges < 0 || max_depth <= 0) {
            
            return -1; // Invalid input
        }

        long long current_relaxation_checks = 0;
        // *** Safety Break: Iteration Limit ***
        // Adjust multiplier as needed, this is a safeguard against true infinite loops
        long long iteration_limit = static_cast<long long>(num_currencies) * num_edges * 2 + num_currencies;
        long long spfa_loop_iterations = 0;
        bool limit_reached = false;

        try {
            // --- SPFA Initialization ---
            std::vector<double> distance(num_currencies, std::numeric_limits<double>::infinity());
            std::vector<int> predecessor(num_currencies, -1);
            std::vector<int> predecessor_edge_index(num_currencies, -1);
            std::vector<int> enqueue_count(num_currencies, 0);
            std::vector<bool> in_queue(num_currencies, false);
            std::queue<int> q;

            // *** Build Adjacency List ***
            std::vector<std::vector<std::pair<int, int>>> adj(num_currencies);
            for (int i = 0; i < num_edges; ++i) {
                const Edge& edge = edges[i];
                if (edge.from_node >= 0 && edge.from_node < num_currencies &&
                    edge.to_node >= 0 && edge.to_node < num_currencies)
                {
                    adj[edge.from_node].push_back({ edge.to_node, i });
                }
            }

            // --- Choose Start Node and Initialize Queue ---
            // SPFA can detect cycles starting from any node, but we need an initial push.
            // Pushing all nodes might be more robust in disconnected graphs, but uses more memory.
            // Pushing node 0 is common practice if the graph is expected to be connected enough.
            // Let's push all nodes to be safer for potentially disconnected components.

            for (int i = 0; i < num_currencies; ++i) {
                distance[i] = 0.0; // Assume all nodes can potentially start a path
                q.push(i);
                in_queue[i] = true;
                enqueue_count[i] = 1;
            }


            std::vector<int> cycle_candidate_nodes;
            cycle_candidate_nodes.reserve(num_currencies / 10);


            // --- SPFA Core Loop ---
            while (!q.empty()) {
                // *** Iteration Limit Check ***
                if (spfa_loop_iterations++ > iteration_limit) {
                    
                    limit_reached = true;
                    break; // Exit the loop
                }
                if (spfa_loop_iterations % 10000 == 0 && spfa_loop_iterations > 0) { // Log progress periodically
                   
                }


                int u = q.front();
                q.pop();
                in_queue[u] = false;
                // LOG_SPFA("Dequeued node " + std::to_string(u) + ". Queue size: " + std::to_string(q.size()));

                // Iterate through neighbors using adjacency list
                for (const auto& edge_pair : adj[u]) {
                    int v = edge_pair.first;
                    int edge_idx = edge_pair.second;
                    const Edge& edge = edges[edge_idx];
                    double weight = edge.weight;

                    current_relaxation_checks++;

                    // Check for NaN/Inf weights (important!)
                    if (std::isnan(weight) || std::isinf(weight) || std::isnan(distance[u]) || std::isinf(distance[u])) {
                        // LOG_SPFA("Skipping edge " + std::to_string(u) + "->" + std::to_string(v) + " due to NaN/Inf value.");
                        continue;
                    }

                    // Relaxation condition
                    double new_distance = distance[u] + weight;
                    if (new_distance < distance[v]) {
                        // LOG_SPFA("Relaxed edge " + std::to_string(u) + "->" + std::to_string(v) + " (idx=" + std::to_string(edge_idx) + "). New dist[" + std::to_string(v) + "]=" + std::to_string(new_distance) + " (Old=" + std::to_string(distance[v]) + ")");
                        distance[v] = new_distance;
                        predecessor[v] = u;
                        predecessor_edge_index[v] = edge_idx;

                        if (!in_queue[v]) {
                            // Check if node v is eligible for re-queueing based on count
                            if (enqueue_count[v] < num_currencies) { // Standard check
                                q.push(v);
                                in_queue[v] = true;
                                enqueue_count[v]++;
                                // LOG_SPFA("Enqueued node " + std::to_string(v) + ". Count: " + std::to_string(enqueue_count[v]) + ". Queue size: " + std::to_string(q.size()));

                                // *** Negative Cycle Detection ***
                                if (enqueue_count[v] >= num_currencies) {
                                   
                                    // Only add if not already added? Or let extraction handle duplicates.
                                    cycle_candidate_nodes.push_back(v);
                                    // Prevent this node from being added many times if multiple cycles affect it
                                    // Or just let the post-processing unique filter handle it.
                                    // Let's rely on the post-processing filter for now.
                                }
                            }
                            else {
                                // Node v has already been enqueued N times, it's part of a negative cycle
                                // No need to enqueue again, the detection should have already happened.
                                // LOG_SPFA("Node " + std::to_string(v) + " already reached enqueue limit, not re-queueing.");
                            }
                        }
                    }
                } // End neighbor loop
            } // End SPFA while loop


            // --- Update Output Relaxation Count ---
            if (out_relaxation_count) {
                *out_relaxation_count = current_relaxation_checks;
            }

            // --- Handle Iteration Limit Reached ---
            if (limit_reached) {
                return -5; // Return specific error code for iteration limit
            }

            // --- Negative Cycle Extraction ---
           
            if (cycle_candidate_nodes.empty()) {

                return 0; // Success, no cycles found
            }


            std::vector<ArbitrageCycle> found_cycles;
            found_cycles.reserve(cycle_candidate_nodes.size());
            std::set<std::vector<int>> found_cycle_signatures;
            std::vector<bool> node_in_found_cycle(num_currencies, false);
            std::vector<int> visited_marker(num_currencies); // Use for backtrack path marking

            // De-duplicate candidate nodes before processing
            std::sort(cycle_candidate_nodes.begin(), cycle_candidate_nodes.end());
            cycle_candidate_nodes.erase(std::unique(cycle_candidate_nodes.begin(), cycle_candidate_nodes.end()), cycle_candidate_nodes.end());



            for (int node_updated : cycle_candidate_nodes) {
                // LOG_SPFA("Processing candidate node: " + std::to_string(node_updated));
                if (node_in_found_cycle[node_updated]) {
                    // LOG_SPFA("Node " + std::to_string(node_updated) + " already part of a found cycle. Skipping.");
                    continue;
                }

                // --- Backtrack to find the cycle ---
                std::fill(visited_marker.begin(), visited_marker.end(), -1);
                std::vector<int> cycle_path_rev;
                cycle_path_rev.reserve(num_currencies);
                int current_node = node_updated;
                int path_pos = 0;

                // Backtrack N steps first to ensure we are *in* the cycle
                int backtrack_node = node_updated;
                for (int i = 0; i < num_currencies && backtrack_node != -1; ++i) {
                    backtrack_node = predecessor[backtrack_node];
                }
                if (backtrack_node == -1) {
                    // LOG_SPFA("Backtrack from " + std::to_string(node_updated) + " failed to reach N steps or hit start.");
                    continue; // Cannot reliably find cycle if backtrack fails
                }
                current_node = backtrack_node; // Start cycle detection from here

                // Now trace back from the node guaranteed to be in the cycle
                path_pos = 0; // Reset path position counter
                while (current_node != -1 && visited_marker[current_node] == -1 && path_pos <= num_currencies + 1) { // Allow slightly longer path for safety
                    visited_marker[current_node] = path_pos;
                    cycle_path_rev.push_back(current_node);
                    path_pos++;
                    current_node = predecessor[current_node];
                }


                if (current_node == -1 || visited_marker[current_node] == -1) {
                    // LOG_SPFA("Failed to close cycle during backtrack from node " + std::to_string(node_updated) + " (start backtrack from " + std::to_string(backtrack_node) + "). Final node: " + std::to_string(current_node));
                    continue;
                }
                if (path_pos > num_currencies + 1) {
                    
                    continue;
                }


                // --- Extract Cycle ---
                int cycle_start_pos_in_rev = visited_marker[current_node];
                // Check if start position is valid
                if (cycle_start_pos_in_rev < 0 || cycle_start_pos_in_rev >= cycle_path_rev.size()) {
                    
                    continue;
                }
                std::vector<int> cycle_nodes_rev(cycle_path_rev.begin() + cycle_start_pos_in_rev, cycle_path_rev.end());
                std::vector<int> cycle_nodes = cycle_nodes_rev;
                std::reverse(cycle_nodes.begin(), cycle_nodes.end());
                cycle_nodes.push_back(cycle_nodes[0]); // Close the loop [A, B, C, A]

                int depth = static_cast<int>(cycle_nodes.size()) - 1;

                if (depth <= 0 || depth > max_depth) {
                    // LOG_SPFA("Extracted cycle depth " + std::to_string(depth) + " is invalid or exceeds max_depth. Skipping.");
                    continue;
                }

                // --- Check Signature for Duplicates ---
                std::vector<int> signature_nodes = cycle_nodes; // Copy
                signature_nodes.pop_back(); // Remove duplicate end node for signature
                std::sort(signature_nodes.begin(), signature_nodes.end());
                if (found_cycle_signatures.count(signature_nodes)) {
                    // LOG_SPFA("Duplicate cycle signature found. Skipping.");
                    continue;
                }

                // --- Reconstruct Trade Steps ---
                std::vector<TradeStep> trades;
                trades.reserve(depth);
                bool reconstruction_ok = true;
                // LOG_SPFA("Reconstructing trades for cycle (depth " + std::to_string(depth) + ")...");
                for (size_t i = 0; i < depth; ++i) {
                    int u_node = cycle_nodes[i];
                    int v_node = cycle_nodes[i + 1];
                    int edge_idx = -1;

                    // Search for the edge u_node -> v_node.
                    // Using predecessor_edge_index[v_node] might be faster if reliable,
                    // but searching all edges is more robust if multiple paths exist.
                    // Let's try searching all edges for robustness.
                    for (int k = 0; k < num_edges; ++k) {
                        if (edges[k].from_node == u_node && edges[k].to_node == v_node) {
                            // Check if weight matches approximately (optional sanity check)
                            // if (abs(distance[u_node] + edges[k].weight - distance[v_node]) < 1e-9) { ... }
                            edge_idx = k;
                            break;
                        }
                    }
                    /* // Alternative: Try using predecessor edge info first
                    int pred_edge_idx = predecessor_edge_index[v_node];
                     if (pred_edge_idx != -1 && edges[pred_edge_idx].from_node == u_node) {
                          edge_idx = pred_edge_idx;
                     } else { // Fallback search if predecessor info is wrong
                         for (int k = 0; k < num_edges; ++k) { // ... search ... }
                     }*/


                    if (edge_idx != -1) {
                        const Edge& edge = edges[edge_idx];
                        trades.push_back({ u_node, v_node, edge.pair_symbol, edge.trade_type });
                        // LOG_SPFA("  Step " + std::to_string(i+1) + ": " + std::to_string(u_node) + " -> " + std::to_string(v_node) + " (Edge idx " + std::to_string(edge_idx) + ")");
                    }
                    else {
                       
                        reconstruction_ok = false;
                        break;
                    }
                }

                // --- Store Valid Cycle ---
                if (reconstruction_ok && trades.size() == depth) {
                    // Construct path string for logging
                    std::string path_str_log;
                    for (size_t node_i = 0; node_i < cycle_nodes.size(); ++node_i) {
                        path_str_log += std::to_string(cycle_nodes[node_i]);
                        if (node_i < cycle_nodes.size() - 1) path_str_log += " -> ";
                    }
                   

                    found_cycles.push_back({ depth, cycle_nodes, trades });
                    found_cycle_signatures.insert(signature_nodes);

                    // Mark nodes as part of a found cycle
                    for (size_t i = 0; i < depth; ++i) {
                        node_in_found_cycle[cycle_nodes[i]] = true;
                    }
                }
                else {
                    // LOG_SPFA("Cycle reconstruction failed or trade count mismatch. Discarding.");
                }
            } // End loop through cycle_candidate_nodes
           

            // --- Format Results as JSON ---
            if (!found_cycles.empty()) {
                
                std::string json_str = "[";
                bool first_cycle = true;
                for (const auto& cycle : found_cycles) {
                    if (!first_cycle) json_str += ",";
                    first_cycle = false;

                    json_str += "{";
                    json_str += "\"depth\": " + std::to_string(cycle.depth) + ",";
                    json_str += "\"nodes\": [";
                    bool first_node = true;
                    for (int node_idx : cycle.node_indices) {
                        if (!first_node) json_str += ",";
                        first_node = false;
                        json_str += std::to_string(node_idx);
                    }
                    json_str += "],";
                    json_str += "\"trades\": [";
                    bool first_trade = true;
                    for (const auto& trade : cycle.trades) {
                        if (!first_trade) json_str += ",";
                        first_trade = false;
                        json_str += "{";
                        json_str += "\"from_node\": " + std::to_string(trade.from_node_idx) + ",";
                        json_str += "\"to_node\": " + std::to_string(trade.to_node_idx) + ",";
                        json_str += "\"pair\": \"" + safe_json_string(trade.pair_symbol) + "\",";
                        json_str += "\"type\": \"" + safe_json_string(trade.trade_type) + "\"";
                        json_str += "}";
                    }
                    json_str += "]"; // End trades array
                    json_str += "}"; // End cycle object
                }
                json_str += "]"; // End main array

                // Allocate memory for the JSON string
                size_t len = json_str.length();
                char* result_buffer = new (std::nothrow) char[len + 1];
                if (result_buffer == nullptr) {
                    
                    return -2; // Memory allocation error
                }
                std::memcpy(result_buffer, json_str.c_str(), len);
                result_buffer[len] = '\0'; // Null-terminate

                *out_json_result = result_buffer;
                
            }
            else {
                
            }

            
            return 0; // Success

        }
        catch (const std::bad_alloc& ba) {
           
            if (out_relaxation_count) *out_relaxation_count = current_relaxation_checks; // Best effort count
            return -2;
        }
        catch (const std::exception& e) {
            
            if (out_relaxation_count) *out_relaxation_count = current_relaxation_checks;
            return -3;
        }
        catch (...) {
            
            if (out_relaxation_count) *out_relaxation_count = current_relaxation_checks;
            return -4;
        }
    } // End find_negative_cycles

    /**
     * @brief Frees memory allocated by find_negative_cycles for the JSON result.
     * @param ptr Pointer to the memory block (JSON string) to be freed.
     */
    DLL_EXPORT void free_memory(char* ptr) {
        // Optionally add logging here too
        // std::ofstream logFile("spfa_log.txt", std::ios::app);
        // if (logFile.is_open()) {
        //     logFile << getCurrentTimestamp() << " [Memory] free_memory called for ptr: " << static_cast<void*>(ptr) << std::endl;
        // }
        if (ptr != nullptr) {
            delete[] ptr;
        }
    }

} // extern "C"

// Optional: Add DllMain for Windows if specific DLL load/unload actions are needed
#ifdef _WIN32
#include <windows.h>
BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        // Optionally close log file if opened globally on detach?
        // if (spfaLogFile.is_open()) { spfaLogFile.close(); }
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        // Close log file when DLL unloads (best effort)
        if (spfaLogFile.is_open()) {
            spfaLogFile << "--- DLL Detaching / Log file closing at " << getCurrentTimestamp() << " ---" << std::endl;
            spfaLogFile.close();
        }
        break;
    }
    return TRUE;
}
#endif // _WIN32

// ---------- END OF FILE dllmain.cpp ----------