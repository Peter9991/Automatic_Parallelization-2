/*
 * AutoParallel Server — CSCI465/ECEN433, Nile University
 *
 * A self-contained C++ HTTP server that:
 *   1. Serves index.html at GET /
 *   2. Accepts C++ code via POST /analyze  → returns JSON with parallelized code + loop info
 *   3. Accepts thread count via POST /benchmark → returns JSON with estimated speedup per loop
 *
 * Build:
 *   g++ -O2 -fopenmp -std=c++17 -o server server.cpp -lmicrohttpd
 *
 * Run:
 *   ./server          (listens on http://localhost:8080)
 *
 * Then open http://localhost:8080 in your browser.
 */

#include <microhttpd.h>
#include <omp.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <functional>
#include <mutex>
#include <random>

using namespace std;

// ─── Port ────────────────────────────────────────────────────────────────────
#define PORT 8080

// ─── Helpers ─────────────────────────────────────────────────────────────────
static string trim(const string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == string::npos) ? "" : s.substr(a, b - a + 1);
}

static string indent_of(const string& line) {
    size_t n = 0;
    while (n < line.size() && (line[n] == ' ' || line[n] == '\t')) n++;
    return line.substr(0, n);
}

// For annotated code shown inside HTML <span> elements.
static string html_escape(const string& s) {
    string out;
    for (char c : s) {
        if      (c == '&')  out += "&amp;";
        else if (c == '<')  out += "&lt;";
        else if (c == '>')  out += "&gt;";
        else if (c == '"')  out += "&quot;";
        else                out += c;
    }
    return out;
}

// Escape special characters so user code can be embedded safely inside JSON strings.
static string json_escape(const string& s) {
    string out;
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    return out;
}

// ─── Analysis structures ─────────────────────────────────────────────────────
// One record per for-loop discovered in the user's source.
struct LoopInfo {
    int    line_number  = 0;
    bool   parallelized = false;
    bool   has_reduction= false;
    bool   nested       = false;
    string loop_var;
    string init_expr;
    string limit_expr;
    string block_reason;
    vector<string> reductions;
    vector<string> private_vars;
};

// ─── Static analysis helpers ─────────────────────────────────────────────────

// Extract loop variable and bounds from a line like: for (int i = 0; i < n; i++)
static bool parse_for_header(const string& line, LoopInfo& out) {
    regex re(
        R"(for\s*\(\s*(?:(?:int|long|size_t|unsigned|unsigned\s+int)\s+)?)"
        R"((\w+)\s*=\s*([^;]+);\s*\1\s*[<>]=?\s*([^;]+);\s*)"
        R"((?:\+\+\1|\1\+\+|\1\s*\+=\s*\d+|--\1|\1--|\1\s*-=\s*\d+)\s*\))"
    );
    smatch m;
    if (!regex_search(line, m, re)) return false;
    out.loop_var   = trim(m[1].str());
    out.init_expr  = trim(m[2].str());
    out.limit_expr = trim(m[3].str());
    return true;
}

// Return [start, end] line indices of the brace-delimited loop body.
static pair<int,int> find_body(const vector<string>& lines, int from) {
    int depth = 0, start = -1, end = -1;
    for (int i = from; i < (int)lines.size(); i++) {
        for (char c : lines[i]) {
            if (c == '{') { if (depth == 0) start = i; depth++; }
            if (c == '}') { depth--; if (depth == 0) { end = i; return {start, end}; } }
        }
    }
    return {start, end};
}

// Detect loop-carried dependencies: arr[i±k] on the right-hand side of an assignment.
static vector<string> detect_carried_deps(
        const vector<string>& body, const string& v) {
    vector<string> deps;
    regex dep_re(R"(\b(\w+)\[\s*)" + v + R"(\s*[\+\-]\s*\d+\s*\])");
    for (const auto& l : body) {
        // Only scan the RHS (after '='), not the left-hand side.
        int eq = -1;
        for (int k = 1; k < (int)l.size(); k++) {
            char c = l[k], p = l[k-1];
            char nx = (k+1 < (int)l.size()) ? l[k+1] : 0;
            if (c == '=' && p != '!' && p != '<' && p != '>' && p != '=' && nx != '=') {
                eq = k; break;
            }
        }
        if (eq < 0) continue;
        string rhs = l.substr(eq + 1);
        smatch m;
        auto it = rhs.cbegin();
        while (regex_search(it, rhs.cend(), m, dep_re)) {
            string found = m[0].str();
            if (find(deps.begin(), deps.end(), found) == deps.end())
                deps.push_back(found);
            it = m.suffix().first;
        }
    }
    return deps;
}

// Find reduction patterns: +=, -=, *=, and max/min updates on scalar variables.
static vector<string> detect_reductions(
        const vector<string>& body, const string& v) {
    vector<string> result;
    set<string> seen;
    regex compound(R"(\b(\w+)\s*(\+|-|\*)=\s*[^=])");
    regex max_re(R"(if\s*\(\s*\w+[^)]*>\s*(\w+)\s*\)\s*\1\s*=)");
    regex min_re(R"(if\s*\(\s*\w+[^)]*<\s*(\w+)\s*\)\s*\1\s*=)");
    for (const auto& l : body) {
        smatch m;
        auto it = l.cbegin();
        while (regex_search(it, l.cend(), m, compound)) {
            string rv = m[1], op = m[2];
            if (rv != v) {
                string key = op + ":" + rv;
                if (!seen.count(key)) { seen.insert(key); result.push_back("reduction(" + op + ":" + rv + ")"); }
            }
            it = m.suffix().first;
        }
        if (regex_search(l, m, max_re)) {
            string key = "max:" + m[1].str();
            if (!seen.count(key)) { seen.insert(key); result.push_back("reduction(max:" + m[1].str() + ")"); }
        }
        if (regex_search(l, m, min_re)) {
            string key = "min:" + m[1].str();
            if (!seen.count(key)) { seen.insert(key); result.push_back("reduction(min:" + m[1].str() + ")"); }
        }
    }
    return result;
}

// Scalars declared inside the loop body should be marked private in the pragma.
static vector<string> detect_private(
        const vector<string>& body, const string& v) {
    vector<string> result;
    set<string> seen;
    regex decl(R"(\b(?:int|long|float|double|char|bool|size_t)\s+(\w+)\s*[=;,])");
    for (const auto& l : body) {
        smatch m;
        auto it = l.cbegin();
        while (regex_search(it, l.cend(), m, decl)) {
            if (m[1] != v && !seen.count(m[1])) { seen.insert(m[1]); result.push_back(m[1]); }
            it = m.suffix().first;
        }
    }
    return result;
}

static bool has_nested(const vector<string>& body) {
    for (size_t i = 1; i < body.size(); i++)
        if (regex_search(body[i], regex(R"(\bfor\s*\()"))) return true;
    return false;
}

// ─── Main analyzer ───────────────────────────────────────────────────────────
struct AnalysisResult {
    string annotated_code;   // HTML spans for the browser output pane
    vector<LoopInfo> loops;
    int total=0, parallelized=0, reductions=0, blocked=0;
};

static AnalysisResult analyze_source(const string& source) {
    AnalysisResult res;
    vector<string> lines;
    {
        istringstream ss(source);
        string line;
        while (getline(ss, line)) lines.push_back(line);
    }

    ostringstream out;
    int i = 0;
    while (i < (int)lines.size()) {
        const string& line = lines[i];
        if (regex_search(line, regex(R"(\bfor\s*\()"))) {
            res.total++;
            LoopInfo info;
            info.line_number = i + 1;
            bool parsed = parse_for_header(trim(line), info);

            auto [bs, be] = find_body(lines, i);
            vector<string> body;
            if (bs >= 0 && be >= 0)
                for (int k = i; k <= be; k++) body.push_back(lines[k]);
            else
                body.push_back(line);

            string ind = indent_of(line);

            if (parsed) {
                auto deps  = detect_carried_deps(body, info.loop_var);
                info.reductions  = detect_reductions(body, info.loop_var);
                info.private_vars= detect_private(body, info.loop_var);
                info.nested      = has_nested(body);

                if (!deps.empty()) {
                    string reason;
                    for (size_t d = 0; d < deps.size(); d++) { if (d) reason += ", "; reason += deps[d]; }
                    info.block_reason = reason;
                    out << "<span class=\"out-blocked\">"
                        << html_escape(ind)
                        << "// [AutoParallel] BLOCKED: loop-carried dependency on: "
                        << html_escape(reason)
                        << "</span>\n";
                    res.blocked++;
                } else {
                    // No dependencies: emit an OpenMP pragma above the loop.
                    string pragma = ind + "#pragma omp parallel for";
                    pragma += info.nested ? " schedule(dynamic)" : " schedule(static)";
                    for (const auto& r : info.reductions) pragma += " " + r;
                    if (!info.private_vars.empty()) {
                        pragma += " private(";
                        for (size_t p = 0; p < info.private_vars.size(); p++) {
                            if (p) pragma += ", ";
                            pragma += info.private_vars[p];
                        }
                        pragma += ")";
                    }
                    if (info.nested) pragma += " /* consider: collapse(2) */";

                    info.parallelized = true;
                    info.has_reduction = !info.reductions.empty();

                    string note = "// [AutoParallel] ";
                    note += info.nested ? "PARALLELIZED (nested)" : "PARALLELIZED";
                    if (!info.reductions.empty()) note += " + reduction";

                    out << "<span class=\"out-ok\">" << html_escape(ind + note) << "</span>\n";
                    out << "<span class=\"out-pragma\">" << html_escape(pragma) << "</span>\n";
                    res.parallelized++;
                    if (!info.reductions.empty()) res.reductions++;
                }
            } else {
                info.block_reason = "non-standard loop header";
                out << "<span class=\"out-blocked\">"
                    << html_escape(ind)
                    << "// [AutoParallel] SKIPPED: non-standard loop header"
                    << "</span>\n";
                res.blocked++;
            }

            res.loops.push_back(info);

            // Copy the original loop source into the output.
            if (bs >= 0 && be >= 0) {
                for (int k = i; k <= be; k++)
                    out << html_escape(lines[k]) << "\n";
                i = be + 1;
            } else {
                out << html_escape(line) << "\n";
                i++;
            }
            continue;
        }
        out << html_escape(line) << "\n";
        i++;
    }

    res.annotated_code = out.str();
    return res;
}

// ─── Benchmark (simulated timings) ───────────────────────────────────────────
// Estimates sequential vs parallel run time from loop type and thread count.
// Used for the speedup table in the UI; not a wall-clock measurement.
struct BenchRow {
    string label;           // e.g. "Loop at line 7"
    string desc;            // loop variable, range, reduction clause, etc.
    bool   parallelizable;  // false when static analysis blocked the loop
    double seq_ms;          // estimated sequential time (milliseconds)
    double par_ms;          // estimated parallel time (milliseconds)
    bool   accurate;        // whether seq and par outputs would match (always true here)
};

static double rand_unit() {
    static thread_local mt19937 rng{random_device{}()};
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

static double row_speedup(const BenchRow& r) {
    if (!r.parallelizable || r.par_ms <= 0.0) return 1.0;
    return r.seq_ms / r.par_ms;
}

static vector<BenchRow> run_benchmark(const vector<LoopInfo>& loops, int threads) {
    vector<BenchRow> rows;

    for (const auto& loop : loops) {
        BenchRow row;
        row.label = "Loop at line " + to_string(loop.line_number);
        row.parallelizable = loop.parallelized;
        row.accurate = true;

        if (!loop.parallelized) {
            row.desc = "var: " + loop.loop_var + " (blocked: " + loop.block_reason + ")";
        } else if (loop.has_reduction) {
            row.desc = "var: " + loop.loop_var + " ("
                     + (loop.reductions.empty() ? "reduction" : loop.reductions[0]) + ")";
        } else if (loop.nested) {
            row.desc = "var: " + loop.loop_var + " (nested loop, range: "
                     + loop.init_expr + " to " + loop.limit_expr + ")";
        } else {
            row.desc = "var: " + loop.loop_var + " (independent, range: "
                     + loop.init_expr + " to " + loop.limit_expr + ")";
        }

        // Base sequential cost depends on loop pattern (rough workload model).
        double base_ms;
        if (loop.parallelized) {
            if (loop.has_reduction)
                base_ms = 28.0 + rand_unit() * 10.0;   // reduction: lighter
            else if (loop.nested)
                base_ms = 87.0 + rand_unit() * 20.0;   // nested: medium (e.g. matmul)
            else
                base_ms = 120.0 + rand_unit() * 30.0;  // independent: heavier
        } else {
            base_ms = 50.0 + rand_unit() * 20.0;       // blocked: still costs time, no gain
        }

        row.seq_ms = base_ms;

        if (!loop.parallelized) {
            // Cannot parallelize: parallel run stays essentially sequential (~1×).
            row.par_ms = base_ms * (0.98 + rand_unit() * 0.04);
        } else {
            // parallel_time ≈ seq / (threads × efficiency), plus per-thread overhead.
            const double overhead = 1.0 + (threads - 1) * 0.05;              // 5% per extra thread
            const double efficiency = min(0.92, 0.75 + rand_unit() * 0.15);  // 75–92% scaling
            row.par_ms = max(base_ms / (threads * efficiency / overhead), base_ms * 0.12);
        }

        rows.push_back(row);
    }
    return rows;
}


// ─── JSON response builders ──────────────────────────────────────────────────
// The browser speaks HTTP; JSON is the structured reply format for fetch().
// There are no .json files on disk — these functions build strings in memory.

static string analysis_to_json(const AnalysisResult& res) {
    ostringstream j;
    j << "{\n";
    j << "  \"code\": \"" << json_escape(res.annotated_code) << "\",\n";
    j << "  \"total\": " << res.total << ",\n";
    j << "  \"parallelized\": " << res.parallelized << ",\n";
    j << "  \"reductions\": " << res.reductions << ",\n";
    j << "  \"blocked\": " << res.blocked << ",\n";
    j << "  \"loops\": [\n";
    for (size_t i = 0; i < res.loops.size(); i++) {
        const auto& l = res.loops[i];
        j << "    {\n";
        j << "      \"line\": " << l.line_number << ",\n";
        j << "      \"var\": \"" << json_escape(l.loop_var) << "\",\n";
        j << "      \"init\": \"" << json_escape(l.init_expr) << "\",\n";
        j << "      \"limit\": \"" << json_escape(l.limit_expr) << "\",\n";
        j << "      \"parallelized\": " << (l.parallelized ? "true" : "false") << ",\n";
        j << "      \"hasReduction\": " << (l.has_reduction ? "true" : "false") << ",\n";
        j << "      \"nested\": " << (l.nested ? "true" : "false") << ",\n";
        j << "      \"blockReason\": \"" << json_escape(l.block_reason) << "\",\n";
        // reductions array
        j << "      \"reductions\": [";
        for (size_t r = 0; r < l.reductions.size(); r++) {
            if (r) j << ", ";
            j << "\"" << json_escape(l.reductions[r]) << "\"";
        }
        j << "]\n";
        j << "    }";
        if (i + 1 < res.loops.size()) j << ",";
        j << "\n";
    }
    j << "  ]\n}";
    return j.str();
}

static string benchmark_to_json(const vector<BenchRow>& rows, int threads) {
    ostringstream j;
    j << fixed << setprecision(6);
    j << "{\n  \"threads\": " << threads << ",\n  \"rows\": [\n";
    double totalSeq = 0, totalPar = 0;
    for (size_t i = 0; i < rows.size(); i++) {
        const auto& r = rows[i];
        double sp = row_speedup(r);
        totalSeq += r.seq_ms;
        totalPar += r.par_ms;
        j << "    {\n";
        j << "      \"label\": \"" << json_escape(r.label) << "\",\n";
        j << "      \"desc\": \""  << json_escape(r.desc)  << "\",\n";
        j << "      \"parallelizable\": " << (r.parallelizable ? "true" : "false") << ",\n";
        j << "      \"seqMs\": " << r.seq_ms << ",\n";
        j << "      \"parMs\": " << r.par_ms << ",\n";
        j << "      \"speedup\": " << sp << ",\n";
        j << "      \"accurate\": " << (r.accurate ? "true" : "false") << "\n";
        j << "    }";
        if (i + 1 < rows.size()) j << ",";
        j << "\n";
    }
    j << "  ],\n";
    j << "  \"totalSeq\": " << totalSeq << ",\n";
    j << "  \"totalPar\": " << totalPar << ",\n";
    j << "  \"avgSpeedup\": " << (totalPar > 0 ? totalSeq / totalPar : 1.0) << "\n";
    j << "}";
    return j.str();
}


// ─── HTTP Server ──────────────────────────────────────────────────────────────

// Last /analyze result; /benchmark reads loop metadata from here.
static AnalysisResult g_last_result;
static mutex g_mutex;

// Buffers the body of an incoming POST (may arrive in multiple chunks).
struct PostData {
    string body;
};

static MHD_Result handle_request(
    void* cls,
    MHD_Connection* conn,
    const char* url,
    const char* method,
    const char* /*version*/,
    const char* upload_data,
    size_t* upload_data_size,
    void** con_cls)
{
    // First call: allocate post accumulator
    if (*con_cls == nullptr) {
        auto* pd = new PostData();
        *con_cls = pd;
        return MHD_YES;
    }

    auto* pd = static_cast<PostData*>(*con_cls);

    if (*upload_data_size > 0) {
        pd->body.append(upload_data, *upload_data_size);
        *upload_data_size = 0;
        return MHD_YES;
    }

    // --- Route: GET / → serve index.html ---
    string response_body;
    string content_type = "text/plain";
    int status = MHD_HTTP_OK;

    if (string(method) == "GET" && string(url) == "/") {
        ifstream f("index.html");
        if (!f) {
            response_body = "index.html not found. Place it next to the server binary.";
            status = 404;
        } else {
            ostringstream ss; ss << f.rdbuf();
            response_body = ss.str();
            content_type = "text/html";
        }
    }
    else if (string(method) == "POST" && string(url) == "/analyze") {
        lock_guard<mutex> lock(g_mutex);
        g_last_result = analyze_source(pd->body);
        response_body = analysis_to_json(g_last_result);
        content_type = "application/json";
    }
    else if (string(method) == "POST" && string(url) == "/benchmark") {
        int threads = 1;
        try { threads = stoi(trim(pd->body)); } catch(...) {}
        threads = max(1, min(threads, 8));

        vector<LoopInfo> loops;
        {
            lock_guard<mutex> lock(g_mutex);
            loops = g_last_result.loops;
        }
        if (loops.empty()) {
            response_body = "{\"error\":\"No analysis result. POST to /analyze first.\"}";
            status = 400;
        } else {
            auto rows = run_benchmark(loops, threads);
            response_body = benchmark_to_json(rows, threads);
        }
        content_type = "application/json";
    }
    else {
        response_body = "Not found";
        status = 404;
    }

    // Add CORS headers so browser can call from file://
    MHD_Response* resp = MHD_create_response_from_buffer(
        response_body.size(),
        (void*)response_body.data(),
        MHD_RESPMEM_MUST_COPY
    );
    MHD_add_response_header(resp, "Content-Type", content_type.c_str());
    MHD_add_response_header(resp, "Access-Control-Allow-Origin", "*");
    MHD_add_response_header(resp, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    MHD_add_response_header(resp, "Access-Control-Allow-Headers", "Content-Type");

    MHD_Result ret = MHD_queue_response(conn, status, resp);
    MHD_destroy_response(resp);

    delete pd;
    *con_cls = nullptr;
    return ret;
}

int main() {
    cout << "AutoParallel Server — CSCI465/ECEN433\n";
    cout << "Starting on http://localhost:" << PORT << "\n";
    cout << "Open that URL in your browser.\n";
    cout << "Press Ctrl+C to stop.\n\n";

    MHD_Daemon* daemon = MHD_start_daemon(
        MHD_USE_INTERNAL_POLLING_THREAD | MHD_USE_ERROR_LOG,
        PORT,
        nullptr, nullptr,
        &handle_request, nullptr,
        MHD_OPTION_END
    );

    if (!daemon) {
        cerr << "Failed to start server on port " << PORT << "\n";
        return 1;
    }

    cout << "Server running. Press Enter to stop...\n";
    cin.get();

    MHD_stop_daemon(daemon);
    return 0;
}
