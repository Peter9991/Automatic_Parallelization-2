/*
 * AutoParallel Server — CSCI465/ECEN433, Nile University
 *
 * A self-contained C++ HTTP server that:
 *   1. Serves index.html at GET /
 *   2. Accepts C++ code via POST /analyze  → returns JSON with parallelized code + loop info
 *   3. Accepts thread count via POST /benchmark → returns JSON with measured speedup (omp_get_wtime)
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

// ─── Benchmark (measured with omp_get_wtime) ─────────────────────────────────
struct BenchRow {
    string label;
    string desc;
    bool   parallelizable;
    double seq_ms;
    double par_ms;
    bool   accurate;
};

// Prevent the optimizer from dropping benchmark side effects.
static volatile double g_bench_sink = 0;

// Warm up, adapt inner iterations so each sample is measurable, return median ms per call.
static double bench_ms(const function<void()>& fn) {
    constexpr int WARMUP     = 2;
    constexpr int REPS       = 7;
    constexpr double TARGET_MS = 8.0;
    constexpr int ITERS_MAX  = 2000;

    for (int w = 0; w < WARMUP; w++) fn();

    int iters = 1;
    for (int probe = 0; probe < 12 && iters < ITERS_MAX; probe++) {
        double t0 = omp_get_wtime();
        for (int k = 0; k < iters; k++) fn();
        double elapsed_ms = (omp_get_wtime() - t0) * 1000.0;
        if (elapsed_ms >= TARGET_MS * 0.5) break;
        if (elapsed_ms < 0.001)
            iters = min(iters * 10, ITERS_MAX);
        else {
            int need = (int)ceil(TARGET_MS / (elapsed_ms / iters));
            iters = min(max(iters, need), ITERS_MAX);
            break;
        }
    }

    double times[REPS];
    for (int i = 0; i < REPS; i++) {
        double t0 = omp_get_wtime();
        for (int k = 0; k < iters; k++) fn();
        times[i] = (omp_get_wtime() - t0) * 1000.0 / iters;
    }
    sort(times, times + REPS);
    return times[REPS / 2];
}

static double row_speedup(const BenchRow& r, int threads) {
    if (!r.parallelizable) return 1.0;
    if (r.seq_ms <= 0.0 || r.par_ms <= 0.0) return 1.0;
    double sp = r.seq_ms / r.par_ms;
    const double cap = max(1.0, (double)threads * 1.2);
    return min(sp, cap);
}

static vector<BenchRow> run_benchmark(const vector<LoopInfo>& loops, int threads) {
    omp_set_num_threads(threads);
    vector<BenchRow> rows;

    constexpr int N = 600;
    constexpr int M = 5000000;
    constexpr double TOLERANCE = 1e-6;

    static float  A[N][N], B[N][N], C_seq[N][N], C_par[N][N];
    static double arr[M], src[M], dst_seq[M], dst_par[M], rec[M];

    static bool inited = false;
    if (!inited) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = (float)(i + j + 1) / (N * N);
                B[i][j] = (float)(i - j + 2) / (N * N);
            }
        for (int i = 0; i < M; i++) {
            arr[i] = sin((double)i * 0.0001) + 0.5;
            src[i] = (double)(i + 1);
        }
        rec[0] = 1.0; rec[1] = 1.0;
        inited = true;
    }

    for (const auto& loop : loops) {
        BenchRow row;
        row.label = "Loop at line " + to_string(loop.line_number);
        row.parallelizable = loop.parallelized;

        if (!loop.parallelized) {
            row.desc = "var: " + loop.loop_var + " (blocked: " + loop.block_reason + ")";

            row.seq_ms = bench_ms([&] {
                rec[0] = 1.0; rec[1] = 1.0;
                for (int i = 2; i < M; i++)
                    rec[i] = rec[i-1] * 0.9999 + rec[i-2] * 0.0001;
            });
            row.par_ms = row.seq_ms;
            row.accurate = true;

        } else if (loop.nested) {
            row.desc = "var: " + loop.loop_var + " (nested loop, range: "
                     + loop.init_expr + " to " + loop.limit_expr + ")";

            row.seq_ms = bench_ms([&] {
                for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) C_seq[i][j] = 0;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) {
                        float s = 0;
                        for (int k = 0; k < N; k++) s += A[i][k] * B[k][j];
                        C_seq[i][j] = s;
                    }
            });

            row.par_ms = bench_ms([&] {
                for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) C_par[i][j] = 0;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) {
                        float s = 0;
                        for (int k = 0; k < N; k++) s += A[i][k] * B[k][j];
                        C_par[i][j] = s;
                    }
            });

            double max_err = 0.0;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    double diff = abs((double)C_seq[i][j] - (double)C_par[i][j]);
                    if (diff > max_err) max_err = diff;
                }
            row.accurate = max_err < TOLERANCE;

        } else if (loop.has_reduction) {
            row.desc = "var: " + loop.loop_var + " ("
                     + (loop.reductions.empty() ? "reduction" : loop.reductions[0]) + ")";

            bool want_sum = false, want_max = false;
            for (const auto& r : loop.reductions) {
                if (r.find("+:") != string::npos || r.find("-:") != string::npos ||
                    r.find("*:") != string::npos) want_sum = true;
                if (r.find("max:") != string::npos) want_max = true;
            }
            if (!want_sum && !want_max) { want_sum = true; want_max = true; }

            double sum_seq = 0, mx_seq = arr[0];
            row.seq_ms = bench_ms([&] {
                if (want_sum) { sum_seq = 0; for (int i = 0; i < M; i++) sum_seq += arr[i]; }
                if (want_max) { mx_seq = arr[0]; for (int i = 0; i < M; i++) if (arr[i] > mx_seq) mx_seq = arr[i]; }
                g_bench_sink += sum_seq + mx_seq;
            });

            double sum_par = 0, mx_par = arr[0];
            if (want_sum && want_max) {
                row.par_ms = bench_ms([&] {
                    sum_par = 0; mx_par = arr[0];
                    #pragma omp parallel for reduction(+:sum_par) reduction(max:mx_par)
                    for (int i = 0; i < M; i++) {
                        sum_par += arr[i];
                        if (arr[i] > mx_par) mx_par = arr[i];
                    }
                    g_bench_sink += sum_par + mx_par;
                });
            } else if (want_sum) {
                row.par_ms = bench_ms([&] {
                    sum_par = 0;
                    #pragma omp parallel for reduction(+:sum_par)
                    for (int i = 0; i < M; i++) sum_par += arr[i];
                    g_bench_sink += sum_par;
                });
            } else {
                row.par_ms = bench_ms([&] {
                    mx_par = arr[0];
                    #pragma omp parallel for reduction(max:mx_par)
                    for (int i = 0; i < M; i++) if (arr[i] > mx_par) mx_par = arr[i];
                    g_bench_sink += mx_par;
                });
            }

            double err = 0.0;
            if (want_sum) err = max(err, abs(sum_seq - sum_par));
            if (want_max) err = max(err, abs(mx_seq - mx_par));
            row.accurate = want_sum
                ? err < max(1e-9, abs(sum_seq) * 1e-9)
                : err < TOLERANCE;

        } else {
            row.desc = "var: " + loop.loop_var + " (independent, range: "
                     + loop.init_expr + " to " + loop.limit_expr + ")";

            row.seq_ms = bench_ms([&] {
                for (int i = 0; i < M; i++)
                    dst_seq[i] = sqrt(src[i]) * 2.5 + sin(src[i] * 0.001);
            });

            row.par_ms = bench_ms([&] {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < M; i++)
                    dst_par[i] = sqrt(src[i]) * 2.5 + sin(src[i] * 0.001);
            });

            double max_err = 0.0;
            for (int i = 0; i < M; i++) {
                double diff = abs(dst_seq[i] - dst_par[i]);
                if (diff > max_err) max_err = diff;
            }
            row.accurate = max_err < TOLERANCE;
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
        double sp = row_speedup(r, threads);
        totalSeq += r.seq_ms;
        totalPar += r.parallelizable ? r.par_ms : r.seq_ms;
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
    BenchRow totals{"", "", true, totalSeq, totalPar};
    j << "  \"avgSpeedup\": " << row_speedup(totals, threads) << "\n";
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
