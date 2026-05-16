# AutoParallel

**Automatic OpenMP parallelization for C++ `for` loops** — a static analyzer with a web UI and live OpenMP benchmarks.

Built for **CSCI465 / ECEN433 — Introduction to Parallel Computing** at Nile University (Spring 2026).

## Overview

AutoParallel reads sequential C/C++ source code, finds `for` loops, and decides whether each loop can safely run in parallel. For parallelizable loops it injects the appropriate OpenMP pragmas (`#pragma omp parallel for`) with `reduction`, `private`, and `schedule` clauses. Loops with loop-carried dependencies are flagged and left sequential.

The project pairs a **C++ HTTP server** (analysis + real timing) with a **browser front end** so you can paste code, inspect annotated output, and measure speedup without leaving the page.

## Features

- **Static loop analysis** — parses standard `for` headers, scans loop bodies, detects dependencies
- **OpenMP pragma generation** — `parallel for`, `reduction(+:…)`, `reduction(max:…)`, `private(…)`, `schedule(static|dynamic)`
- **Dependency blocking** — detects patterns like `a[i-1]` on the right-hand side of assignments
- **Reduction detection** — `+=`, `-=`, `*=`, and max/min accumulator idioms
- **Nested loop hints** — suggests `collapse(2)` for nested loops
- **Live benchmark** — runs representative OpenMP workloads and reports sequential vs parallel wall time and speedup (1–8 threads)

## Architecture

```
┌─────────────────┐     POST /analyze      ┌──────────────────┐
│   index.html    │ ─────────────────────► │   server.cpp     │
│   (browser UI)  │     POST /benchmark    │   libmicrohttpd  │
│                 │ ◄───────────────────── │   + OpenMP       │
└─────────────────┘         JSON           └──────────────────┘
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves `index.html` |
| `/analyze` | POST | Body: C++ source → JSON with annotated HTML, loop stats, dependency report |
| `/benchmark` | POST | Body: thread count (1–8) → JSON with per-loop timings and speedup (requires prior `/analyze`) |

## Team

| Name | ID |
|------|-----|
| Peter Yakoub Younan | 221000217 |
| Sotir Usama Latif | 221000694 |
| Omar Mohamed Sawan | 221000756 |
| Omar Alaa Hassan | 202002417 |
| Sondos Mohamed | 202002250 |

## Prerequisites

- **C++17** compiler with **OpenMP** (`g++` recommended)
- **[GNU libmicrohttpd](https://www.gnu.org/software/libmicrohttpd/)** development headers

### Installing libmicrohttpd

**Ubuntu / Debian**

```bash
sudo apt install libmicrohttpd-dev
```

**MSYS2 / MinGW (Windows)**

```bash
pacman -S mingw-w64-x86_64-libmicrohttpd
```

**macOS (Homebrew)**

```bash
brew install libmicrohttpd
```

## Build & run

From the project directory (same folder as `server.cpp` and `index.html`):

**Linux / macOS**

```bash
g++ -O2 -fopenmp -std=c++17 -o server server.cpp -lmicrohttpd
./server
```

**Windows (MinGW / MSYS2)**

```bash
g++ -O2 -fopenmp -std=c++17 -o server.exe server.cpp -lmicrohttpd
./server.exe
```

Then open **[http://localhost:8080](http://localhost:8080)** in your browser.

1. Paste C++ code (or use an example) and click **Analyze**
2. Review injected pragmas and the dependency report
3. Choose a thread count and click **Run Benchmark** for timing results

Press **Enter** in the server terminal to stop.

## How analysis works

1. **Parse** — locate `for` loops and extract loop variable, init, and limit from the header
2. **Dependency check** — flag loop-carried deps when the loop index appears in subscript expressions on assignment RHS (e.g. `arr[i-1]`)
3. **Pattern detection** — reductions, private scalars declared in the body, nested inner loops
4. **Pragma injection** — emit OpenMP directives above each safe loop; annotate blocked loops with reasons

### What gets parallelized

| Pattern | Result |
|---------|--------|
| Independent `arr[i]` access | `#pragma omp parallel for schedule(static)` |
| Reduction (`sum += …`, max/min) | Adds `reduction(+:sum)` etc. |
| Nested inner `for` | `schedule(dynamic)` + `collapse(2)` hint |
| Loop-carried dependency (`a[i-1]`) | **Blocked** — no pragma |

## Example

**Input**

```cpp
for (int i = 0; i < n; i++) {
    b[i] = a[i] * 2 + 1;
}
```

**Output**

```cpp
// [AutoParallel] PARALLELIZED
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; i++) {
    b[i] = a[i] * 2 + 1;
}
```

## Limitations

This is an **educational static analyzer**, not a production compiler pass:

- Only recognizes a subset of `for` loop header syntax
- Dependency analysis is regex-based (no full AST or pointer analysis)
- Benchmark workloads are **representative** stand-ins mapped from loop type, not a recompile of your exact pasted code
- No support for `while` loops, function calls with side effects, or complex aliasing

See `Parallel_Final_IEEE_Paper.pdf` for the full project write-up.

## Project structure

```
Automatic_Parallelization-2/
├── server.cpp          # HTTP server, analyzer, OpenMP benchmarks
├── index.html          # Web UI
├── README.md           # This file
└── Parallel_Final_IEEE_Paper.pdf
```

## License

Academic project — Nile University, Spring 2026.
