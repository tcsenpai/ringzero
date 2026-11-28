# demo
- 2025-08-10 run /demo/stats.py with python3; {"people": 5, "mean_age": 38.8, ...}
written /demo/report.json
- 2026-11-06 run hello.lua in /demo; the 3
brown 1
dog 1
fib(20) = 6765 in 2 ms
pcall: false something went wrong
written by Lua 5.4
- 2026-11-08 print /demo/data.json with jq
- 2026-11-08 {
  "os": "ringzero",
  "version": "0.8",
  "models": [
    { "name": "Qwen3.5-2B", "quant": "q8_0", "size_mb": 2210 },
    { "name": "Qwen3-Embedding-0.6B", "quant": "q8_0", "size_mb": 640 }
  ],
  "apps": ["coreutils", "ripgrep", "jq", "sqlite3", "lua", "python3", "qjs", "kibi"]
}
