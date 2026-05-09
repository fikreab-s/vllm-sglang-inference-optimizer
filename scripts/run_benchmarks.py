"""Inference engine benchmark simulation."""
import json, numpy as np, argparse
from pathlib import Path
np.random.seed(42)

ENGINES = {
    "vllm": {"ttft_ms": 28, "decode_tps": 85, "batch32_tps": 1840, "vram_gb": 3.2},
    "sglang": {"ttft_ms": 25, "decode_tps": 92, "batch32_tps": 2100, "vram_gb": 3.4},
    "llamacpp": {"ttft_ms": 65, "decode_tps": 58, "batch32_tps": 420, "vram_gb": 1.2},
    "onnx_rt": {"ttft_ms": 32, "decode_tps": 78, "batch32_tps": 1650, "vram_gb": 2.8},
}

def benchmark_engine(name, config, n_requests=100):
    ttfts = np.random.normal(config["ttft_ms"], config["ttft_ms"]*0.15, n_requests)
    decodes = np.random.normal(config["decode_tps"], config["decode_tps"]*0.1, n_requests)
    return {"engine": name, "ttft_p50": round(np.median(ttfts), 1), "ttft_p99": round(np.percentile(ttfts, 99), 1),
            "decode_tps_mean": round(np.mean(decodes), 1), "batch32_tps": config["batch32_tps"],
            "vram_gb": config["vram_gb"]}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output_dir", default="outputs"); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    results = [benchmark_engine(n, c) for n, c in ENGINES.items()]
    with open(out / "benchmark_results.json", "w") as f: json.dump(results, f, indent=2)
    print("✅ Inference Engine Benchmarks\n")
    print(f"  {'Engine':<12} {'TTFT p50':<10} {'TTFT p99':<10} {'Decode':<10} {'Batch32':<10} {'VRAM':<8}")
    print("  " + "-"*60)
    for r in results:
        print(f"  {r['engine']:<12} {r['ttft_p50']:<10.1f} {r['ttft_p99']:<10.1f} {r['decode_tps_mean']:<10.1f} {r['batch32_tps']:<10} {r['vram_gb']:<8.1f}")
    best = max(results, key=lambda x: x["decode_tps_mean"])
    print(f"\n  Fastest decode: {best['engine']} ({best['decode_tps_mean']} tok/s)")

if __name__ == "__main__": main()
