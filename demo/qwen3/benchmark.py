#!/usr/bin/env python3

import subprocess
import re
import os

def run_mirage_benchmark():
    cmd = [
        "python", "demo.py", 
        "--model", "Qwen/Qwen3-8B", 
        "--use-mirage"
    ]
    
    env = {
        **os.environ,
        'PATH': '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', ''),
        'LD_LIBRARY_PATH': '/usr/local/cuda-12.8/lib64:../../build/abstract_subexpr/release:../../build/formal_verifier/release:' + os.environ.get('LD_LIBRARY_PATH', ''),
        'CUDA_HOME': '/usr/local/cuda-12.8',
        'MIRAGE_HOME': '../..'
    }
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        output = result.stdout
        per_token_match = re.search(r'per-token latency ([\d.]+) ms', output)
        
        if per_token_match:
            per_token_ms = float(per_token_match.group(1))
            throughput_tps = 1000.0 / per_token_ms
            
            return {
                'per_token_latency_ms': per_token_ms,
                'throughput_tps': throughput_tps
            }
    
    print("Error running benchmark:", result.stderr)
    return None

def main():
    print("Running Mirage MPK benchmark...")
    results = run_mirage_benchmark()
    
    if not results:
        print("Benchmark failed")
        return
    
    per_token_ms = results['per_token_latency_ms']
    throughput_tps = results['throughput_tps']
    
    # Calculate latency for standard comparison (assume ~30 tokens like your SGLang results)
    comparison_latency_s = (per_token_ms * 30) / 1000
    
    print("\nAndre MegaKernel Persistent Kernel Baseline")
    print("=" * 40)
    print(f"Latency: {comparison_latency_s:.2f}s")
    print(f"Throughput: {throughput_tps:.1f}tps")
    print(f"Per-token: {per_token_ms:.1f}ms")
    print(f"Context: 32K tokens")
    print()
    
    print("COMPARISON WITH OPENROUTER")
    print("=" * 60)
    print(f"{'Provider':<25} {'Latency':<10} {'Throughput'}")
    print("-" * 60)
    print(f"{'NovitaAI':<25} {'0.72s':<10} {'64.28 tps'}")
    print(f"{'Andre SGLang':<25} {'0.30s':<10} {'42.8 tps'}")
    print(f"{'Andre MegaKernel MPK':<25} {comparison_latency_s:<10.2f}s {throughput_tps:.1f} tps")
    print("-" * 60)

if __name__ == "__main__":
    main()
