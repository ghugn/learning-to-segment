# Multi-Scale Empirical SOTA Benchmark: CVRP 1k, 2k, 3k

- **Time Budget per Method**: 20.0s
- **Datasets**: `vrp1000_test_seed1234.pkl`, `vrp2000_test_seed1234.pkl`, Synthetic CVRP3k

| Scale | Method | Solution Cost (Obj) | Gap vs HGS (%) | Execution Time | Search Space Reduction |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **CVRP-1000** | **PyVRP (HGS Vidal 2022)** | 39.910 | 0.00% | 22.32s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | 40.170 | +0.65% | 24.26s | 0.0% (Full Graph) |
| | **L2Seg-SYN-LNS (Our FSTA)** | **42.430** | **+6.31%** | **20.05s** | **-78.4%** |
| **CVRP-2000** | **PyVRP (HGS Vidal 2022)** | 55.399 | 0.00% | 35.03s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | 60.200 | +8.67% | 27.44s | 0.0% (Full Graph) |
| | **L2Seg-SYN-LNS (Our FSTA)** | **58.527** | **+5.65%** | **21.93s** | **-85.8%** |
| **CVRP-3000** | **PyVRP (HGS Vidal 2022)** | 67.395 | 0.00% | 41.81s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | - | - | - | OOM / Unsupported Scale |
| | **L2Seg-SYN-LNS (Our FSTA)** | **69.291** | **+2.81%** | **20.10s** | **-86.4%** |
