# Live Empirical Head-to-Head Benchmark on CVRP-1000

- **Dataset**: `vrp1000_test_seed1234.pkl` (exact test set from NDS / Kool et al.)
- **Evaluated Instances**: 1
- **Time Budget per Instance**: 10.0s

| Method | Implementation | Cost (Obj) | Gap vs HGS (%) | Execution Time | Graph Search Space Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PyVRP (HGS Vidal 2022)** | Official C++ Engine | **40.178** | **0.00%** | 12.38s | 0.0% (Full Graph) |
| **NDS (Hottung et al. 2022)** | Official C++ / PyTorch Repo | **40.400** | **+0.55%** | 15.12s | 0.0% (Full Graph) |
| **L2Seg-SYN-LNS** | L2Seg AI + FSTA + Focused LNS | **167.215** | **+316.19%** | 12.50s | **-77.4%** |
