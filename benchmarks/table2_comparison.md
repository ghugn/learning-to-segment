# Table 2: Performance comparisons of L2Seg-SYN against Baselines on Benchmark CVRP Instances

*The gap % (lower the better) is with respect to the performance of HGS (Vidal, 2022).*

| Category | Methods | CVRP1k Obj | CVRP1k Gap% | CVRP1k Time | CVRP2k Obj | CVRP2k Gap% | CVRP2k Time | CVRP5k Obj | CVRP5k Gap% | CVRP5k Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Heuristics** | HGS (Vidal, 2022) | 41.20 | 0.00% | 5m | 57.20 | 0.00% | 5m | 126.20 | 0.00% | 5m |
|  | LKH-3 (Helsgaun, 2017) | 42.98 | +4.32% | 6.6m | 57.94 | +1.29% | 11.4m | 175.70 | +39.22% | 2.5m |
|  | LNS (Shaw, 1998) | 42.44 | +3.01% | 2.5m | 57.62 | +0.73% | 4.0m | 126.58 | +0.30% | 5.0m |
| **Neural End-to-End** | BQ (Drakulic et al., 2023) | 44.17 | +7.21% | 55s | 62.59 | +9.42% | 3m | 139.80 | +10.78% | 45m |
|  | LEHD (Luo et al., 2023) | 43.96 | +6.70% | 1.3m | 61.58 | +7.66% | 9.5m | 138.20 | +9.51% | 3h |
|  | ELG (Gao et al., 2024) | 43.58 | +5.78% | 15.6m | - | - | - | - | - | - |
|  | ICAM (Zhou et al., 2024) | 43.07 | +4.54% | 26s | 61.34 | +7.24% | 3.7m | 136.90 | +8.48% | 50m |
|  | L2R (Zhou et al., 2025a) | 44.20 | +7.28% | 34.2s | - | - | - | 131.10 | +3.88% | 1.8m |
|  | SIL (Luo et al., 2024) | 42.00 | +1.94% | 1.3m | 57.10 | -0.17% | 2.4m | 123.10 | -2.52% | 5.9m |
| **Decomposition / Large-Scale** | TAM(LKH-3) (Hou et al., 2023) | 46.30 | +12.38% | 4m | 64.80 | +13.29% | 9.6m | 144.60 | +14.58% | 35m |
|  | GLOP-G(LKH-3) (Ye et al., 2024) | 45.90 | +11.41% | 2m | 63.02 | +10.52% | 2.5m | 140.40 | +11.25% | 8m |
|  | UDC (Zheng et al., 2024) | 43.00 | +4.37% | 1.2h | 60.01 | +4.90% | 2.15h | 136.70 | +8.32% | 16m |
|  | L2D (Li et al., 2021) | 42.07 | +2.11% | 2.5m | 57.44 | +0.42% | 4.2m | 126.48 | +0.22% | 5.3m |
|  | NDS (Hottung et al., 2025) | 41.16 | -0.01% | 2.5m | 56.11 | -1.91% | 4m | - | - | - |
| **Proposed L2Seg Framework** | **L2Seg-SYN-LKH-3** | 41.42 | +0.53% | 2.5m | 56.37 | -1.45% | 4.4m | 122.34 | -3.16% | 5.1m |
|  | **L2Seg-SYN-LNS** | 41.36 | +0.39% | 2.5m | 56.08 | -1.96% | 4.1m | 121.96 | -3.48% | 5.1m |
|  | **L2Seg-SYN-L2D** | 41.23 | +0.07% | 2.5m | 56.05 | -2.01% | 4.1m | 121.87 | -3.55% | 5.1m |