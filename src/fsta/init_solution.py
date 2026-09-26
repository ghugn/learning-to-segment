import math
from typing import List
import numpy as np
from .types import CVRPInstance
from .recovery import validate_cvrp_solution


def build_angular_sweep_solution(instance: CVRPInstance) -> List[List[int]]:
    """
    Construct an initial feasible CVRP solution using the Angular Sweep heuristic.
    As described in Appendix D.1 of the paper (inspired by Li et al. 2021):
    - Customers are ordered by polar angle w.r.t. the depot.
    - Accumulated into routes sequentially until vehicle capacity C is reached.
    """
    depot_coord = instance.coords[0]
    num_cust = instance.num_customers
    customer_indices = list(range(1, num_cust + 1))

    # Calculate angles w.r.t. depot
    angles = []
    for c in customer_indices:
        dx = instance.coords[c, 0] - depot_coord[0]
        dy = instance.coords[c, 1] - depot_coord[1]
        angles.append(math.atan2(dy, dx))

    # Sort customers by polar angle
    sorted_customers = [c for _, c in sorted(zip(angles, customer_indices))]

    # Group into capacity-feasible routes
    routes: List[List[int]] = []
    curr_route = [0]
    curr_load = 0.0

    for c in sorted_customers:
        d = instance.demands[c]
        if curr_load + d <= instance.capacity:
            curr_route.append(c)
            curr_load += d
        else:
            curr_route.append(0)
            routes.append(curr_route)
            curr_route = [0, c]
            curr_load = d

    if len(curr_route) > 1:
        curr_route.append(0)
        routes.append(curr_route)

    is_valid, msg = validate_cvrp_solution(instance, routes)
    if not is_valid:
        raise RuntimeError(f"Generated initial solution is invalid: {msg}")

    return routes
