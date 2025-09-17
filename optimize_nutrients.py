"""Обёртка, делегирующая вызовы эталонной реализации оптимизатора."""

from __future__ import annotations

from GENESIS_optimize_nutrients import (
    DEFAULT_CSV_PATH,
    NUTRIENT_ORDER,
    OptimizationProduct,
    compute_calories,
    load_product_database,
    main,
    optimise_diet,
    optimize_from_payload,
    rmse,
)

__all__ = [
    "DEFAULT_CSV_PATH",
    "NUTRIENT_ORDER",
    "OptimizationProduct",
    "compute_calories",
    "load_product_database",
    "optimise_diet",
    "optimize_from_payload",
    "rmse",
]


if __name__ == "__main__":
    main()
