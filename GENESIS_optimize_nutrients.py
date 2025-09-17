"""Эталонный оптимизатор рациона."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


NUTRIENT_ORDER: tuple[str, ...] = (
    "proteins",
    "saturated",
    "unsaturated",
    "simple",
    "complex",
    "soluble",
    "insoluble",
)
DEFAULT_CSV_PATH = "Nutrients DB.csv"


@dataclass
class OptimizationProduct:
    """Представление продукта для расчёта рациона."""

    name: str
    nutrients: Dict[str, float]
    step: float
    max_weight: float
    fix_weight: bool = False
    fixed_weight: float = 0.0

    def resolved_fixed_weight(self) -> float:
        """Вернуть вес фиксированного продукта в пределах допустимого диапазона."""

        if not self.fix_weight or self.max_weight <= 0:
            return 0.0
        weight = self.fixed_weight if self.fixed_weight > 0 else self.max_weight
        return min(self.max_weight, max(0.0, weight))


def compute_calories(nutrients: Iterable[float]) -> float:
    """Calculate calories from macronutrient amounts."""

    protein, fat, carbs = nutrients
    return protein * 4 + fat * 9 + carbs * 4


def _macro_tuple_from_vector(vector: Sequence[float], nutrient_keys: Sequence[str]) -> tuple[float, float, float]:
    index_map = {key: idx for idx, key in enumerate(nutrient_keys)}

    def _value(key: str) -> float:
        idx = index_map.get(key)
        if idx is None:
            return 0.0
        try:
            return float(vector[idx])
        except (TypeError, ValueError):
            return 0.0

    protein = _value("proteins")
    fats = _value("saturated") + _value("unsaturated")
    carbs = sum(_value(key) for key in ("simple", "complex", "soluble", "insoluble"))
    return protein, fats, carbs


def rmse(
    predicted: Iterable[float],
    target: Iterable[float],
    *,
    nutrient_keys: Optional[Iterable[str]] = None,
    calorie_weight: Optional[float] = None,
    predicted_calories: Optional[float] = None,
    target_calories: Optional[float] = None,
) -> float:
    """Calculate root mean squared error between nutrient vectors."""

    predicted_arr = np.array(predicted, dtype=float)
    target_arr = np.array(target, dtype=float)

    diffs = predicted_arr - target_arr

    if calorie_weight is not None:
        try:
            cal_weight = float(calorie_weight)
        except (TypeError, ValueError):
            cal_weight = 0.0

        if math.isfinite(cal_weight) and cal_weight != 0.0:
            cal_weight = abs(cal_weight)
            if predicted_calories is None or target_calories is None:
                if nutrient_keys is None:
                    raise ValueError("nutrient_keys are required to compute calories automatically")
                protein_pred, fat_pred, carb_pred = _macro_tuple_from_vector(predicted_arr, nutrient_keys)
                protein_target, fat_target, carb_target = _macro_tuple_from_vector(target_arr, nutrient_keys)
                if predicted_calories is None:
                    predicted_calories = compute_calories((protein_pred, fat_pred, carb_pred))
                if target_calories is None:
                    target_calories = compute_calories((protein_target, fat_target, carb_target))

            calories_diff = (float(predicted_calories or 0.0) - float(target_calories or 0.0)) * cal_weight
            diffs = np.append(diffs, calories_diff)

    return float(np.sqrt(np.mean(diffs**2)))


def _parse_float(value: Optional[str]) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _residual_share_sequence(start_fraction: float) -> List[float]:
    """Построить последовательность долей остатка с шагом 1% до 100%."""

    try:
        fraction = float(start_fraction or 0.0)
    except (TypeError, ValueError):
        fraction = 0.0

    if not math.isfinite(fraction):
        fraction = 0.0

    fraction = max(0.0, min(1.0, fraction))

    start_percent = fraction * 100.0
    shares: List[float] = []
    seen: set[float] = set()

    def add_share(value: float) -> None:
        key = round(value, 6)
        if key not in seen:
            seen.add(key)
            shares.append(value)

    add_share(fraction)
    next_percent = math.ceil(start_percent)
    for percent in range(int(next_percent), 101):
        share_value = percent / 100.0
        if share_value < fraction:
            continue
        add_share(share_value)

    if not shares:
        shares.append(1.0)

    return shares


def load_product_database(csv_path: str = DEFAULT_CSV_PATH) -> Dict[str, Dict[str, float]]:
    """Загрузить базу продуктов из CSV и вернуть словарь с ключами нутриентов."""

    products: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_name = (row.get("Продукт") or "").strip().replace("\ufeff", "")
            if not raw_name:
                continue
            products[raw_name] = {
                "proteins": _parse_float(row.get("Белки")),
                "saturated": _parse_float(row.get("Насыщенные")),
                "unsaturated": _parse_float(row.get("НЕнасыщенные")),
                "simple": _parse_float(row.get("Простые")),
                "complex": _parse_float(row.get("Сложные перевариваемые")),
                "soluble": _parse_float(row.get("Растворимая")),
                "insoluble": _parse_float(row.get("Нерастворимая")),
                "kcal": _parse_float(row.get("ККал")),
            }
    return products


def _quantize_weight(
    weight: float,
    step: float,
    max_weight: float,
    *,
    min_weight: float = 0.0,
) -> float:
    if max_weight <= 0:
        return 0.0

    effective = max(0.0, min(weight, max_weight))
    if step <= 0:
        quantized = effective
    else:
        steps = round(effective / step)
        quantized = steps * step
        if quantized > max_weight:
            quantized = math.floor(max_weight / step) * step
        quantized = max(0.0, quantized)

    if min_weight > 0:
        min_weight = max(0.0, min(min_weight, max_weight))
        if step > 0:
            min_steps = max(1, math.ceil(min_weight / step))
            min_candidate = min(max_weight, min_steps * step)
            min_weight = min_candidate if min_candidate > 0 else min(max_weight, step)
        if quantized < min_weight:
            quantized = min_weight

    return max(0.0, min(quantized, max_weight))


def _normalise_targets(raw: Dict[str, float]) -> Dict[str, float]:
    result = {}
    for key in (*NUTRIENT_ORDER, "calories"):
        try:
            result[key] = float(raw.get(key, 0) or 0)
        except (TypeError, ValueError):
            result[key] = 0.0
    return result


def _prepare_product(entry: Dict[str, object], product_db: Dict[str, Dict[str, float]]) -> OptimizationProduct:
    if not isinstance(entry, dict):
        raise ValueError("Некорректные данные продукта.")
    name = str(entry.get("name") or "").strip()
    if not name:
        raise ValueError("Не указано название продукта.")

    nutrients_data = entry.get("nutrients")
    if isinstance(nutrients_data, dict):
        nutrients = {
            "proteins": _parse_float(nutrients_data.get("proteins")),
            "saturated": _parse_float(nutrients_data.get("saturated")),
            "unsaturated": _parse_float(nutrients_data.get("unsaturated")),
            "simple": _parse_float(nutrients_data.get("simple")),
            "complex": _parse_float(nutrients_data.get("complex")),
            "soluble": _parse_float(nutrients_data.get("soluble")),
            "insoluble": _parse_float(nutrients_data.get("insoluble")),
            "kcal": _parse_float(nutrients_data.get("kcal")),
        }
    else:
        if name not in product_db:
            raise ValueError(f'Продукт "{name}" отсутствует в базе.')
        nutrients = product_db[name]

    step = _parse_float(entry.get("step"))
    max_weight = max(0.0, _parse_float(entry.get("max_weight")))
    fix_weight = bool(entry.get("fix_weight"))
    fixed_weight = _parse_float(entry.get("fixed_weight"))
    if fix_weight and fixed_weight <= 0:
        fixed_weight = max_weight

    return OptimizationProduct(
        name=name,
        nutrients=nutrients,
        step=max(0.0, step),
        max_weight=max_weight,
        fix_weight=fix_weight,
        fixed_weight=fixed_weight,
    )


def _nutrient_vector(product: OptimizationProduct) -> np.ndarray:
    return np.array([product.nutrients.get(key, 0.0) / 100.0 for key in NUTRIENT_ORDER], dtype=float)


def _calorie_value(product: OptimizationProduct) -> float:
    kcal_value = product.nutrients.get("kcal")
    if kcal_value is None:
        fats = product.nutrients.get("saturated", 0.0) + product.nutrients.get("unsaturated", 0.0)
        carbs = (
            product.nutrients.get("simple", 0.0)
            + product.nutrients.get("complex", 0.0)
            + product.nutrients.get("soluble", 0.0)
            + product.nutrients.get("insoluble", 0.0)
        )
        kcal_value = compute_calories((product.nutrients.get("proteins", 0.0), fats, carbs))
    return float(kcal_value or 0.0) / 100.0


def _build_product_arrays(
    products: List[OptimizationProduct],
    share_fraction: float,
    allow_zero_weights: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    nutrient_matrix = np.vstack([_nutrient_vector(product) for product in products])
    calorie_per_gram = np.array([_calorie_value(product) for product in products], dtype=float)
    steps = np.array([product.step for product in products], dtype=float)
    fixed_flags = np.array([product.fix_weight for product in products], dtype=bool)
    fixed_weights = np.zeros(len(products), dtype=float)
    min_bounds = np.zeros(len(products), dtype=float)
    max_bounds = np.zeros(len(products), dtype=float)

    for idx, product in enumerate(products):
        if fixed_flags[idx]:
            weight = product.resolved_fixed_weight()
            fixed_weights[idx] = _quantize_weight(weight, product.step, product.max_weight, min_weight=weight)
            min_bounds[idx] = fixed_weights[idx]
            max_bounds[idx] = fixed_weights[idx]
            continue

        scaled_max = product.max_weight * share_fraction if share_fraction > 0 else 0.0
        if share_fraction >= 1.0:
            scaled_max = product.max_weight
        scaled_max = min(product.max_weight, scaled_max)
        scaled_max = max(0.0, scaled_max)

        if allow_zero_weights:
            min_weight = 0.0
        else:
            min_candidate = product.step if product.step > 0 else min(product.max_weight, 1.0)
            if min_candidate <= 0:
                min_candidate = min(product.max_weight, 1.0)
            min_weight = min_candidate

        if scaled_max <= 0:
            min_weight = 0.0
        elif min_weight > scaled_max:
            min_weight = scaled_max

        min_bounds[idx] = max(0.0, min_weight)
        max_bounds[idx] = max(0.0, scaled_max)

    return (
        nutrient_matrix,
        calorie_per_gram,
        steps,
        fixed_flags,
        fixed_weights,
        min_bounds,
        max_bounds,
    )


def _initialise_weights(
    rng: random.Random,
    nutrient_matrix: np.ndarray,
    calorie_per_gram: np.ndarray,
    steps: np.ndarray,
    fixed_flags: np.ndarray,
    fixed_weights: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = fixed_weights.copy()
    totals = nutrient_matrix.T @ weights
    total_calories = float(np.dot(calorie_per_gram, weights))

    for idx in range(len(weights)):
        if fixed_flags[idx]:
            continue

        max_bound = max_bounds[idx]
        min_bound = min_bounds[idx]
        if max_bound <= 0:
            weight = 0.0
        elif max_bound <= min_bound:
            weight = max_bound
        else:
            weight = rng.uniform(min_bound, max_bound)

        weight = _quantize_weight(weight, steps[idx], max_bound if max_bound > 0 else max_bound, min_weight=min_bound)
        weights[idx] = weight
        totals += nutrient_matrix[idx] * weight
        total_calories += calorie_per_gram[idx] * weight

    return weights, totals, total_calories


def _coordinate_descent(
    rng: random.Random,
    nutrient_matrix: np.ndarray,
    calorie_per_gram: np.ndarray,
    steps: np.ndarray,
    fixed_flags: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    weights: np.ndarray,
    totals: np.ndarray,
    total_calories: float,
    target_vector: np.ndarray,
    target_calories: float,
    calorie_weight: float,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    weight_factor = calorie_weight * calorie_weight if calorie_weight else 0.0

    for _ in range(max_iterations):
        max_delta = 0.0
        indices = list(range(len(weights)))
        rng.shuffle(indices)

        for idx in indices:
            if fixed_flags[idx]:
                continue

            max_bound = max_bounds[idx]
            min_bound = min_bounds[idx]
            if max_bound <= 0:
                if weights[idx] != 0:
                    delta = -weights[idx]
                    totals += nutrient_matrix[idx] * delta
                    total_calories += calorie_per_gram[idx] * delta
                    weights[idx] = 0.0
                continue

            current_weight = weights[idx]
            base_totals = totals - nutrient_matrix[idx] * current_weight
            base_calories = total_calories - calorie_per_gram[idx] * current_weight

            numerator = float(np.dot(nutrient_matrix[idx], target_vector - base_totals))
            denominator = float(np.dot(nutrient_matrix[idx], nutrient_matrix[idx]))

            if weight_factor > 0 and calorie_per_gram[idx] != 0:
                numerator += weight_factor * calorie_per_gram[idx] * (target_calories - base_calories)
                denominator += weight_factor * (calorie_per_gram[idx] ** 2)

            if denominator <= 0:
                candidate = min_bound
            else:
                candidate = numerator / denominator

            candidate = max(min_bound, min(max_bound, candidate))
            candidate = _quantize_weight(candidate, steps[idx], max_bound, min_weight=min_bound)
            candidate = max(min_bound, min(max_bound, candidate))

            if not math.isfinite(candidate):
                candidate = current_weight

            delta = candidate - current_weight
            if abs(delta) < 1e-9:
                continue

            weights[idx] = candidate
            totals = base_totals + nutrient_matrix[idx] * candidate
            total_calories = base_calories + calorie_per_gram[idx] * candidate
            max_delta = max(max_delta, abs(delta))

        if max_delta < tolerance:
            break

    return weights, totals, total_calories


def optimise_diet(
    products: List[OptimizationProduct],
    targets: Dict[str, float],
    run_count: int,
    residual_share: float,
    allow_zero_weights: bool = True,
    calorie_weight: float = 0.01,
) -> Dict[str, object]:
    if not products:
        raise ValueError("Список продуктов для оптимизации пуст.")
    if run_count <= 0:
        raise ValueError("Количество прогонов должно быть положительным.")

    target_vector = np.array([targets.get(key, 0.0) for key in NUTRIENT_ORDER], dtype=float)
    target_calories = float(targets.get("calories", 0.0))
    residual_sequence = _residual_share_sequence(residual_share)

    best_score = math.inf
    best_run = 0
    best_share = residual_sequence[0] if residual_sequence else 0.0
    best_weights: List[float] = []
    best_totals = np.zeros(len(NUTRIENT_ORDER), dtype=float)
    best_calories = target_calories

    rng = random.Random()

    for share_fraction in residual_sequence:
        (
            nutrient_matrix,
            calorie_per_gram,
            steps,
            fixed_flags,
            fixed_weights,
            min_bounds,
            max_bounds,
        ) = _build_product_arrays(products, share_fraction, allow_zero_weights)

        for run_index in range(1, run_count + 1):
            run_rng = random.Random(rng.random() * 1_000_000 + run_index)
            weights, totals, total_calories = _initialise_weights(
                run_rng,
                nutrient_matrix,
                calorie_per_gram,
                steps,
                fixed_flags,
                fixed_weights,
                min_bounds,
                max_bounds,
            )

            weights, totals, total_calories = _coordinate_descent(
                run_rng,
                nutrient_matrix,
                calorie_per_gram,
                steps,
                fixed_flags,
                min_bounds,
                max_bounds,
                weights,
                totals,
                total_calories,
                target_vector,
                target_calories,
                calorie_weight,
            )

            predicted_vector = np.array(totals, dtype=float)
            score = rmse(
                predicted_vector,
                target_vector,
                nutrient_keys=NUTRIENT_ORDER,
                calorie_weight=calorie_weight,
                predicted_calories=total_calories,
                target_calories=target_calories,
            )

            if score < best_score:
                best_score = score
                best_run = run_index
                best_share = share_fraction
                best_weights = list(map(float, weights))
                best_totals = predicted_vector.copy()
                best_calories = float(total_calories)

    weight_summary = [
        {"name": product.name, "weight": round(best_weights[idx], 2) if idx < len(best_weights) else 0.0}
        for idx, product in enumerate(products)
    ]

    totals_dict = {
        key: round(float(best_totals[idx]), 4)
        for idx, key in enumerate(NUTRIENT_ORDER)
    }
    totals_dict["calories"] = round(best_calories, 4)

    return {
        "rmse": round(float(best_score), 6) if math.isfinite(best_score) else 0.0,
        "run": best_run,
        "residual_share": round(float(best_share), 6),
        "weights": weight_summary,
        "totals": totals_dict,
    }


def optimize_from_payload(
    payload: Dict[str, object],
    product_db: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Построить рацион на основании данных из UI."""

    if not isinstance(payload, dict):
        raise ValueError("Некорректные данные запроса.")

    if product_db is None:
        product_db = load_product_database()

    targets_raw = payload.get("targets")
    if not isinstance(targets_raw, dict):
        raise ValueError("Целевые значения не переданы.")
    targets = _normalise_targets(targets_raw)

    try:
        run_count = int(payload.get("run_count", 10))
    except (TypeError, ValueError):
        run_count = 10
    if run_count <= 0:
        raise ValueError("Количество прогонов должно быть положительным.")

    residual_share = payload.get("residual_share", 0.0)
    try:
        residual_share = float(residual_share)
    except (TypeError, ValueError):
        residual_share = 0.0

    allow_zero_weights = _parse_bool(payload.get("allow_zero_weights"), default=True)

    products_payload = payload.get("products")
    if not isinstance(products_payload, list) or not products_payload:
        raise ValueError("Список продуктов пуст.")

    products = [_prepare_product(item, product_db) for item in products_payload]
    result = optimise_diet(
        products,
        targets,
        run_count=run_count,
        residual_share=residual_share,
        allow_zero_weights=allow_zero_weights,
    )
    result["targets"] = targets
    result["allow_zero_weights"] = allow_zero_weights
    return result


def main() -> None:
    sample_products = [
        OptimizationProduct(
            name="Sample Product",
            nutrients={
                "proteins": 20,
                "saturated": 5,
                "unsaturated": 5,
                "simple": 10,
                "complex": 40,
                "soluble": 5,
                "insoluble": 5,
                "kcal": 350,
            },
            step=10,
            max_weight=200,
        )
    ]
    sample_targets = {key: 50 for key in NUTRIENT_ORDER}
    sample_targets["calories"] = 2000
    result = optimise_diet(sample_products, sample_targets, run_count=5, residual_share=0.5)
    print(result)


if __name__ == "__main__":
    main()


