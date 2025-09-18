import csv
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


NUTRIENT_ORDER: tuple[str, ...] = (
    'proteins',
    'saturated',
    'unsaturated',
    'simple',
    'complex',
    'soluble',
    'insoluble',
)
CALORIE_KEY = 'calories'
DEFAULT_CSV_PATH = 'Nutrients DB.csv'

# Порядок нутриентов, который учитывается при расчётах (включая калории).
NUTRIENT_VECTOR_KEYS: tuple[str, ...] = (*NUTRIENT_ORDER, CALORIE_KEY)

# Веса для расчёта RMSE — соответствуют значениям из UI GENESIS.
RMSE_WEIGHTS: Dict[str, float] = {
    'proteins': 2.0,
    'saturated': 1.0,
    'unsaturated': 1.0,
    'simple': 1.0,
    'complex': 1.0,
    'soluble': 1.0,
    'insoluble': 1.0,
    'calories': 3.0,
}
NEGATIVE_TOLERANCE = 1e-9
RESIDUAL_EPS = 1e-6


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
        """Вернуть вес для фиксированного продукта, ограниченный допустимыми границами."""
        if not self.fix_weight or self.max_weight <= 0:
            return 0.0
        weight = self.fixed_weight if self.fixed_weight > 0 else self.max_weight
        weight = max(0.0, min(weight, self.max_weight))
        return weight

    def nutrient_value(self, key: str) -> float:
        """Получить содержание нутриента в 100 г продукта."""
        if key == CALORIE_KEY:
            value = self.nutrients.get('calories')
            if value is None:
                value = self.nutrients.get('kcal')
            if value is None:
                value = _compute_calories_from_macros(self.nutrients)
            return float(value or 0.0)
        return float(self.nutrients.get(key, 0.0) or 0.0)


def compute_calories(nutrients: Iterable[float]) -> float:
    """Calculate calories from macronutrient amounts (proteins, fats, carbs)."""
    protein, fat, carbs = nutrients
    return protein * 4 + fat * 9 + carbs * 4


def _compute_calories_from_macros(nutrients: Mapping[str, float]) -> float:
    protein = float(nutrients.get('proteins', 0.0) or 0.0)
    fats = float(nutrients.get('saturated', 0.0) or 0.0) + float(
        nutrients.get('unsaturated', 0.0) or 0.0
    )
    carbs = (
        float(nutrients.get('simple', 0.0) or 0.0)
        + float(nutrients.get('complex', 0.0) or 0.0)
        + float(nutrients.get('soluble', 0.0) or 0.0)
        + float(nutrients.get('insoluble', 0.0) or 0.0)
    )
    return compute_calories((protein, fats, carbs))


def _parse_float(value: Optional[str]) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return default


def _clamp_fraction(value: float) -> float:
    try:
        fraction = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(fraction):
        return 0.0
    if fraction < 0.0:
        return 0.0
    if fraction > 1.0:
        return 1.0
    return fraction


def load_product_database(csv_path: str = DEFAULT_CSV_PATH) -> Dict[str, Dict[str, float]]:
    """Загрузить базу продуктов из CSV и вернуть словарь с ключами нутриентов."""
    products: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_name = (row.get('Продукт') or '').strip().replace('\ufeff', '')
            if not raw_name:
                continue
            proteins = _parse_float(row.get('Белки'))
            saturated = _parse_float(row.get('Насыщенные'))
            unsaturated = _parse_float(row.get('НЕнасыщенные'))
            simple = _parse_float(row.get('Простые'))
            complex_carbs = _parse_float(row.get('Сложные перевариваемые'))
            soluble = _parse_float(row.get('Растворимая'))
            insoluble = _parse_float(row.get('Нерастворимая'))
            kcal = _parse_float(row.get('ККал'))
            if kcal <= 0:
                kcal = _compute_calories_from_macros(
                    {
                        'proteins': proteins,
                        'saturated': saturated,
                        'unsaturated': unsaturated,
                        'simple': simple,
                        'complex': complex_carbs,
                        'soluble': soluble,
                        'insoluble': insoluble,
                    }
                )
            products[raw_name] = {
                'proteins': proteins,
                'saturated': saturated,
                'unsaturated': unsaturated,
                'simple': simple,
                'complex': complex_carbs,
                'soluble': soluble,
                'insoluble': insoluble,
                'kcal': kcal,
                'calories': kcal,
            }
    return products


def _quantize_to_step(value: float, step: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    value = max(0.0, min(value, maximum))
    if step <= 0:
        return value
    rounded = round(value / step) * step
    if rounded > maximum:
        rounded = math.floor(maximum / step) * step
    return max(0.0, min(rounded, maximum))


def _compute_residuals(
    targets: Mapping[str, float], totals: Mapping[str, float]
) -> Dict[str, float]:
    residuals: Dict[str, float] = {}
    for key in NUTRIENT_VECTOR_KEYS:
        residuals[key] = float(targets.get(key, 0.0) - totals.get(key, 0.0))
    return residuals


def _positive_residual_indices(residuals: Mapping[str, float]) -> List[int]:
    indices: List[int] = []
    for pos, key in enumerate(NUTRIENT_VECTOR_KEYS):
        if residuals.get(key, 0.0) > RESIDUAL_EPS:
            indices.append(pos)
    return indices


def _estimate_alpha(
    nutrient_indices: Sequence[int],
    active_indices: Sequence[int],
    residuals: Mapping[str, float],
    per_gram_map: Mapping[int, np.ndarray],
    capacities: Mapping[int, float],
    additions: Mapping[int, float],
    steps: Mapping[int, float],
    requested_fraction: float,
) -> float:
    alpha_candidates: List[float] = []
    for nutrient_index in nutrient_indices:
        key = NUTRIENT_VECTOR_KEYS[nutrient_index]
        residual_value = residuals.get(key, 0.0)
        if residual_value <= RESIDUAL_EPS:
            continue
        ratios: List[float] = []
        for idx in active_indices:
            available = max(0.0, capacities.get(idx, 0.0) - additions.get(idx, 0.0))
            if available <= RESIDUAL_EPS:
                continue
            per_nutrient = float(per_gram_map[idx][nutrient_index])
            if per_nutrient <= 0.0:
                continue
            step = steps.get(idx, 0.0)
            if step > 0.0:
                increment = min(available, step)
            else:
                increment = available
            if increment <= RESIDUAL_EPS:
                continue
            portion_value = per_nutrient * increment
            if portion_value <= 0.0:
                continue
            ratios.append(portion_value / residual_value)
        if ratios:
            alpha_candidates.append(min(ratios))

    if not alpha_candidates:
        return 0.0

    computed = max(alpha_candidates)
    requested = _clamp_fraction(requested_fraction)
    if requested > 0.0:
        computed = max(computed, requested)
    return max(0.0, min(1.0, computed))


def _build_weighted_system(
    nutrient_indices: Sequence[int],
    active_indices: Sequence[int],
    per_gram_map: Mapping[int, np.ndarray],
    residuals: Mapping[str, float],
    alpha: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not nutrient_indices or not active_indices:
        return None

    matrix = np.zeros((len(nutrient_indices), len(active_indices)), dtype=float)
    for col, idx in enumerate(active_indices):
        matrix[:, col] = np.array(per_gram_map[idx].take(nutrient_indices), dtype=float)

    if not np.any(matrix):
        return None

    weights = np.array(
        [math.sqrt(RMSE_WEIGHTS.get(NUTRIENT_VECTOR_KEYS[pos], 1.0)) for pos in nutrient_indices],
        dtype=float,
    )
    targets = np.array(
        [max(0.0, residuals.get(NUTRIENT_VECTOR_KEYS[pos], 0.0)) * alpha for pos in nutrient_indices],
        dtype=float,
    )

    weighted_matrix = np.nan_to_num(matrix * weights[:, None], nan=0.0, posinf=0.0, neginf=0.0)
    weighted_target = np.nan_to_num(targets * weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weighted_matrix, weighted_target


def _best_math_optimisation(
    variable_indices: Sequence[int],
    per_gram_map: Mapping[int, np.ndarray],
    capacities: Mapping[int, float],
    steps: Mapping[int, float],
    base_totals: Mapping[str, float],
    targets: Mapping[str, float],
    requested_fraction: float,
    max_iterations: int,
) -> tuple[Dict[int, float], Dict[str, float], float, int]:
    additions: Dict[int, float] = {idx: 0.0 for idx in variable_indices}
    totals = {key: float(base_totals.get(key, 0.0)) for key in NUTRIENT_VECTOR_KEYS}
    residuals = _compute_residuals(targets, totals)
    last_alpha = 0.0
    iterations = 0
    max_iterations = max(1, int(max_iterations))

    while iterations < max_iterations:
        nutrient_indices = _positive_residual_indices(residuals)
        if not nutrient_indices:
            break

        active_indices = [
            idx
            for idx in variable_indices
            if capacities.get(idx, 0.0) - additions.get(idx, 0.0) > RESIDUAL_EPS
        ]
        if not active_indices:
            break

        alpha = _estimate_alpha(
            nutrient_indices,
            active_indices,
            residuals,
            per_gram_map,
            capacities,
            additions,
            steps,
            requested_fraction,
        )
        if alpha <= 0.0:
            break

        system = _build_weighted_system(
            nutrient_indices,
            active_indices,
            per_gram_map,
            residuals,
            alpha,
        )
        if system is None:
            break
        weighted_matrix, weighted_target = system
        gram = np.nan_to_num(weighted_matrix.T @ weighted_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        b_vec = np.nan_to_num(weighted_matrix.T @ weighted_target, nan=0.0, posinf=0.0, neginf=0.0)

        solution = _solve_non_negative_system(gram, b_vec)
        if solution.size == 0:
            break

        any_added = False
        for sol_pos, idx in enumerate(active_indices):
            grams = float(solution[sol_pos])
            if not math.isfinite(grams) or grams <= 0.0:
                continue
            available = max(0.0, capacities.get(idx, 0.0) - additions.get(idx, 0.0))
            if available <= RESIDUAL_EPS:
                continue
            grams = min(grams, available)
            step = steps.get(idx, 0.0)
            if step > 0.0:
                grams = _quantize_to_step(grams, step, available)
            else:
                grams = max(0.0, min(grams, available))
            if grams <= RESIDUAL_EPS:
                continue

            additions[idx] += grams
            any_added = True
            per_vector = per_gram_map[idx]
            for pos, key in enumerate(NUTRIENT_VECTOR_KEYS):
                totals[key] = totals.get(key, 0.0) + float(per_vector[pos] * grams)
                residuals[key] = float(targets.get(key, 0.0) - totals[key])

        last_alpha = alpha
        iterations += 1

        if not any_added:
            break

    if last_alpha <= 0.0:
        last_alpha = _clamp_fraction(requested_fraction)

    return additions, totals, last_alpha, iterations


def _normalise_targets(raw: Mapping[str, float]) -> Dict[str, float]:
    result = {}
    for key in NUTRIENT_VECTOR_KEYS:
        try:
            result[key] = float(raw.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            result[key] = 0.0
    return result


def _prepare_product(entry: Mapping[str, object], product_db: Mapping[str, Dict[str, float]]) -> OptimizationProduct:
    if not isinstance(entry, Mapping):
        raise ValueError('Некорректные данные продукта.')
    name = str(entry.get('name') or '').strip()
    if not name:
        raise ValueError('Не указано название продукта.')

    nutrients_data = entry.get('nutrients')
    if isinstance(nutrients_data, Mapping):
        nutrients = {
            'proteins': _parse_float(nutrients_data.get('proteins')),
            'saturated': _parse_float(nutrients_data.get('saturated')),
            'unsaturated': _parse_float(nutrients_data.get('unsaturated')),
            'simple': _parse_float(nutrients_data.get('simple')),
            'complex': _parse_float(nutrients_data.get('complex')),
            'soluble': _parse_float(nutrients_data.get('soluble')),
            'insoluble': _parse_float(nutrients_data.get('insoluble')),
            'kcal': _parse_float(nutrients_data.get('kcal')),
        }
    else:
        if name not in product_db:
            raise ValueError(f'Продукт "{name}" отсутствует в базе.')
        nutrients = dict(product_db[name])

    if nutrients.get('calories') is None:
        calories = nutrients.get('kcal')
        if not calories or calories <= 0:
            calories = _compute_calories_from_macros(nutrients)
        nutrients['calories'] = float(calories or 0.0)
    else:
        nutrients['calories'] = float(nutrients['calories'] or 0.0)

    step = _parse_float(entry.get('step'))
    max_weight = max(0.0, _parse_float(entry.get('max_weight')))
    fix_weight = _parse_bool(entry.get('fix_weight'))
    fixed_weight = _parse_float(entry.get('fixed_weight'))
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


def _minimum_variable_weight(product: OptimizationProduct) -> float:
    if product.max_weight <= 0:
        return 0.0
    step = product.step
    if step > 0:
        minimum = min(step, product.max_weight)
    else:
        minimum = min(product.max_weight, 1.0)
    if minimum <= 0:
        return 0.0
    return max(0.0, min(minimum, product.max_weight))


def _build_per_gram_map(products: Sequence[OptimizationProduct]) -> Dict[int, np.ndarray]:
    per_gram: Dict[int, np.ndarray] = {}
    for idx, product in enumerate(products):
        values = [product.nutrient_value(key) / 100.0 for key in NUTRIENT_VECTOR_KEYS]
        per_gram[idx] = np.array(values, dtype=float)
    return per_gram


def _accumulate_totals(
    totals: Dict[str, float],
    per_gram_vector: np.ndarray,
    grams: float,
) -> None:
    for pos, key in enumerate(NUTRIENT_VECTOR_KEYS):
        totals[key] = totals.get(key, 0.0) + float(per_gram_vector[pos] * grams)


def _weighted_rmse(
    totals: Mapping[str, float],
    targets: Mapping[str, float],
) -> float:
    error_sum = 0.0
    count = 0
    for key in NUTRIENT_VECTOR_KEYS:
        weight = RMSE_WEIGHTS.get(key, 1.0)
        diff = float(targets.get(key, 0.0) - totals.get(key, 0.0))
        error_sum += weight * diff * diff
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(error_sum / count)

def _solve_non_negative_system(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    size = vector.shape[0]
    if size == 0:
        return np.zeros(0, dtype=float)

    matrix_active = np.array(matrix, dtype=float)
    vector_active = np.array(vector, dtype=float)
    active_indices = list(range(size))
    solution = np.zeros(size, dtype=float)

    while active_indices:
        if matrix_active.size == 0:
            break
        matrix_to_solve = matrix_active + np.eye(matrix_active.shape[0]) * 1e-12
        try:
            sol_active = np.linalg.solve(matrix_to_solve, vector_active)
        except np.linalg.LinAlgError:
            sol_active = np.linalg.lstsq(matrix_to_solve, vector_active, rcond=None)[0]

        if sol_active.size == 0:
            break

        neg_indices = [i for i, val in enumerate(sol_active) if val < -NEGATIVE_TOLERANCE]
        if not neg_indices:
            sol_active = np.where(sol_active < 0, 0.0, sol_active)
            for idx, val in zip(active_indices, sol_active):
                solution[idx] = float(val)
            return solution

        if len(neg_indices) == len(active_indices):
            return solution

        keep_mask = [i for i in range(len(active_indices)) if i not in neg_indices]
        if not keep_mask:
            return solution

        active_indices = [active_indices[i] for i in keep_mask]
        matrix_active = matrix_active[np.ix_(keep_mask, keep_mask)]
        vector_active = vector_active[keep_mask]

    return solution


def optimise_diet(
    products: Sequence[OptimizationProduct],
    targets: Mapping[str, float],
    run_count: int,
    residual_share: float,
    allow_zero_weights: bool = True,
    rng: Optional[object] = None,
) -> Dict[str, object]:
    del rng  # генератор случайных чисел не используется в детерминированной версии

    if not products:
        raise ValueError('Список продуктов для оптимизации пуст.')
    if run_count <= 0:
        raise ValueError('Количество прогонов должно быть положительным.')

    targets_map = _normalise_targets(targets)
    per_gram_map = _build_per_gram_map(products)

    base_totals = {key: 0.0 for key in NUTRIENT_VECTOR_KEYS}
    base_weights: Dict[int, float] = {}
    capacities: Dict[int, float] = {}
    steps: Dict[int, float] = {}
    variable_indices: List[int] = []

    for idx, product in enumerate(products):
        step_value = max(0.0, product.step)
        steps[idx] = step_value
        max_weight = max(0.0, product.max_weight)
        base_weight = 0.0
        if product.fix_weight:
            base_weight = product.resolved_fixed_weight()
        elif not allow_zero_weights:
            base_weight = _minimum_variable_weight(product)
        base_weight = max(0.0, min(base_weight, max_weight))
        base_weights[idx] = base_weight
        remaining_capacity = max(0.0, max_weight - base_weight)
        capacities[idx] = remaining_capacity
        if base_weight > 0:
            _accumulate_totals(base_totals, per_gram_map[idx], base_weight)
        if not product.fix_weight and remaining_capacity > RESIDUAL_EPS:
            variable_indices.append(idx)

    requested_fraction = _clamp_fraction(residual_share)

    if variable_indices:
        additions, totals, alpha, iterations = _best_math_optimisation(
            variable_indices,
            per_gram_map,
            capacities,
            steps,
            base_totals,
            targets_map,
            requested_fraction,
            max_iterations=run_count,
        )
    else:
        additions = {}
        totals = dict(base_totals)
        alpha = requested_fraction
        iterations = 0

    weight_summary = []
    for idx, product in enumerate(products):
        total_weight = base_weights.get(idx, 0.0) + additions.get(idx, 0.0)
        total_weight = min(total_weight, product.max_weight)
        weight_summary.append({'name': product.name, 'weight': round(total_weight, 2)})

    result_totals = {key: round(totals.get(key, 0.0), 4) for key in NUTRIENT_VECTOR_KEYS}
    score = _weighted_rmse(totals, targets_map)

    return {
        'rmse': round(float(score), 6),
        'run': int(iterations),
        'residual_share': round(float(alpha), 6),
        'weights': weight_summary,
        'totals': result_totals,
    }


def optimize_from_payload(
    payload: Dict[str, object],
    product_db: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError('Некорректные данные запроса.')

    if product_db is None:
        product_db = load_product_database()

    targets_raw = payload.get('targets')
    if not isinstance(targets_raw, Mapping):
        raise ValueError('Целевые значения не переданы.')
    targets = _normalise_targets(targets_raw)

    try:
        run_count = int(payload.get('run_count', 10))
    except (TypeError, ValueError):
        run_count = 10
    if run_count <= 0:
        raise ValueError('Количество прогонов должно быть положительным.')

    residual_share = payload.get('residual_share', 0.0)
    try:
        residual_share = float(residual_share)
    except (TypeError, ValueError):
        residual_share = 0.0

    allow_zero_weights = _parse_bool(payload.get('allow_zero_weights'), default=True)

    products_payload = payload.get('products')
    if not isinstance(products_payload, list) or not products_payload:
        raise ValueError('Список продуктов пуст.')

    products = [_prepare_product(item, product_db) for item in products_payload]

    result = optimise_diet(
        products,
        targets,
        run_count=run_count,
        residual_share=residual_share,
        allow_zero_weights=allow_zero_weights,
    )
    result['targets'] = targets
    result['allow_zero_weights'] = allow_zero_weights
    return result


def main() -> None:
    sample_products = [
        OptimizationProduct(
            name='Sample Product',
            nutrients={
                'proteins': 20,
                'saturated': 5,
                'unsaturated': 5,
                'simple': 10,
                'complex': 40,
                'soluble': 5,
                'insoluble': 5,
                'kcal': 350,
                'calories': 350,
            },
            step=10,
            max_weight=200,
        )
    ]
    sample_targets = {key: 50 for key in NUTRIENT_ORDER}
    sample_targets[CALORIE_KEY] = 2000
    result = optimise_diet(sample_products, sample_targets, run_count=5, residual_share=0.5)
    print(result)


if __name__ == '__main__':
    main()
