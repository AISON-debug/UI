import csv
import math
import os
import random
import time
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
MAX_OPTIMISATION_ROUNDS = 10
NEGATIVE_TOLERANCE = 1e-9


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


def _run_iterative_optimisation(
    ordered_indices: Sequence[int],
    per_gram_map: Mapping[int, np.ndarray],
    residual: Mapping[str, float],
    capacities: Mapping[int, float],
    steps: Mapping[int, float],
    residual_fraction: float,
    rng: random.Random,
    noise_strength: float = 0.0,
) -> Dict[int, float]:
    if not ordered_indices:
        return {}

    base_residual_vec = {
        key: max(0.0, float(residual.get(key, 0.0))) for key in NUTRIENT_VECTOR_KEYS
    }
    noise_scale = max(0.0, float(noise_strength or 0.0))
    if noise_scale > 0.0:
        noise_vector = [
            max(0.0, 1.0 + rng.uniform(-noise_scale, noise_scale))
            for _ in NUTRIENT_VECTOR_KEYS
        ]
    else:
        noise_vector = [1.0] * len(NUTRIENT_VECTOR_KEYS)

    residual_vec = {
        key: base_residual_vec[key] * noise_vector[pos]
        for pos, key in enumerate(NUTRIENT_VECTOR_KEYS)
    }
    additions = {idx: 0.0 for idx in ordered_indices}
    active = list(ordered_indices)
    iteration = 0
    base_weight_vector = np.array(
        [RMSE_WEIGHTS.get(key, 1.0) for key in NUTRIENT_VECTOR_KEYS], dtype=float
    )
    if noise_scale > 0.0:
        weight_noise = np.array(
            [max(0.05, 1.0 + rng.uniform(-noise_scale, noise_scale)) for _ in NUTRIENT_VECTOR_KEYS],
            dtype=float,
        )
        weight_vector = base_weight_vector * weight_noise
    else:
        weight_vector = base_weight_vector

    while active and iteration < MAX_OPTIMISATION_ROUNDS:
        filtered: List[int] = []
        for idx in active:
            step = steps.get(idx, 0.0)
            remaining = max(0.0, capacities.get(idx, 0.0) - additions[idx])
            if step > 0 and remaining >= step / 2:
                filtered.append(idx)
        active = filtered
        if not active:
            break

        per_gram_matrix = np.stack([per_gram_map[idx] for idx in active])
        target_scaled = np.array(
            [residual_vec[key] * residual_fraction for key in NUTRIENT_VECTOR_KEYS],
            dtype=float,
        )
        weighted_matrix = np.nan_to_num(per_gram_matrix * weight_vector, nan=0.0, posinf=0.0, neginf=0.0)
        target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        b_vec = np.sum(weighted_matrix * target_scaled, axis=1)
        gram_matrix = np.nan_to_num(weighted_matrix @ per_gram_matrix.T, nan=0.0, posinf=0.0, neginf=0.0)

        solution = _solve_non_negative_system(gram_matrix, b_vec)
        if solution.size == 0:
            break

        any_positive = False
        for position, idx in enumerate(active):
            grams = float(solution[position])
            if not math.isfinite(grams) or grams <= 0:
                continue
            remaining_capacity = max(0.0, capacities.get(idx, 0.0) - additions[idx])
            if remaining_capacity <= 0:
                continue
            if grams > remaining_capacity:
                grams = remaining_capacity

            step = steps.get(idx, 0.0)
            if step > 0:
                grams = _quantize_to_step(grams, step, remaining_capacity)
            else:
                grams = max(0.0, min(grams, remaining_capacity))

            if grams <= 0:
                continue

            additions[idx] += grams
            any_positive = True

            per_gram_vector = per_gram_map[idx]
            for pos, key in enumerate(NUTRIENT_VECTOR_KEYS):
                residual_vec[key] -= float(per_gram_vector[pos] * grams)

        if not any_positive:
            break

        iteration += 1

    return {idx: additions[idx] for idx in ordered_indices if additions[idx] > 0}


def _randomised_candidate(
    variable_indices: Sequence[int],
    base_totals: Mapping[str, float],
    per_gram_map: Mapping[int, np.ndarray],
    capacities: Mapping[int, float],
    steps: Mapping[int, float],
    share: float,
    rng: random.Random,
) -> tuple[Dict[int, float], Dict[str, float]]:
    additions: Dict[int, float] = {}
    totals = dict(base_totals)
    share = max(0.0, min(1.0, share))
    for idx in variable_indices:
        capacity = max(0.0, capacities.get(idx, 0.0))
        if capacity <= 0:
            continue
        target_capacity = max(0.0, min(capacity, capacity * share))
        if target_capacity <= 0:
            continue
        step = steps.get(idx, 0.0)
        if step > 0:
            max_steps = int(math.floor(target_capacity / step + 1e-9))
            if max_steps <= 0:
                continue
            chosen_steps = rng.randint(0, max_steps)
            weight = chosen_steps * step
        else:
            weight = rng.uniform(0.0, target_capacity)
        weight = max(0.0, min(weight, capacity))
        if weight <= 0:
            continue
        additions[idx] = weight
        _accumulate_totals(totals, per_gram_map[idx], weight)
    return additions, totals


def _derive_seed() -> int:
    try:
        return int.from_bytes(os.urandom(16), 'big')
    except NotImplementedError:
        return int(time.time_ns())


def _noise_for_iteration(iteration: int) -> float:
    if iteration <= 1:
        return 0.0
    # увеличиваем амплитуду шума постепенно, но ограничиваем сверху
    growth = 0.015 * math.sqrt(float(iteration))
    return min(0.3, 0.05 + growth)


def optimise_diet(
    products: Sequence[OptimizationProduct],
    targets: Mapping[str, float],
    run_count: int,
    residual_share: float,
    allow_zero_weights: bool = True,
    rng: Optional[random.Random] = None,
) -> Dict[str, object]:
    if not products:
        raise ValueError('Список продуктов для оптимизации пуст.')
    if run_count <= 0:
        raise ValueError('Количество прогонов должно быть положительным.')

    if rng is None:
        seed = _derive_seed()
        rng = random.Random(seed)

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
        if not product.fix_weight and remaining_capacity > 0:
            variable_indices.append(idx)

    residual_vector = {
        key: targets_map.get(key, 0.0) - base_totals.get(key, 0.0)
        for key in NUTRIENT_VECTOR_KEYS
    }
    residual_sequence = _residual_share_sequence(residual_share)

    best_totals = dict(base_totals)
    best_score = _weighted_rmse(best_totals, targets_map)
    best_run = 0
    best_share = residual_sequence[0] if residual_sequence else 1.0
    best_additions: Dict[int, float] = {}

    iteration_counter = 0

    if variable_indices:
        for share in residual_sequence:
            share = max(0.0, min(1.0, share))
            for run_index in range(1, run_count + 1):
                iteration_counter += 1
                shuffled = list(variable_indices)
                rng.shuffle(shuffled)
                if iteration_counter == 1:
                    noise_strength = 0.0
                else:
                    noise_strength = _noise_for_iteration(iteration_counter)
                additions = _run_iterative_optimisation(
                    shuffled,
                    per_gram_map,
                    residual_vector,
                    capacities,
                    steps,
                    share,
                    rng,
                    noise_strength=noise_strength,
                )
                totals = dict(base_totals)
                for idx, grams in additions.items():
                    _accumulate_totals(totals, per_gram_map[idx], grams)
                score = _weighted_rmse(totals, targets_map)

                if noise_strength > 0.0 and variable_indices:
                    trial_count = max(
                        1,
                        min(len(variable_indices), int(1 + noise_strength * 10)),
                    )
                    for _ in range(trial_count):
                        random_additions, random_totals = _randomised_candidate(
                            variable_indices,
                            base_totals,
                            per_gram_map,
                            capacities,
                            steps,
                            share,
                            rng,
                        )
                        random_score = _weighted_rmse(random_totals, targets_map)
                        if random_score + 1e-9 < score:
                            additions = random_additions
                            totals = random_totals
                            score = random_score

                if score + 1e-9 < best_score:
                    best_score = score
                    best_run = iteration_counter
                    best_share = share
                    best_totals = totals
                    best_additions = additions

    weight_summary = []
    for idx, product in enumerate(products):
        total_weight = base_weights.get(idx, 0.0) + best_additions.get(idx, 0.0)
        total_weight = min(total_weight, product.max_weight)
        weight_summary.append({'name': product.name, 'weight': round(total_weight, 2)})

    result_totals = {key: round(best_totals.get(key, 0.0), 4) for key in NUTRIENT_VECTOR_KEYS}

    return {
        'rmse': round(float(best_score), 6),
        'run': best_run,
        'residual_share': round(float(best_share), 6),
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
