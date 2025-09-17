import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

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
DEFAULT_CSV_PATH = 'Nutrients DB.csv'


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
        return min(self.max_weight, max(0.0, weight))


def compute_calories(nutrients: Iterable[float]) -> float:
    """Calculate calories from macronutrient amounts."""
    protein, fat, carbs = nutrients
    return protein * 4 + fat * 9 + carbs * 4


def _macro_tuple_from_vector(vector: Iterable[float], nutrient_keys: Iterable[str]) -> tuple[float, float, float]:
    """Построить кортеж (белки, жиры, углеводы) из вектора нутриентов."""

    index_map = {key: idx for idx, key in enumerate(nutrient_keys)}

    def _value(key: str) -> float:
        position = index_map.get(key)
        if position is None:
            return 0.0
        try:
            return float(vector[position])
        except (TypeError, ValueError):
            return 0.0

    protein = _value('proteins')
    fats = _value('saturated') + _value('unsaturated')
    carbs = sum(_value(key) for key in ('simple', 'complex', 'soluble', 'insoluble'))

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
                    raise ValueError('nutrient_keys are required to compute calories automatically')
                protein_pred, fat_pred, carb_pred = _macro_tuple_from_vector(predicted_arr, nutrient_keys)
                protein_target, fat_target, carb_target = _macro_tuple_from_vector(target_arr, nutrient_keys)
                if predicted_calories is None:
                    predicted_calories = compute_calories((protein_pred, fat_pred, carb_pred))
                if target_calories is None:
                    target_calories = compute_calories((protein_target, fat_target, carb_target))

            calories_diff = (float(predicted_calories or 0.0) - float(target_calories or 0.0)) * cal_weight
            diffs = np.append(diffs, calories_diff)

    return float(np.sqrt(np.mean(diffs ** 2)))


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
    seen: Set[float] = set()

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
            products[raw_name] = {
                'proteins': _parse_float(row.get('Белки')),
                'saturated': _parse_float(row.get('Насыщенные')),
                'unsaturated': _parse_float(row.get('НЕнасыщенные')),
                'simple': _parse_float(row.get('Простые')),
                'complex': _parse_float(row.get('Сложные перевариваемые')),
                'soluble': _parse_float(row.get('Растворимая')),
                'insoluble': _parse_float(row.get('Нерастворимая')),
                'kcal': _parse_float(row.get('ККал')),
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
    for key in (*NUTRIENT_ORDER, 'calories'):
        try:
            result[key] = float(raw.get(key, 0) or 0)
        except (TypeError, ValueError):
            result[key] = 0.0
    return result


def _prepare_product(entry: Dict[str, object], product_db: Dict[str, Dict[str, float]]) -> OptimizationProduct:
    if not isinstance(entry, dict):
        raise ValueError('Некорректные данные продукта.')
    name = str(entry.get('name') or '').strip()
    if not name:
        raise ValueError('Не указано название продукта.')

    nutrients_data = entry.get('nutrients')
    if isinstance(nutrients_data, dict):
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
        nutrients = product_db[name]

    step = _parse_float(entry.get('step'))
    max_weight = max(0.0, _parse_float(entry.get('max_weight')))
    fix_weight = bool(entry.get('fix_weight'))
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


def optimise_diet(
    products: List[OptimizationProduct],
    targets: Dict[str, float],
    run_count: int,
    residual_share: float,
    allow_zero_weights: bool = True,
    calorie_weight: float = 0.01,
) -> Dict[str, object]:
    if not products:
        raise ValueError('Список продуктов для оптимизации пуст.')
    if run_count <= 0:
        raise ValueError('Количество прогонов должно быть положительным.')

    target_vector = np.array([targets.get(key, 0.0) for key in NUTRIENT_ORDER], dtype=float)
    target_calories = float(targets.get('calories', 0.0))
    residual_sequence = _residual_share_sequence(residual_share)

    best_score = math.inf
    best_run = 0
    best_share = residual_sequence[0] if residual_sequence else 0.0
    best_weights: List[float] = []
    best_totals: Dict[str, float] = {}

    for share_fraction in residual_sequence:
        for run_index in range(1, run_count + 1):
            totals = {key: 0.0 for key in NUTRIENT_ORDER}
            totals['calories'] = 0.0
            weights: List[float] = []

            for product in products:
                if product.max_weight <= 0:
                    weights.append(0.0)
                    continue

                min_weight = 0.0
                if (
                    not allow_zero_weights
                    and not product.fix_weight
                    and product.max_weight > 0
                ):
                    if product.step > 0:
                        min_weight = min(product.step, product.max_weight)
                    else:
                        min_weight = min(product.max_weight, 1.0)
                    if min_weight <= 0:
                        min_weight = product.max_weight

                if product.fix_weight:
                    weight = product.resolved_fixed_weight()
                else:
                    candidate = (
                        product.max_weight * random.random() * share_fraction
                        if share_fraction > 0
                        else 0.0
                    )
                    weight = _quantize_weight(
                        candidate,
                        product.step,
                        product.max_weight,
                        min_weight=min_weight,
                    )

                weights.append(weight)
                weight_factor = weight / 100.0
                for key in NUTRIENT_ORDER:
                    totals[key] += product.nutrients.get(key, 0.0) * weight_factor
                kcal_value = product.nutrients.get('kcal')
                if kcal_value is None:
                    fats = product.nutrients.get('saturated', 0.0) + product.nutrients.get('unsaturated', 0.0)
                    carbs = (
                        product.nutrients.get('simple', 0.0)
                        + product.nutrients.get('complex', 0.0)
                        + product.nutrients.get('soluble', 0.0)
                        + product.nutrients.get('insoluble', 0.0)
                    )
                    kcal_value = compute_calories((product.nutrients.get('proteins', 0.0), fats, carbs))
                totals['calories'] += kcal_value * weight_factor

            predicted_vector = np.array([totals[key] for key in NUTRIENT_ORDER], dtype=float)
            predicted_calories = totals.get('calories')
            score = rmse(
                predicted_vector,
                target_vector,
                nutrient_keys=NUTRIENT_ORDER,
                calorie_weight=calorie_weight,
                predicted_calories=predicted_calories if predicted_calories else None,
                target_calories=target_calories if target_calories else None,
            )
            score = float(score)

            if score < best_score:
                best_score = score
                best_run = run_index
                best_share = share_fraction
                best_weights = weights.copy()
                best_totals = totals.copy()

    weight_summary = [
        {'name': product.name, 'weight': round(weight, 2)}
        for product, weight in zip(products, best_weights)
    ]

    result_totals = {key: round(best_totals.get(key, 0.0), 4) for key in (*NUTRIENT_ORDER, 'calories')}
    result_totals['calories'] = round(best_totals.get('calories', target_calories), 4)

    return {
        'rmse': round(best_score, 6) if math.isfinite(best_score) else 0.0,
        'run': best_run,
        'residual_share': round(best_share, 6),
        'weights': weight_summary,
        'totals': result_totals,
    }


def optimize_from_payload(payload: Dict[str, object], product_db: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, object]:
    """Построить рацион на основании данных из UI."""
    if not isinstance(payload, dict):
        raise ValueError('Некорректные данные запроса.')

    if product_db is None:
        product_db = load_product_database()

    targets_raw = payload.get('targets')
    if not isinstance(targets_raw, dict):
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
    """Небольшой пример использования."""
    sample_products = [
        OptimizationProduct(
            name='Sample Product',
            nutrients={'proteins': 20, 'saturated': 5, 'unsaturated': 5, 'simple': 10, 'complex': 40, 'soluble': 5, 'insoluble': 5, 'kcal': 350},
            step=10,
            max_weight=200,
        )
    ]
    sample_targets = {key: 50 for key in NUTRIENT_ORDER}
    sample_targets['calories'] = 2000
    result = optimise_diet(sample_products, sample_targets, run_count=5, residual_share=0.5)
    print(result)


if __name__ == '__main__':
    main()
