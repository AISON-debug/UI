from typing import Dict, Mapping

from GENESIS_optimize_nutrients import (
    CALORIE_KEY,
    DEFAULT_CSV_PATH,
    NUTRIENT_ORDER,
    OptimizationProduct,
    compute_calories,
    load_product_database,
    main,
    optimise_diet,
    optimize_from_payload as _optimize_from_payload,
)


_NUTRIENT_BREAKDOWN_KEYS = (*NUTRIENT_ORDER, CALORIE_KEY)


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resolve_product_nutrients(
    product_db: Mapping[str, Mapping[str, float]] | None, name: str
) -> Mapping[str, float] | None:
    if not isinstance(product_db, Mapping):
        return None
    data = product_db.get(name)
    if isinstance(data, Mapping):
        return data
    return None


def _build_nutrient_breakdown(
    weights: object,
    product_db: Mapping[str, Mapping[str, float]] | None,
) -> Dict[str, object]:
    rows: list[Dict[str, object]] = []
    totals: Dict[str, float] = {key: 0.0 for key in _NUTRIENT_BREAKDOWN_KEYS}

    if isinstance(weights, Mapping):
        iterable = []
    elif isinstance(weights, (list, tuple)):
        iterable = weights
    else:
        try:
            iterable = list(weights or [])  # type: ignore[arg-type]
        except TypeError:
            iterable = []

    for entry in iterable:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get('name') or '').strip()
        if not name and 'weight' not in entry:
            continue
        weight = max(0.0, _safe_float(entry.get('weight')))
        nutrients: Dict[str, float] = {}
        product_info = _resolve_product_nutrients(product_db, name)

        for key in _NUTRIENT_BREAKDOWN_KEYS:
            if isinstance(product_info, Mapping):
                if key == CALORIE_KEY:
                    raw_value = product_info.get('calories')
                    if raw_value is None:
                        raw_value = product_info.get('kcal')
                else:
                    raw_value = product_info.get(key)
            else:
                raw_value = 0.0
            per_100 = _safe_float(raw_value)
            contribution = weight * per_100 / 100.0
            contribution = round(contribution, 6)
            nutrients[key] = contribution
            totals[key] += contribution

        rows.append({'name': name, 'weight': round(weight, 4), 'nutrients': nutrients})

    totals = {key: round(value, 6) for key, value in totals.items()}
    return {'products': rows, 'totals': totals}


def optimize_from_payload(
    payload: Dict[str, object],
    product_db: Mapping[str, Mapping[str, float]] | None = None,
) -> Dict[str, object]:
    if product_db is None:
        product_db = load_product_database()

    result = _optimize_from_payload(payload, product_db=product_db)

    weights = result.get('weights') if isinstance(result, Mapping) else None
    breakdown = _build_nutrient_breakdown(weights, product_db)
    result['nutrient_breakdown'] = breakdown
    return result

__all__ = [
    'CALORIE_KEY',
    'DEFAULT_CSV_PATH',
    'NUTRIENT_ORDER',
    'OptimizationProduct',
    'compute_calories',
    'load_product_database',
    'optimise_diet',
    'optimize_from_payload',
    'main',
]


if __name__ == '__main__':
    main()
