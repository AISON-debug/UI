import csv
import random
import math

# Mapping of CSV column names to internal keys
COMPLEX_KEY = 'Сложные \nперевариваемые'
CSV_TO_KEY = {
    'Белки': 'protein',
    'Насыщенные': 'saturatedFat',
    'НЕнасыщенные': 'unsaturatedFat',
    'Простые': 'simpleCarbs',
    COMPLEX_KEY: 'complexCarbs',
    'Растворимая': 'solubleFiber',
    'Нерастворимая': 'insolubleFiber',
    'ККал': 'calories',
}

# Order of nutrient keys used in optimisation and output
NUT_KEYS = [
    'protein',
    'saturatedFat',
    'unsaturatedFat',
    'simpleCarbs',
    'complexCarbs',
    'solubleFiber',
    'insolubleFiber',
    'calories',
]

# Russian labels for output
KEY_TO_RUS = {v: k for k, v in CSV_TO_KEY.items()}

# Weights for error calculation (from nutrition_webapp 31.08.2025.html)
WEIGHTS = {
    'protein': 2,
    'saturatedFat': 1,
    'unsaturatedFat': 1,
    'simpleCarbs': 1,
    'complexCarbs': 1,
    'solubleFiber': 1,
    'insolubleFiber': 1,
    'calories': 3,
}


def js_round(x: float) -> int:
    """Round half up as JavaScript's Math.round."""
    return math.floor(x + 0.5)


def load_products(path: str):
    """Load products with nutrient composition, step and max weight."""
    products = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nutrients = {CSV_TO_KEY[k]: float(row[k]) for k in CSV_TO_KEY}
            step = float(row['Шаг'])
            max_weight = float(row['Макс. порций']) * step
            products.append({
                'name': row['Продукт'],
                'nutrients': nutrients,
                'step': step,
                'max_weight': max_weight,
            })
    return products


def run_iterative_optimization(products, var_idxs, resid, alpha, max_iter=10):
    """Replicate runIterativeOptimization from nutrition_webapp."""
    resid_vec = {k: resid[k] for k in resid}
    var_add = {idx: 0.0 for idx in var_idxs}
    step_vals = {idx: products[idx]['step'] for idx in var_idxs}
    max_vals = {idx: products[idx]['max_weight'] for idx in var_idxs}
    nut_keys = NUT_KEYS
    active = var_idxs[:]
    iteration = 0
    while active and iteration < max_iter:
        active = [
            idx
            for idx in active
            if step_vals[idx] > 0 and (max_vals[idx] - var_add[idx]) >= step_vals[idx] / 2
        ]
        if not active:
            break
        m = len(active)
        pmat = []
        for idx in active:
            p = products[idx]['nutrients']
            pmat.append([p[k] / 100.0 for k in nut_keys])
        G = [[0.0] * m for _ in range(m)]
        b = [0.0] * m
        for i in range(m):
            sumB = 0.0
            for k, key in enumerate(nut_keys):
                w = WEIGHTS[key]
                target_scaled = resid_vec[key] * alpha
                sumB += w * target_scaled * pmat[i][k]
            b[i] = sumB
            for j in range(m):
                sumG = 0.0
                for k2, key2 in enumerate(nut_keys):
                    w = WEIGHTS[key2]
                    sumG += w * pmat[i][k2] * pmat[j][k2]
                G[i][j] = sumG
        active_idxs = active[:]
        activeG = [row[:] for row in G]
        activeB = b[:]
        sol = []
        while True:
            m2 = len(active_idxs)
            if m2 == 0:
                break
            aug = [activeG[i][:] + [activeB[i]] for i in range(m2)]
            for col in range(m2):
                pivot = max(range(col, m2), key=lambda r: abs(aug[r][col]))
                if abs(aug[pivot][col]) < 1e-12:
                    continue
                if pivot != col:
                    aug[col], aug[pivot] = aug[pivot], aug[col]
                piv = aug[col][col]
                for j in range(col, m2 + 1):
                    aug[col][j] /= piv
                for i2 in range(m2):
                    if i2 == col:
                        continue
                    factor = aug[i2][col]
                    for j in range(col, m2 + 1):
                        aug[i2][j] -= factor * aug[col][j]
            sol_vec = [0.0] * m2
            for i2 in range(m2 - 1, -1, -1):
                sol_vec[i2] = aug[i2][m2]
                for j in range(i2 + 1, m2):
                    sol_vec[i2] -= aug[i2][j] * sol_vec[j]
            neg = [i for i, v in enumerate(sol_vec) if v < 0]
            if not neg:
                sol = sol_vec
                break
            keep = [i for i in range(m2) if i not in neg]
            active_idxs = [active_idxs[i] for i in keep]
            activeB = [activeB[i] for i in keep]
            activeG = [[row[j] for j in keep] for row_idx, row in enumerate(activeG) if row_idx in keep]
        if not sol:
            break
        any_positive = False
        for i, idx in enumerate(active_idxs):
            grams = sol[i]
            if grams <= 0:
                continue
            remaining = max_vals[idx] - var_add[idx]
            if remaining <= 0:
                continue
            if grams > remaining:
                grams = remaining
            step = step_vals[idx]
            if step > 0:
                rounded = js_round(grams / step) * step
                if rounded > remaining:
                    rounded = math.floor(remaining / step) * step
                if rounded < 0:
                    rounded = 0
                grams = rounded
            if grams <= 0:
                continue
            var_add[idx] += grams
            any_positive = True
            p = products[idx]['nutrients']
            for key in nut_keys:
                resid_vec[key] -= (p[key] / 100.0) * grams
        if not any_positive:
            break
        iteration += 1
    return var_add


def evaluate_diet(products, var_idxs, targets, alpha):
    resid = {k: targets[k] for k in NUT_KEYS}
    var_map = run_iterative_optimization(products, var_idxs, resid, alpha)
    totals = {k: 0.0 for k in NUT_KEYS}
    for idx in var_idxs:
        grams = var_map.get(idx, 0.0)
        if grams > 0:
            p = products[idx]['nutrients']
            factor = grams / 100.0
            for key in NUT_KEYS:
                totals[key] += p[key] * factor
    err_sum = 0.0
    n_keys = 0
    for key in NUT_KEYS:
        diff = targets[key] - totals[key]
        err_sum += WEIGHTS[key] * diff * diff
        n_keys += 1
    rmse = math.sqrt(err_sum / n_keys) if n_keys > 0 else 0.0
    return var_map, totals, rmse


def calculate_calories(targets):
    return (
        4 * targets['protein']
        + 9 * (targets['saturatedFat'] + targets['unsaturatedFat'])
        + 4 * (targets['simpleCarbs'] + targets['complexCarbs'])
        + 1.5 * (targets['solubleFiber'] + targets['insolubleFiber'])
    )


def main():
    products = load_products('Nutrients DB.csv')
    print('Введите целевые значения для нутриентов (в граммах):')
    targets = {}
    order = [
        ('protein', 'Белки'),
        ('saturatedFat', 'Насыщенные'),
        ('unsaturatedFat', 'НЕнасыщенные'),
        ('simpleCarbs', 'Простые'),
        ('complexCarbs', 'Сложные перевариваемые'),
        ('solubleFiber', 'Растворимая'),
        ('insolubleFiber', 'Нерастворимая'),
    ]
    for key, label in order:
        while True:
            try:
                val = float(input(f'{label}: '))
                break
            except ValueError:
                print('Введите числовое значение')
        targets[key] = val
    targets['calories'] = calculate_calories(targets)
    while True:
        try:
            start_alpha = int(input('Начальное значение альфа (1-100): '))
            if 1 <= start_alpha <= 100:
                break
            print('Введите число от 1 до 100')
        except ValueError:
            print('Введите целое число')
    while True:
        try:
            runs_per_alpha = int(
                input('Количество прогонов для каждого альфа (0-100): ')
            )
            if 0 <= runs_per_alpha <= 100:
                break
            print('Введите число от 0 до 100')
        except ValueError:
            print('Введите целое число')
    var_indices = list(range(len(products)))
    best = None
    repeats = max(1, runs_per_alpha)
    for alpha in range(start_alpha, 101):
        for run in range(1, repeats + 1):
            shuffled = var_indices[:]
            random.shuffle(shuffled)
            var_map, totals, rmse = evaluate_diet(
                products, shuffled, targets, alpha / 100.0
            )
            weights = [0.0] * len(products)
            for idx, grams in var_map.items():
                weights[idx] = grams
            if best is None or rmse < best['rmse']:
                best = {
                    'alpha': alpha,
                    'run': run,
                    'weights': weights,
                    'totals': totals,
                    'rmse': rmse,
                }
    if not best:
        print('Не удалось найти решение')
        return
    print('Сравнение нутриентов:')
    print(f"{'Нутриент':<45}{'Цель':>10}{'Рацион':>10}")
    for key in NUT_KEYS:
        tgt = targets[key]
        act = best['totals'][key]
        label = KEY_TO_RUS[key].replace('\n', ' ')
        if key == 'calories':
            print(f"{label:<45}{tgt:>10.1f}{act:>10.1f}")
        else:
            print(f"{label:<45}{tgt:>10.0f}{act:>10.0f}")
    print(
        f"\nМинимальная RMSE: {best['rmse']:.3f} при Альфа={best['alpha']} (прогон {best['run']})"
    )
    print('Продукты и граммовки:')
    for prod, w in zip(products, best['weights']):
        if w > 0:
            print(f"- {prod['name']}: {w:.2f} г")


if __name__ == '__main__':
    main()

