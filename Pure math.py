# Входные данные
# T[k]        - целевые значения нутриентов (k = 0..K-1)
# r[k]        - текущий остаток нутриентов
# products[j] - список продуктов, для каждого j:
#   s[j]      - шаг порции (граммы)
#   p[j][k]   - содержание нутриента k в 100 г продукта j
#   max_weight[j] - максимальный допустимый вес продукта j
#   x[j]      - уже выбранный вес продукта j (начально 0)

# Вспомогательная функция: вклад нутриента k от одной порции продукта j
def portion_value(j, k):
    return (s[j] / 100.0) * p[j][k]

# Основной цикл оптимизации
while True:
    # 1. Для каждого нутриента вычислить минимальную долю alpha_k
    alpha_list = []
    for k in range(K):
        if r[k] > 0:
            ratios = []
            for j in range(len(products)):
                val = portion_value(j, k)
                if val > 0:
                    ratios.append(val / r[k])
            if len(ratios) > 0:
                alpha_list.append(min(ratios))
    if len(alpha_list) == 0:
        break  # все нутриенты закрыты

    # 2. Определить лимитирующую alpha
    alpha = max(alpha_list)

    # 3. Составить целевой вектор покрытия: target[k] = alpha * r[k]
    target = [alpha * r[k] for k in range(K)]

    # 4. Решить задачу минимизации:
    #     найти delta_x[j] >= 0, чтобы
    #     sum_j portion_value(j, k) * delta_x[j] ~ target[k]
    # Реализуется через метод наименьших квадратов с ограничениями
    delta_x = solve_nonnegative_least_squares(target, products, r)

    # 5. Применить ограничения по весу и округление
    for j in range(len(products)):
        # максимум допустимый прирост
        available = max_weight[j] - x[j]
        delta_x[j] = min(delta_x[j], available)
        # округление до шага
        delta_x[j] = round(delta_x[j] / s[j]) * s[j]
        if delta_x[j] < 0:
            delta_x[j] = 0

    # 6. Обновить веса и остаток
    any_added = False
    for j in range(len(products)):
        if delta_x[j] > 0:
            x[j] += delta_x[j]
            any_added = True
            for k in range(K):
                r[k] -= (p[j][k] / 100.0) * delta_x[j]

    # 7. Условия остановки
    if not any_added:
        break
    if all(val <= 0 for val in r):
        break
