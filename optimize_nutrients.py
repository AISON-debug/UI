import numpy as np


def compute_calories(nutrients):
    """Calculate calories from macronutrient amounts.

    Parameters
    ----------
    nutrients : array-like
        Sequence containing grams of protein, fat and carbohydrates respectively.

    Returns
    -------
    float
        Calculated caloric value.
    """
    protein, fat, carbs = nutrients
    return protein * 4 + fat * 9 + carbs * 4


def rmse(predicted, target, calorie_weight=None):
    """Calculate root mean squared error between vectors.

    By default only gram based nutrients are compared.  When ``calorie_weight``
    is provided the difference in calories is appended to the error vector and
    scaled by the weight to avoid calories dominating the RMSE.

    Parameters
    ----------
    predicted : array-like
        Predicted nutrient values in grams.
    target : array-like
        Target nutrient values in grams.
    calorie_weight : float, optional
        When provided, include the difference in calories in the RMSE and scale
        it by this weight so that its impact is comparable with gram based
        nutrients.
    """
    predicted = np.array(predicted, dtype=float)
    target = np.array(target, dtype=float)

    diffs = predicted - target

    if calorie_weight is not None:
        pred_calories = compute_calories(predicted)
        target_calories = compute_calories(target)
        diffs = np.append(diffs, (pred_calories - target_calories) * calorie_weight)

    return np.sqrt(np.mean(diffs ** 2))


def main():
    # Example: protein, fat, carbs in grams
    predicted = [25, 10, 40]
    target = [30, 8, 50]

    # Previously calories were appended to ``target`` which distorted RMSE.
    # Now RMSE is calculated strictly on gram based nutrients.
    print(f"RMSE without calories: {rmse(predicted, target):.2f}")

    # Caloric contribution can optionally be included using a small weight to
    # normalise it relative to gram based nutrients.
    print(f"RMSE with calories (weight 0.01): {rmse(predicted, target, calorie_weight=0.01):.2f}")


if __name__ == "__main__":
    main()
