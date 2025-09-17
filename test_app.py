import json
import shutil
from pathlib import Path
import importlib

import pytest

import optimize_nutrients

FIELDNAMES = ["Продукт","Белки","Насыщенные","НЕнасыщенные","Простые",
              "Сложные перевариваемые","Растворимая","Нерастворимая",
              "ККал","Макс. порций","Шаг"]
INPUT_FIELDS = [f for f in FIELDNAMES if f != "ККал"]


def calc_kcal(data: dict) -> float:
    g = lambda k: float(data[k])
    return round(
        g("Белки") * 4
        + (g("Насыщенные") + g("НЕнасыщенные")) * 9
        + (g("Простые") + g("Сложные перевариваемые")) * 4
        + (g("Растворимая") + g("Нерастворимая")) * 1.5,
        2,
    )

@pytest.fixture
def client(tmp_path, monkeypatch):
    src = Path(__file__).with_name('Nutrients DB.csv')
    dst = tmp_path / 'db.csv'
    shutil.copy(src, dst)
    monkeypatch.setenv('NUTRIENTS_DB', str(dst))
    import app
    importlib.reload(app)
    return app.app.test_client()


def sample_product(name):
    return {fn: '1' for fn in INPUT_FIELDS} | {"Продукт": name}


def test_crud_flow(client):
    resp = client.get('/api/products')
    assert resp.status_code == 200
    data = resp.get_json()
    base_count = len(data)

    new_prod = sample_product('Тест')
    resp = client.post('/api/products', json=new_prod)
    assert resp.status_code == 201

    resp = client.get('/api/products')
    products = resp.get_json()
    assert len(products) == base_count + 1
    created = next(p for p in products if p['Продукт'] == 'Тест')
    assert created['ККал'] == f"{calc_kcal(new_prod):.2f}"

    updated = sample_product('Тест')
    updated['Белки'] = '2'
    resp = client.put('/api/products/Тест', json=updated)
    assert resp.status_code == 200

    resp = client.get('/api/products')
    products = resp.get_json()
    edited = next(p for p in products if p['Продукт'] == 'Тест')
    assert edited['ККал'] == f"{calc_kcal(updated):.2f}"

    resp = client.delete('/api/products/Тест')
    assert resp.status_code == 200

    resp = client.get('/api/products')
    assert len(resp.get_json()) == base_count


def test_optimize_returns_breakdown():
    payload = {
        'targets': {
            'proteins': 10,
            'saturated': 5,
            'unsaturated': 5,
            'simple': 5,
            'complex': 5,
            'soluble': 5,
            'insoluble': 5,
            'calories': 100,
        },
        'run_count': 1,
        'residual_share': 0,
        'allow_zero_weights': True,
        'products': [
            {
                'name': 'Test Product',
                'step': 1,
                'max_weight': 100,
                'fix_weight': True,
                'fixed_weight': 100,
            }
        ],
    }
    product_db = {
        'Test Product': {
            'proteins': 10,
            'saturated': 5,
            'unsaturated': 5,
            'simple': 5,
            'complex': 5,
            'soluble': 5,
            'insoluble': 5,
            'calories': 100,
            'kcal': 100,
        }
    }

    result = optimize_nutrients.optimize_from_payload(payload, product_db=product_db)

    assert 'nutrient_breakdown' in result
    breakdown = result['nutrient_breakdown']
    assert isinstance(breakdown, dict)
    assert breakdown['products'], 'breakdown should contain product rows'
    product_row = breakdown['products'][0]
    assert pytest.approx(product_row['nutrients']['proteins'], rel=1e-4) == 10
    assert pytest.approx(product_row['nutrients']['calories'], rel=1e-4) == 100
    totals = breakdown['totals']
    assert pytest.approx(totals['proteins'], rel=1e-4) == 10
    assert pytest.approx(totals['calories'], rel=1e-4) == 100
