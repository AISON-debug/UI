from flask import Flask, jsonify, request, send_from_directory

from optimize_nutrients import load_product_database, optimize_from_payload

app = Flask(__name__, static_folder='.', static_url_path='')


@app.route('/')
def index():
    """Serve the optimisation UI."""
    return send_from_directory('.', 'UI.html')


@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """Оптимизировать рацион на основе данных из UI."""
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'error': 'Некорректный JSON.'}), 400

    try:
        product_db = load_product_database()
        result = optimize_from_payload(payload, product_db=product_db)
    except ValueError as exc:  # ошибки валидации данных
        return jsonify({'error': str(exc)}), 400
    except Exception:
        return jsonify({'error': 'Не удалось выполнить оптимизацию.'}), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
