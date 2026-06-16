import os
import sys
import time
import base64
from io import BytesIO

# Force unbuffered output so logs appear immediately in the terminal
sys.stdout.reconfigure(line_buffering=True)

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np

from utils.metrics import laplacian_variance
import config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

_models: dict = {}

MODEL_NAMES = ['restormer', 'realesrgan', 'diffir']


def _get_model(name: str):
    if name in _models:
        return _models[name]

    if name == 'restormer':
        from models.restormer_model import RestormerModel
        m = RestormerModel()
    elif name == 'realesrgan':
        from models.realesrgan_model import RealESRGANModel
        m = RealESRGANModel()
    elif name == 'diffir':
        from models.diffir_model import DiffIRModel
        m = DiffIRModel()
    else:
        return None

    m.load()
    _models[name] = m
    return m


def _pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _resize_if_needed(img: Image.Image) -> Image.Image:
    limit = config.MAX_IMAGE_SIZE
    w, h = img.size
    if max(w, h) > limit:
        scale = limit / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    result = {}
    for name in MODEL_NAMES:
        try:
            m = _get_model(name)
            result[name] = {
                'loaded': m.loaded,
                'device': str(m.device) if m.loaded else 'N/A',
                'error': getattr(m, 'load_error', None),
            }
        except Exception as e:
            result[name] = {'loaded': False, 'device': 'N/A', 'error': str(e)}
    return jsonify(result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    model_name = request.form.get('model', 'restormer')
    if model_name not in MODEL_NAMES:
        return jsonify({'error': f'Unknown model: {model_name}'}), 400

    file = request.files['image']
    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    img = _resize_if_needed(img)
    input_score = laplacian_variance(img)

    try:
        model = _get_model(model_name)
        if model is None or not model.loaded:
            err = getattr(model, 'load_error', 'Model not available') if model else 'Model not found'
            return jsonify({'error': err}), 503

        print(f'[{model_name}] Running inference on {img.size[0]}x{img.size[1]} image ...')
        t0 = time.time()
        result_img = model.predict(img)
        elapsed = round(time.time() - t0, 3)
        print(f'[{model_name}] Done in {elapsed}s')

        output_score = laplacian_variance(result_img)
        improvement = round(((output_score - input_score) / max(input_score, 1e-6)) * 100, 1)

        return jsonify({
            'input_b64':      _pil_to_b64(img),
            'output_b64':     _pil_to_b64(result_img),
            'input_score':    round(input_score, 2),
            'output_score':   round(output_score, 2),
            'improvement':    improvement,
            'inference_time': elapsed,
            'model':          model_name,
            'width':          result_img.width,
            'height':         result_img.height,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_all', methods=['POST'])
def api_predict_all():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    img = _resize_if_needed(img)
    input_score = laplacian_variance(img)
    input_b64 = _pil_to_b64(img)

    results = {'input_b64': input_b64, 'input_score': round(input_score, 2), 'models': {}}

    for name in MODEL_NAMES:
        try:
            model = _get_model(name)
            if model is None or not model.loaded:
                err = getattr(model, 'load_error', 'Not available') if model else 'Not found'
                results['models'][name] = {'error': err}
                continue

            print(f'[{name}] Running inference on {img.size[0]}x{img.size[1]} image ...')
            t0 = time.time()
            out_img = model.predict(img)
            elapsed = round(time.time() - t0, 3)
            print(f'[{name}] Done in {elapsed}s')

            out_score = laplacian_variance(out_img)
            results['models'][name] = {
                'output_b64':     _pil_to_b64(out_img),
                'output_score':   round(out_score, 2),
                'improvement':    round(((out_score - input_score) / max(input_score, 1e-6)) * 100, 1),
                'inference_time': elapsed,
            }
        except Exception as e:
            results['models'][name] = {'error': str(e)}

    return jsonify(results)


def _preload_all_models():
    print('=' * 50, flush=True)
    print('  Image Deblurring Website', flush=True)
    print('  Pre-loading all models at startup...', flush=True)
    print('=' * 50, flush=True)
    for name in MODEL_NAMES:
        _get_model(name)
    print('=' * 50, flush=True)
    loaded = [n for n in MODEL_NAMES if _models.get(n) and _models[n].loaded]
    failed = [n for n in MODEL_NAMES if n not in loaded]
    print(f'  Ready: {", ".join(loaded) if loaded else "-"}', flush=True)
    if failed:
        print(f'  Failed: {", ".join(failed)}', flush=True)
    print('  Open browser: http://localhost:5000', flush=True)
    print('=' * 50 + '\n', flush=True)


if __name__ == '__main__':
    _preload_all_models()
    app.run(debug=False, host='0.0.0.0', port=5000)
