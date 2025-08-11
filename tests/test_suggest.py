import sys, types

# Stub external dependencies to import app without heavy installs
class _Flask:
    def __init__(self, *a, **k):
        self.config = {}
    def route(self, *a, **k):
        def deco(f):
            return f
        return deco

flask_stub = types.SimpleNamespace(
    Flask=_Flask,
    render_template=lambda *a, **k: None,
    request=None,
    send_file=None,
    jsonify=lambda x, **k: x,
    abort=lambda *a, **k: None,
)
sys.modules.setdefault('flask', flask_stub)
sys.modules.setdefault('whisper', types.SimpleNamespace(load_audio=lambda *a, **k: [0.0], load_model=lambda *a, **k: None))
sys.modules.setdefault('torch', types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
sys.modules.setdefault('numpy', types.SimpleNamespace(save=lambda *a, **k: None, load=lambda *a, **k: [0.0]))

import app

def test_analyze_models_has_entries():
    res = app.analyze_models(60.0)
    assert isinstance(res, list) and res
    assert any(r['model'] == 'base' for r in res)
    for entry in res:
        assert 'eta_seconds' in entry and 'memory_gb' in entry

def test_suggest_model_returns_supported():
    rec = app.suggest_model(30.0)
    assert rec in app.SUPPORTED_MODELS
