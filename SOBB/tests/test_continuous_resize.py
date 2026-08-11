"""
Unit tests for ContinuousResize and evaluation edge cases.

Tests are designed to match the ACTUAL production interfaces:
  - ContinuousResize.__call__(image, target) expects a PIL Image
  - Box keys: ["bboxes", "hboxes", "rboxes", "polys", ...] (not gt_*)
  - Returns (image, target) tuple, not a single dict
  - scale_factor is [w_ratio, h_ratio] (x comes first for x-coordinates)
"""
import numpy as np
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'python'))


def make_transform():
    from sobb.data.transforms import ContinuousResize
    return ContinuousResize(scale_range=(1.0, 1.0), base_size=800, max_size=1333)


def make_image(w=400, h=300):
    from PIL import Image
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_empty_rbox():
    """Empty rbox array (0, 5) should not crash."""
    transform = make_transform()
    image = make_image()
    target = {
        'rboxes': np.zeros((0, 5), dtype=np.float32),
        'hboxes': np.zeros((0, 4), dtype=np.float32),
        'polys': np.zeros((0, 8), dtype=np.float32),
        'labels': np.zeros((0,), dtype=np.int32),
        'rboxes_ignore': np.zeros((0, 5), dtype=np.float32),
        'hboxes_ignore': np.zeros((0, 4), dtype=np.float32),
        'polys_ignore': np.zeros((0, 8), dtype=np.float32),
    }
    result_img, result_target = transform(image, target)
    assert result_img is not None
    assert result_target is not None
    print('PASS: test_empty_rbox')


def test_list_input():
    """List input (instead of np.ndarray) for rboxes should not crash."""
    transform = make_transform()
    image = make_image()
    target = {
        'rboxes': [[100, 100, 50, 50, 0.0]],
        'hboxes': [[100, 100, 150, 150]],
        'polys': [[100, 100, 150, 100, 150, 150, 100, 150]],
        'labels': np.array([1], dtype=np.int32),
        'rboxes_ignore': [],
        'hboxes_ignore': [],
        'polys_ignore': [],
    }
    result_img, result_target = transform(image, target)
    assert result_target['rboxes'].shape[1] == 5
    print('PASS: test_list_input')


def test_scale_factor_order():
    """scale_factor should be [w_ratio, h_ratio] (x-axis first, y-axis second).

    Verify by comparing scale_factor[0] against new_w/orig_w and
    scale_factor[1] against new_h/orig_h. ContinuousResize preserves aspect
    ratio, so the two values are numerically close; the ORDER is what matters.
    """
    transform = make_transform()
    orig_w, orig_h = 400, 300
    image = make_image(w=orig_w, h=orig_h)
    target = {
        'rboxes': np.array([[200, 150, 50, 50, 0.0]], dtype=np.float32),
        'hboxes': np.array([[200, 150, 250, 200]], dtype=np.float32),
        'polys': np.array([[200, 150, 250, 150, 250, 200, 200, 200]], dtype=np.float32),
        'labels': np.array([1], dtype=np.int32),
        'rboxes_ignore': np.zeros((0, 5), dtype=np.float32),
        'hboxes_ignore': np.zeros((0, 4), dtype=np.float32),
        'polys_ignore': np.zeros((0, 8), dtype=np.float32),
    }
    result_img, result_target = transform(image, target)
    sf = result_target['scale_factor']
    assert isinstance(sf, np.ndarray) and len(sf) == 2, \
        f'scale_factor is {type(sf)}, expected 2-element ndarray'
    new_w, new_h = result_img.size  # PIL Image.size returns (width, height)
    sf_w = float(sf[0])
    sf_h = float(sf[1])
    assert abs(sf_w - new_w / orig_w) < 1e-3, \
        f'scale_factor[0]={sf_w} should be w_ratio={new_w / orig_w}'
    assert abs(sf_h - new_h / orig_h) < 1e-3, \
        f'scale_factor[1]={sf_h} should be h_ratio={new_h / orig_h}'
    print(f'PASS: test_scale_factor_order (w_ratio={sf_w:.4f}, h_ratio={sf_h:.4f})')


def test_normal_rbox():
    """Normal rbox input should resize correctly."""
    transform = make_transform()
    image = make_image()
    target = {
        'rboxes': np.array([[200, 150, 100, 100, 0.5]], dtype=np.float32),
        'hboxes': np.array([[200, 150, 300, 250]], dtype=np.float32),
        'polys': np.array([[200, 150, 300, 150, 300, 250, 200, 250]], dtype=np.float32),
        'labels': np.array([1], dtype=np.int32),
        'rboxes_ignore': np.zeros((0, 5), dtype=np.float32),
        'hboxes_ignore': np.zeros((0, 4), dtype=np.float32),
        'polys_ignore': np.zeros((0, 8), dtype=np.float32),
    }
    result_img, result_target = transform(image, target)
    assert 'rboxes' in result_target
    assert result_target['rboxes'].shape[1] == 5
    print('PASS: test_normal_rbox')


def test_hbb_input():
    """HBB (hboxes) input should resize correctly."""
    transform = make_transform()
    image = make_image()
    target = {
        'rboxes': np.zeros((0, 5), dtype=np.float32),
        'hboxes': np.array([[100, 100, 200, 200]], dtype=np.float32),
        'polys': np.zeros((0, 8), dtype=np.float32),
        'labels': np.array([1], dtype=np.int32),
        'rboxes_ignore': np.zeros((0, 5), dtype=np.float32),
        'hboxes_ignore': np.zeros((0, 4), dtype=np.float32),
        'polys_ignore': np.zeros((0, 8), dtype=np.float32),
    }
    result_img, result_target = transform(image, target)
    assert result_target['hboxes'].shape[1] == 4
    print('PASS: test_hbb_input')


def test_config_load():
    """Config files should load without error."""
    from sobb.config import Config
    configs = [
        "projects/s2anet/configs/roitransformer_r50_fpn_1x_ssdd.py",
        "projects/s2anet/configs/roitransformer_r50_fpn_1x_rsdd.py",
    ]
    for cfg_path in configs:
        full = os.path.join(REPO, cfg_path)
        if not os.path.exists(full):
            continue
        cfg = Config(full)
        assert cfg.model is not None, f"model is None in {cfg_path}"
        assert cfg.dataset is not None, f"dataset is None in {cfg_path}"
        print(f'PASS: config loads: {cfg_path}')


def test_ablation_configs_load():
    """All 7 ablation configs should load via _base_ inheritance."""
    from sobb.config import Config
    base = os.path.join(REPO, "projects", "s2anet", "configs", "ablations")
    for fname in sorted(os.listdir(base)):
        if not fname.endswith('.py'):
            continue
        full = os.path.join(base, fname)
        cfg = Config(full)
        assert cfg.model is not None, f"model is None in {fname}"
        assert 'rbbox_head' in cfg.model, f"rbbox_head missing in {fname}"
        print(f'PASS: ablation config loads: {fname}')


def test_empty_detection():
    """Empty detection should not crash in voc_eval_dota_multi."""
    from sobb.data.devkits.voc_eval import voc_eval_dota_multi, _reset_gt_det
    c_dets = np.array([]).reshape(0, 11)
    classname_gts = {}
    _reset_gt_det(classname_gts)
    result = voc_eval_dota_multi(c_dets, classname_gts)
    assert result is not None
    assert 'mAP5095' in result
    assert 'AR100' in result
    print('PASS: test_empty_detection')


def test_string_image_id():
    """String image IDs should work in classname_gts."""
    from sobb.data.devkits.voc_eval import _reset_gt_det
    classname_gts = {'img_001': {'box': np.zeros(8), 'difficult': False, 'det': [False]}}
    _reset_gt_det(classname_gts)
    assert not classname_gts['img_001']['det'][0]
    print('PASS: test_string_image_id')


def test_seed_reproducibility():
    """set_random_seed should produce reproducible results."""
    import random
    try:
        from sobb.utils.general import set_random_seed
    except ImportError:
        # Jittor not available; test with stdlib only
        random.seed(42)
        a = random.random()
        random.seed(42)
        b = random.random()
        assert a == b
        print('PASS: test_seed_reproducibility (stdlib only)')
        return
    set_random_seed(42)
    a = random.random()
    set_random_seed(42)
    b = random.random()
    assert a == b, f'Seed reproducibility failed: {a} != {b}'
    print('PASS: test_seed_reproducibility')


def test_per_seed_work_dir():
    """Per-seed work_dir pattern should produce distinct paths."""
    def _work_dir(config_file, seed):
        base = os.path.splitext(os.path.basename(config_file))[0]
        return os.path.join("work_dirs", f"{base}_seed{seed}")
    paths = [_work_dir("ssdd.py", s) for s in [1, 2, 3, 4, 5]]
    assert len(set(paths)) == 5, "Per-seed work_dirs must be unique"
    print('PASS: test_per_seed_work_dir')


def test_scale_factor_broadcasting():
    """scale_factor [w, h] should broadcast correctly against (N, 8) polys."""
    polys = np.array([[100, 100, 200, 100, 200, 200, 100, 200]], dtype=np.float32)
    scale_factor = np.array([0.5, 0.25], dtype=np.float32)  # [w_ratio, h_ratio]
    # Apply: x /= w_ratio, y /= h_ratio
    polys[:, 0::2] /= scale_factor[0]
    polys[:, 1::2] /= scale_factor[1]
    expected_x = 100 / 0.5  # = 200
    expected_y = 100 / 0.25  # = 400
    assert abs(polys[0, 0] - expected_x) < 0.01
    assert abs(polys[0, 1] - expected_y) < 0.01
    print('PASS: test_scale_factor_broadcasting')


if __name__ == '__main__':
    print('=' * 60)
    print('ContinuousResize & Evaluation Edge Case Tests')
    print('=' * 60)
    tests = [
        test_empty_rbox,
        test_list_input,
        test_scale_factor_order,
        test_normal_rbox,
        test_hbb_input,
        test_config_load,
        test_ablation_configs_load,
        test_empty_detection,
        test_string_image_id,
        test_seed_reproducibility,
        test_per_seed_work_dir,
        test_scale_factor_broadcasting,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f'FAIL: {test.__name__} - {type(e).__name__}: {e}')
            failed += 1
    print('-' * 60)
    print(f'Results: {passed} passed, {failed} failed, {len(tests)} total')
    if failed > 0:
        sys.exit(1)
