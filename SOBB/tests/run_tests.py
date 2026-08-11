"""
Run all SOBB edge case tests.

Usage:
    python tests/run_tests.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# S0-test: Import the actual test function names from test_continuous_resize.
# The previous version imported test_non_square_scale_factor which was
# renamed to test_scale_factor_order, causing ImportError before any test ran.
from test_continuous_resize import (
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
)

if __name__ == '__main__':
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
        except Exception:
            failed += 1
            import traceback
            traceback.print_exc()
    print(f'\nResults: {passed} passed, {failed} failed, {len(tests)} total')
    sys.exit(1 if failed > 0 else 0)
