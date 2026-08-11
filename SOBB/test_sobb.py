import jittor as jt
import numpy as np
from sobb.models.boxes.sobb_ops import sobb_encode, sobb_decode
import math

def test_sobb_cycle():
    # HBB: (x_ctr, y_ctr, w, h)
    hbb = jt.array([[100., 100., 200., 100.]])
    # OBB: (x_ctr, y_ctr, w, h, angle)
    # Let's create an OBB that is inscribed in this HBB.
    # At 30 degrees (pi/6):
    # W = w*cos + h*sin, H = w*sin + h*cos
    # This is tricky to do in reverse. Let's just pick an OBB and get its outer HBB.
    obb_w, obb_h, obb_a = 150.0, 50.0, math.pi / 6.0
    cos_a, sin_a = abs(math.cos(obb_a)), abs(math.sin(obb_a))
    GW = obb_w * cos_a + obb_h * sin_a
    GH = obb_w * sin_a + obb_h * cos_a
    
    hbb = jt.array([[100., 100., GW, GH]])
    obb = jt.array([[100., 100., obb_w, obb_h, obb_a]])
    
    print(f"Original OBB: {obb}")
    print(f"Outer HBB: {hbb}")
    
    # Encode
    deltas = sobb_encode(hbb, obb)
    print(f"Encoded SOBB deltas: {deltas}")
    
    # Decode
    decoded_obb = sobb_decode(hbb, deltas)
    print(f"Decoded OBB: {decoded_obb}")
    
    # Check if they match
    diff = jt.abs(obb - decoded_obb)
    print(f"Diff: {diff}")
    
    assert jt.all(diff < 1e-3)
    print("Test passed!")

if __name__ == "__main__":
    test_sobb_cycle()
