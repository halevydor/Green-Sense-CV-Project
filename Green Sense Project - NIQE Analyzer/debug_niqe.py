
import time
print("Starting debug script...", flush=True)
t0 = time.time()
try:
    import numpy as np
    print(f"Numpy imported in {time.time()-t0:.2f}s", flush=True)
    from PIL import Image
    print(f"Pillow imported in {time.time()-t0:.2f}s", flush=True)
    from niqe import niqe
    print(f"NIQE imported in {time.time()-t0:.2f}s", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

print("Done.", flush=True)
