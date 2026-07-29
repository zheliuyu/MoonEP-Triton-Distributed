# MoonEP-TD: no C++/CUDA extension.

MoonEP uses `csrc/` for CUDA VMM + NVSwitch multicast (`moonep._C`).

This project replaces that layer with **NVSHMEM** via Triton-distributed
(see `moonep_td/buffer.py`).

There is intentionally no `moonep_td._C` module.
