"""VGAC reference implementation (PEARC '26 short paper).

Submodules:
    calibration  - post-hoc isotonic calibration helpers
    evaluation   - calibration metrics, bootstrap CIs, deterministic seeding
    features     - universal feature schema (cross-cluster)
    harness      - PSI / temporal-ECE drift sensors
    integration  - submit-time capture and policy translation
    ops          - sliding-window recalibrator
    policy       - graduated-intervention generator and gpu_ext bridge

The packaging here is deliberately small: this is a 4-page short paper's
artifact, not a production library. Modules are designed to be readable
top-to-bottom and to mirror the section structure of the paper.
"""
