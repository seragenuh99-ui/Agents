# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path.cwd()

common_excludes = [
    'sentence_transformers',
    'transformers',
    'torch',
    'torchvision',
    'torchaudio',
    'sklearn',
    'pandas',
    'tensorflow',
    'PySide6.QtDBus',
    'gi',
    'keyring',
    'SecretStorage',
    'jeepney',
    'cryptography',
    'bcrypt',
    'httplib2',
    'psutil',
]

a = Analysis(
    ['demo_dashboard.py'],
    pathex=[],
    binaries=[],
    datas=[
        (str(ROOT / 'output' / 'results' / 'full24_controlled_comparison.json'), 'output/results'),
        (str(ROOT / 'docs' / 'full24_controlled_comparison.md'), 'docs'),
        (str(ROOT / 'src'), 'src'),
        (str(ROOT / 'experiments'), 'experiments'),
        (str(ROOT / 'run.py'), '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=common_excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='702solver_full24_demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='702solver_full24_demo_openeuler',
)
