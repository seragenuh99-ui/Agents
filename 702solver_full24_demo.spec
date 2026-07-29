# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['demo_dashboard.py'],
    pathex=[],
    binaries=[],
    datas=[('output\\results\\full24_controlled_comparison.json', 'output\\results'), ('docs\\full24_controlled_comparison.md', 'docs'), ('src', 'src'), ('experiments', 'experiments'), ('run.py', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
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
