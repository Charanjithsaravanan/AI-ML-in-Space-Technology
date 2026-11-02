# PyInstaller build specification for the native GEOSAT application.
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


root = Path(SPECPATH)
datas = [
    (str(root / "runs" / "best.pt"), "runs"),
    (str(root / "runs" / "cloud_screener.joblib"), "runs"),
    (str(root / "frontend" / "out"), "frontend/out"),
]

a = Analysis(
    [str(root / "geosat_app.py")],
    pathex=[str(root)],
    datas=datas,
    hiddenimports=collect_submodules("uvicorn"),
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="geosat",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="geosat",
)
