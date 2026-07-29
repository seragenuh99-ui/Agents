#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

dnf install -y \
  python3-pip \
  findutils \
  file \
  binutils \
  mesa-libGL \
  mesa-libEGL \
  libX11 \
  libX11-xcb \
  libxcb \
  libxkbcommon-x11 \
  xcb-util \
  xcb-util-cursor \
  xcb-util-image \
  xcb-util-keysyms \
  xcb-util-renderutil \
  xcb-util-wm \
  libxkbcommon \
  fontconfig \
  freetype \
  dbus-libs \
  libtiff \
  gtk3 \
  gdk-pixbuf2 \
  cairo \
  pango \
  atk \
  harfbuzz
python3 -m pip install --no-cache-dir -r requirements.txt PySide6 pyinstaller

rm -rf build/702solver_full24_demo_openeuler dist/702solver_full24_demo_openeuler
python3 -m PyInstaller --clean --noconfirm 702solver_full24_demo_openeuler.spec
rm -f dist/702solver_full24_demo_openeuler/_internal/PySide6/Qt/plugins/imageformats/libqtiff.so

file dist/702solver_full24_demo_openeuler/702solver_full24_demo
dist/702solver_full24_demo_openeuler/702solver_full24_demo --help
du -sh dist/702solver_full24_demo_openeuler
