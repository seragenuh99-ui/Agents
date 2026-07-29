#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

dnf install -y \
  mesa-libGL \
  mesa-libEGL \
  libglvnd \
  libglvnd-glx \
  libglvnd-egl \
  libdrm \
  libX11 \
  libXext \
  libXfixes \
  libXxf86vm \
  libxshmfence

target="dist/702solver_full24_demo_openeuler/_internal"
mkdir -p "$target"

for lib in \
  libGL.so.1 \
  libEGL.so.1 \
  libGLdispatch.so.0 \
  libGLX.so.0 \
  libOpenGL.so.0 \
  libglapi.so.0 \
  libdrm.so.2 \
  libXext.so.6 \
  libXfixes.so.3 \
  libXxf86vm.so.1 \
  libxcb.so.1 \
  libXau.so.6 \
  libxshmfence.so.1
do
  path="$(ldconfig -p | awk -v lib="$lib" '$1 == lib {print $NF; exit}')"
  if [[ -n "$path" ]]; then
    cp -L "$path" "$target/"
  fi
done
