#!/usr/bin/env bash

init_new_paths() {
  local caller_dir
  caller_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"

  export NEW_ROOT
  NEW_ROOT="$(cd "${caller_dir}/../.." && pwd)"

  export WORKSPACE_ROOT
  WORKSPACE_ROOT="$(cd "${NEW_ROOT}/.." && pwd)"

  export CODE_DIR
  CODE_DIR="${NEW_ROOT}/src"
}

normalize_fall_latent_view() {
  case "$1" in
    front|frontview)
      printf '%s\n' "front"
      ;;
    left|leftview)
      printf '%s\n' "left"
      ;;
    right|rightview)
      printf '%s\n' "right"
      ;;
    bird|birdview)
      printf '%s\n' "bird"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

normalize_fixed_robotics_latent_view() {
  case "$1" in
    front|frontview)
      printf '%s\n' "front"
      ;;
    side|sideview)
      printf '%s\n' "side"
      ;;
    bird|birdview)
      printf '%s\n' "bird"
      ;;
    agent|agentview)
      printf '%s\n' "agent"
      ;;
    eye|robot0_eye_in_hand)
      printf '%s\n' "robot0_eye_in_hand"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

latent_view_suffix() {
  local view="$1"
  printf '_%s\n' "${view}"
}
