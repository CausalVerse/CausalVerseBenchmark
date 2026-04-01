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
