#!/bin/bash
#SBATCH --job-name=nauchka
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Universal SLURM entrypoint for this project.
# Choose mode with APP_MODE:
#   APP_MODE=gpu_topology
#   APP_MODE=network_test
#   APP_MODE=interconnect_benchmark
#   APP_MODE=clustering
#   APP_MODE=custom      (uses RUN_CMD)
#
# Optional vars:
#   PROJECT_ROOT=/path/to/network-tests2
#   NET_MAP=/path/to/inter-node-map.in  (required for gpu_topology mode)
#   APP_ARGS="..."                      (extra args for selected mode)
#   RUN_CMD="binary --flag value"       (required for custom mode)

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
APP_MODE="${APP_MODE:-gpu_topology}"
APP_ARGS="${APP_ARGS:-}"
NET_MAP="${NET_MAP:-}"
RUN_CMD="${RUN_CMD:-}"

cd "${PROJECT_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

resolve_binary() {
  local name="$1"
  shift
  local c
  for c in "$@"; do
    if [[ -x "${c}" ]]; then
      printf "%s\n" "${c}"
      return 0
    fi
  done
  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
    return 0
  fi
  return 1
}

run_gpu_topology() {
  local gpu_dir="${PROJECT_ROOT}/src/gpu_tests"
  local input_dir="${gpu_dir}/input"
  local nnodes="${SLURM_JOB_NUM_NODES:-1}"
  local i
  local -a node_args

  if [[ -z "${NET_MAP}" ]]; then
    echo "ERROR: NET_MAP is required for APP_MODE=gpu_topology"
    echo "Example: sbatch --export=ALL,APP_MODE=gpu_topology,NET_MAP=src/gpu_tests/input/1_8.in run_project.slurm"
    exit 2
  fi

  mkdir -p "${input_dir}"
  srun --ntasks="${nnodes}" --ntasks-per-node=1 bash -lc '
    idx=${SLURM_NODEID:-0}
    out="'"${input_dir}"'/live_node${idx}.in"
    nvidia-smi topo -m > "${out}"
    echo "saved ${out} on $(hostname)"
  '

  for ((i=0; i<nnodes; i++)); do
    node_args+=("node${i}:${input_dir}/live_node${i}.in")
  done

  python3 "${gpu_dir}/gpu_topology.py" --net-map "${NET_MAP}" "${node_args[@]}"
}

run_network_test() {
  local bin
  bin="$(resolve_binary "network_test2" \
    "${PROJECT_ROOT}/src/network_test/network_test2" \
    "${PROJECT_ROOT}/bin/network_test2")" || {
    echo "ERROR: network_test2 not found. Build it first."
    exit 2
  }
  srun --ntasks="${SLURM_NTASKS:-1}" bash -lc "${bin} ${APP_ARGS}"
}

run_interconnect_benchmark() {
  local bin
  bin="$(resolve_binary "interconnect_benchmark" \
    "${PROJECT_ROOT}/src/interconnect_benchmark/interconnect_benchmark" \
    "${PROJECT_ROOT}/bin/interconnect_benchmark")" || {
    echo "ERROR: interconnect_benchmark not found. Build it first."
    exit 2
  }
  srun --ntasks="${SLURM_NTASKS:-1}" bash -lc "${bin} ${APP_ARGS}"
}

run_clustering() {
  local bin
  bin="$(resolve_binary "test_data_clustering" \
    "${PROJECT_ROOT}/src/clustering/test_data_clustering" \
    "${PROJECT_ROOT}/bin/test_data_clustering")" || {
    echo "ERROR: test_data_clustering not found. Build it first."
    exit 2
  }
  srun --ntasks=1 bash -lc "${bin} ${APP_ARGS}"
}

run_custom() {
  if [[ -z "${RUN_CMD}" ]]; then
    echo "ERROR: RUN_CMD is required for APP_MODE=custom"
    exit 2
  fi
  srun --ntasks="${SLURM_NTASKS:-1}" bash -lc "${RUN_CMD}"
}

case "${APP_MODE}" in
  gpu_topology) run_gpu_topology ;;
  network_test) run_network_test ;;
  interconnect_benchmark) run_interconnect_benchmark ;;
  clustering) run_clustering ;;
  custom) run_custom ;;
  *)
    echo "ERROR: unknown APP_MODE='${APP_MODE}'"
    exit 2
    ;;
esac
