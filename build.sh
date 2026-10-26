#!/usr/bin/env bash
# build.sh (F64 port): host programs with cc into build/host; card programs by the Intel-Phi-LLVM S3 clang (-march=knc)
# and linked static by the stack's knc-cc into build/card; phi-isa-audit checks each card binary.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$here" in /mnt/*) [ "$(findmnt -no SOURCE /mnt)" = /dev/sda2 ] || { echo "build.sh: /mnt is not /dev/sda2" >&2; exit 1; } ;; esac
out="$here/build"
mkdir -p "$out/host" "$out/card" "$out/obj"
flags=(-O2 -std=gnu11 -Wall -Wextra)
VCC="${PHI_LLVM_BIN:-$HOME/.cache/intel-phi-llvm-build/llvm/bin}/clang"
SYS="${PHI_SYSROOT_7120A:-$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
knc() {
	local name=$1 objs=() s o
	shift
	for s in "$@"; do
		o="$out/obj/$(basename "${s%.c}")-${name}-knc.o"
		"$VCC" --target=x86_64-unknown-linux-musl -march=knc --sysroot="$SYS" "${flags[@]}" ${defs:-} -fno-jump-tables \
			-falign-loops=1 -fPIE -c -o "$o" "$here/$s"
		objs+=("$o")
	done
	knc-cc "${flags[@]}" -static -o "$out/card/$name" "${objs[@]}" -lm -lpthread
	printf '%s: ' "$name"
	"$audit" "$out/card/$name" | tail -1 || true
}
# shellcheck disable=SC1091
. "${PHI_TREE:-$HOME/Intel Phi 7120A}/toolchain/env.sh"
audit="${PHI_TREE:-$HOME/Intel Phi 7120A}/host/target/debug/phi-isa-audit"

# F64 ISA gate
knc isa card/isa.c
cc "${flags[@]}" -o "$out/host/isa-host" "$here/tools/isa-host.c" -lm

# Stage 1 (F64): one column forward
knc col-gate tools/col-gate.c
cc "${flags[@]}" -ffp-contract=off -o "$out/host/col-ref" "$here/tools/col-ref.c" -lm

# Stage 3 (F64): training reference, gates and checkers
cc "${flags[@]}" -O2 -o "$out/host/train-ref" "$here/tools/train-ref.c" -lm
knc train-gate tools/train-gate.c
cc "${flags[@]}" -O2 -o "$out/host/train-check" "$here/tools/train-check.c" -lm
cc "${flags[@]}" -O2 -o "$out/host/ctrain-ref" "$here/tools/ctrain-ref.c" -lm

# Stage 4 (F64): the cluster gate and checker
knc cluster-gate tools/cluster-gate.c
cc "${flags[@]}" -O2 -o "$out/host/ctrain-check" "$here/tools/ctrain-check.c" -lm

# Stage 5 (F64): the generation daemon and its host client
knc cortexd card/cortexd.c card/cdma.c
cc "${flags[@]}" -O2 -o "$out/host/cortex" "$here/host/cortex.c" -lm
cc "${flags[@]}" -O2 -o "$out/host/bpe" "$here/tools/bpe.c"

# the coach (host)
cc "${flags[@]}" -O2 -Wno-format-truncation -Wno-stringop-truncation -o "$out/host/coach" "$here/host/coach.c" -lm
