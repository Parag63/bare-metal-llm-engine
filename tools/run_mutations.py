#!/usr/bin/env python3
"""tools/run_mutations.py -- Reproducible mutation proof for FlashDecoding tests.

Applies each fault patch in tools/mutations/ to kernels/attention_decode.cu,
rebuilds engine_tests, runs kernels.attention_decode_long_sequence, asserts that
the test FAILS (exit code != 0), reverts the patch cleanly, and finally rebuilds
and verifies the unmutated kernel passes.

Usage:
  python3 tools/run_mutations.py
  python3 tools/run_mutations.py --build-dir build --filter attention_decode_long_sequence
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=check)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FlashDecoding tests catch injected faults.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory (default: build)")
    parser.add_argument("--mutations-dir", default="tools/mutations", help="Directory containing .patch files")
    parser.add_argument("--filter", default="attention_decode_long_sequence", help="engine_tests --filter pattern")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 4, help="Parallel build jobs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    mut_dir = repo_root / args.mutations_dir
    patches = sorted(mut_dir.glob("*.patch"))
    if not patches:
        print(f"No .patch files found in {mut_dir}", file=sys.stderr)
        return 2

    status = run(["git", "status", "--porcelain", "--", "kernels/attention_decode.cu"], repo_root)
    if status.stdout.strip():
        print("Error: kernels/attention_decode.cu has uncommitted changes; commit or stash first.", file=sys.stderr)
        return 2

    test_bin = repo_root / args.build_dir / "bin" / "engine_tests"
    caught = 0
    failed_to_catch = []

    try:
        for patch in patches:
            print(f"\n=== Mutation: {patch.name} ===")
            run(["git", "apply", str(patch)], repo_root)
            try:
                build_res = run(
                    ["cmake", "--build", args.build_dir, "--target", "engine_tests", "-j", str(args.jobs)],
                    repo_root,
                    check=False,
                )
                if build_res.returncode != 0:
                    print(build_res.stdout)
                    print(f"[ERROR] Build failed on {patch.name}", file=sys.stderr)
                    return 2

                test_res = run([str(test_bin), f"--filter={args.filter}"], repo_root, check=False)
                tail = "\n".join(test_res.stdout.strip().splitlines()[-12:])
                print(tail)
                if test_res.returncode != 0:
                    print(f"[CAUGHT] {patch.name} failed {args.filter} as expected (exit={test_res.returncode}).")
                    caught += 1
                else:
                    print(f"[MISSED] {patch.name} unexpectedly PASSED {args.filter}!", file=sys.stderr)
                    failed_to_catch.append(patch.name)
            finally:
                run(["git", "checkout", "--", "kernels/attention_decode.cu"], repo_root)
    finally:
        run(["git", "checkout", "--", "kernels/attention_decode.cu"], repo_root)
        print("\n=== Restoring clean build and verifying baseline ===")
        run(["cmake", "--build", args.build_dir, "--target", "engine_tests", "-j", str(args.jobs)], repo_root)
        clean_res = run([str(test_bin), f"--filter={args.filter}"], repo_root, check=False)
        print("\n".join(clean_res.stdout.strip().splitlines()[-6:]))
        if clean_res.returncode != 0:
            print("[ERROR] Clean baseline failed after mutation cleanup!", file=sys.stderr)
            return 1

    print(f"\nSummary: caught {caught}/{len(patches)} injected faults.")
    if caught != len(patches) or failed_to_catch:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

