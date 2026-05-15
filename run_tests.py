"""Cross-platform test runner for FireANTs."""
import argparse
import subprocess
import sys


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_basic_tests():
    print("Running basic tests...")
    run([sys.executable, "-m", "pytest", "-v",
         "--log-cli-level=INFO", "tests/"])


def run_distributed_tests():
    torchrun = [sys.executable, "-m", "torch.distributed.run"]

    print("Running single GPU tests...")
    run(torchrun + ["--nproc-per-node=1", "-m", "pytest", "-v",
                     "tests/distributed/test_image_io.py"])

    print("Running parallel state tests with 4 GPUs...")
    run(torchrun + ["--nproc-per-node=4", "-m", "pytest", "-v",
                     "tests/distributed/test_parallel_state.py"])

    print("Running distributed greedy tests with 4 GPUs...")
    run(torchrun + ["--nproc-per-node=4", "-m", "pytest", "-v",
                     "tests/distributed/test_distributed_greedy.py"])

    print("Running ring sampler tests with multiple GPU configs...")
    for n in [2, 3, 4, 7, 8]:
        print(f"Testing with {n} GPUs...")
        run(torchrun + [f"--nproc-per-node={n}", "-m", "pytest", "-v",
                         "tests/distributed/test_ring_sampler.py"])


def main():
    parser = argparse.ArgumentParser(description="FireANTs test runner")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests (basic + distributed)")
    parser.add_argument("--distributed-only", action="store_true",
                        help="Run distributed tests only")
    args = parser.parse_args()

    if args.all:
        run_basic_tests()
        run_distributed_tests()
    elif args.distributed_only:
        run_distributed_tests()
    else:
        run_basic_tests()

    print("\nTests completed successfully!")


if __name__ == "__main__":
    main()
