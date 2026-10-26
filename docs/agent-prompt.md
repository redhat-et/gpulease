# GPU access on this host (gpulease)

All GPU work on this machine must go through `gpulease`. Never set
`HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, or `CUDA_VISIBLE_DEVICES` yourself,
and never target GPUs that `gpulease status` reports as reserved.

- Foreground jobs (tests, benchmarks, short runs):

      gpulease run -n <N> -- <command>

  Blocks until N GPUs are free, exports the device env for the process, and
  releases on exit. Use `--wait 0` to fail fast instead of queueing.

- Long-lived servers (vllm serve, etc.):

      eval "$(gpulease checkout -n <N> --ttl <minutes>)"
      # launch the server with the exported env (nohup/setsid is fine)
      gpulease release <lease-id>   # when done

- Status: `gpulease status` or `gpulease status --json`
- If GPUs are busy, wait or ask for fewer GPUs; do not launch anyway.
- Optional selection: `--prefer 6` (use 6 if free, else any free pool GPU) or
  `--gpu 5,6` (exactly 5,6; waits until free; order = rank order).
- Claims and reservations are enforced automatically; never use `--ignore-claims`
  unless a human explicitly instructs you to, with a reason.

Examples:

    gpulease run -n 4 -- python -m pytest tests/kernels
    gpulease run -n 1 -- python bench_one_gpu.py
    gpulease run -n 2 --wait 0 -- python quick_check.py
