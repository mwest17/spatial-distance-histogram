## Spatial Distance Histogram
This project creates a histogram based on a normally distributed set of three-dimensional points.
It contains both a provided CPU-based implementation and a custom GPU implementation.

The GPU implementation started as a naive approach where every thread computed the distance from its own index to the top of the input array.

This suffered due to poor memory bandwidth utilization and atomic operation conflicts.

The current implementation now utilizes tiling of array sections, output privatization in shared memory, and improved memory coalescing.

The current, optimized implementation achieves nearly a 2x speedup compared to the naive GPU implementation on an input of 1,000,000 points. The results are shown below:

### Optimized:

<img width="781" height="381" alt="optimizedApproach" src="https://github.com/user-attachments/assets/70c9b8d3-97ca-475f-86a3-db37b83bdab7" />

### Naive:

<img width="774" height="381" alt="naiveApproach" src="https://github.com/user-attachments/assets/fe7b1195-9498-4c7b-8d6a-110308cb723b" />
