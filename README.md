## Spatial Distance Histogram
This project creates a histogram based on a normally distributed set of three-dimensional points.
It contains both a provided CPU-based implementation and a custom GPU implementation.

The GPU implementation started as a naive approach where every thread computed the distance from its own index to the top of the input array.

This suffered due to poor memory bandwidth utilization and atomic operation conflicts.

The current implementation now utilizes tiling of array sections, output privatization in shared memory, and improved memory coalescing.

The current, optimized implementation achieves over a 3x speedup compared to the naive GPU implementation. The results are shown below:


<img width="837" height="225" alt="Screenshot_20251023_234417" src="https://github.com/user-attachments/assets/913e11e5-1cf7-434d-8275-3824e82efd42" />


<img width="758" height="272" alt="Screenshot_20251023_235047" src="https://github.com/user-attachments/assets/79651371-62f2-4b35-8964-b4f1d95364c0" />
