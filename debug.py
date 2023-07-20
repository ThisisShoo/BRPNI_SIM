"""Testing ground for new features, or to run a specific function and see what's wrong with it"""
from ray_tracing_fns import make_ray
import numpy as np

b_field = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

print(np.shape(b_field))
for i, count in enumerate(b_field):
    for j, count1 in enumerate(count):
        print(j, count1)
