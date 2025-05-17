from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("yeray142/first-impressions-v2")
print(ds)

reader = ds["train"][0]["video"]
frame = next(reader)
print(frame['data'])

image = np.array(frame['data']).transpose(1, 2, 0)
plt.imshow(image)
plt.show()
