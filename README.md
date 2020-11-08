# Detection-of-tumor-mutation-in-histopathological-images---colorectal-cancer

This project involves a novel way to train CNN's for the purposes of detecting mutation in colorectal cancer cells. The pipiline involves training 3 different architectures each with 3 models of different resolutions, specifically 128x128, 256x256 and 512x512. Each model feeds their weight to a larger model similar to transfer learning. So in our case, 128x128--> 256x256 --> 512x512. The reason to do this is to overcome invariance in CNN, as the process involves cutting the original tumour tissue of a patient into several tiles because of memory restrictions to train on a GPU.


Example data:





Data source : https://zenodo.org/record/3832231
Code: Notebook in this repo.
Credits and acknowledgements: Supervisor : Rachel Cavill, co-supervisor: Jakob nicholas
