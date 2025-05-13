import torch
import math


def mil_forward_features(encoder, patches, batch_size):
    """receive all patches as one tensor and iter over it with encoder, batch size>1 allowed"""

    # iter over data batch size
    total_features = []
    for img in patches:
        # iter over the image with a given MIL batch_size and encode the patches individually
        features_per_img = []

        for i in range(0, len(img), batch_size):

            if i + batch_size > len(img):

                if len(img) > 1:
                    batch = img[i:]
                else:
                    batch = img  # [i]

            else:
                batch = img[i : i + batch_size]

            features_per_img.append(encoder(batch))

        # save encoded features
        features_per_img = torch.concat(features_per_img, dim=0).unsqueeze(0)
        total_features.append(features_per_img)

    total_features = torch.concat(total_features, dim=0)

    return total_features


def mil_forward_features_with_bs1(encoder, patches, batch_size):
    """receive all patches as one tensor and iter over it with encoder"""

    # iter over the image with a given batch_size and encode the patches individually
    features = []

    for i in range(0, len(patches), batch_size):

        if i + batch_size > len(patches):

            if len(patches) > 1:
                batch = patches[i:]
            else:
                batch = patches  # [i]

        else:
            batch = patches[i : i + batch_size]

        features.append(encoder(batch))

    # unsqueeze to add batch dim
    # MIL expects batch_size of 1 since individual patches need to be encoded with mil_batch_size already
    # if images are small enough that a higher batch_size would be possible you can simply drop mil completely
    # and encode the entire image at once
    features = torch.concat(features, dim=0).unsqueeze(0)

    return features


def mil_forward_features_with_patchextractor(encoder, patch_extractor, img):
    """receive entire memory mapped img and do sliding window here, or draw certain amount of patches randomly"""
    # set image so that patch extractor can calculate patches
    patch_extractor.set_array(img)

    # iter over the image with a given batch_size and encode the patches individually
    features = []
    for batch in patch_extractor:
        features.append(encoder(batch))

    # unsqueeze to add batch dim
    # MIL expects batch_size of 1 since individual patches need to be encoded with mil_batch_size already
    # if images are small enough that a higher batch_size would be possible you can simply drop mil completely
    # and encode the entire image at once
    features = torch.concat(features, dim=0).unsqueeze(0)

    return features


class PatchExtractor:
    def __init__(
        self,
        patch_size: tuple,
        step_size: tuple,
        padding_value=0,
        random=False,
        num_random_patches=100,
        batch_size=1,
    ):
        """
        Class that extracts patches from a 4D (batch, channel, h, w) or 5D (batch, channel, d, h, w) tensor using either a sliding window approach or random sampling with on-the-fly padding.

        Parameters:
        patch_size (tuple): Size of the patches (d, h, w) for 3D or (h, w) for 2D
        step_size (tuple): Step size for sliding window (sd, sh, sw) for 3D or (sh, sw) for 2D. Can be a float representing a fraction of the patch size.
        padding_value (int, optional): Value to use for padding. Defaults to 0.
        random (bool, optional): If True, yields random patches instead of using a sliding window. Defaults to False.
        num_random_patches (int or float, optional): If an integer, it specifies the number of random patches to yield. If a float in (0,1], it represents the ratio of maximum possible patches to sample randomly. Defaults to 100.
        batch_size (int, optional): Number of patches to yield at once. Defaults to 1.
        """
        self.array = None
        self.patch_size = patch_size
        self.step_size = self._compute_step_size(step_size)
        self.padding_value = padding_value
        self.random = random
        self.num_random_patches = num_random_patches
        self.batch_size = batch_size
        self.max_patches = 0

    def _compute_step_size(self, step_size):
        """Computes the actual step size for the sliding window, treating all values as floats."""
        return tuple(max(1, round(p * s)) for s, p in zip(step_size, self.patch_size))

    def set_array(self, array: torch.Tensor):
        """Sets the input array and computes patch count."""
        if len(array.shape) not in [4, 5]:
            raise ValueError(
                "Input array must be 4D (batch, channel, h, w) or 5D (batch, channel, d, h, w)"
            )

        self.array = array
        self._compute_patch_count()

    def _compute_patch_count(self):
        """Computes the number of patches based on the extraction method."""
        if self.array is None:
            raise ValueError("Array must be set before computing patches")

        shape = self.array.shape[-3:]  # Support 2D (h, w) or 3D (d, h, w)
        if len(self.array.shape) == 4:
            d, h, w = (1, *shape)
        else:
            d, h, w = shape

        pd, ph, pw = self.patch_size
        sd, sh, sw = self.step_size

        self.max_patches = (
            (math.ceil((d - pd) / sd) + 1 if d > pd else 1)
            * (math.ceil((h - ph) / sh) + 1 if h > ph else 1)
            * (math.ceil((w - pw) / sw) + 1 if w > pw else 1)
        )

        # Ensure at least one patch is considered
        self.max_patches = max(1, self.max_patches)

        # Adjust random patches if a fraction is provided
        if (
            isinstance(self.num_random_patches, float)
            and 0 < self.num_random_patches <= 1
        ):
            self.num_random_patches = int(self.num_random_patches * self.max_patches)

    def __len__(self):
        if self.array is None:
            raise ValueError("Array must be set before querying length")
        return self.num_random_patches if self.random else self.max_patches

    def __iter__(self):
        if self.array is None:
            raise ValueError("Array must be set before iterating")
        return self._generate_patches()

    def _generate_patches(self):
        """Generator that yields batches of patches."""
        if self.array is None:
            raise ValueError("Array must be set before generating patches")

        patches = []
        batch_dim, channel_dim = self.array.shape[:2]
        shape = self.array.shape[-3:]  # Support 2D or 3D (h, w) or (d, h, w)
        if len(self.array.shape) == 4:
            d, h, w = (1, *shape)
        else:
            d, h, w = shape

        pd, ph, pw = self.patch_size
        sd, sh, sw = self.step_size
        device = self.array.device

        if self.random:
            for _ in range(self.num_random_patches):
                b = torch.randint(0, batch_dim, (1,)).item()
                i = torch.randint(0, max(1, d - pd + 1), (1,)).item()
                j = torch.randint(0, max(1, h - ph + 1), (1,)).item()
                k = torch.randint(0, max(1, w - pw + 1), (1,)).item()

                patch = torch.full(
                    (channel_dim, pd, ph, pw),
                    self.padding_value,
                    dtype=self.array.dtype,
                    device=device,
                )
                d_end = min(i + pd, d)
                h_end = min(j + ph, h)
                w_end = min(k + pw, w)

                patch[:, : d_end - i, : h_end - j, : w_end - k] = (
                    self.array[b, :, i:d_end, j:h_end, k:w_end]
                    if len(self.array.shape) == 5
                    else self.array[b, :, j:h_end, k:w_end]
                )
                patches.append(patch)

                if len(patches) == self.batch_size:
                    yield torch.stack(patches)
                    patches = []
        else:
            for i in range(0, d, sd):
                for j in range(0, h, sh):
                    for k in range(0, w, sw):
                        d_end = i + pd
                        h_end = j + ph
                        w_end = k + pw

                        # Slice region within bounds
                        d_start = i
                        h_start = j
                        w_start = k

                        d_end = min(d_end, d)
                        h_end = min(h_end, h)
                        w_end = min(w_end, w)

                        patch = torch.full(
                            (channel_dim, pd, ph, pw),
                            self.padding_value,
                            dtype=self.array.dtype,
                            device=device,
                        )

                        # Define slices
                        patch_d_slice = slice(0, d_end - d_start)
                        patch_h_slice = slice(0, h_end - h_start)
                        patch_w_slice = slice(0, w_end - w_start)

                        if len(self.array.shape) == 5:
                            patch[:, patch_d_slice, patch_h_slice, patch_w_slice] = (
                                self.array[
                                    :, :, d_start:d_end, h_start:h_end, w_start:w_end
                                ]
                            )
                        else:
                            patch[:, 0, patch_h_slice, patch_w_slice] = self.array[
                                :, :, h_start:h_end, w_start:w_end
                            ]

                        patches.append(patch)

                        if len(patches) == self.batch_size:
                            yield torch.stack(patches)
                            patches = []

        if patches:
            yield torch.stack(patches)
