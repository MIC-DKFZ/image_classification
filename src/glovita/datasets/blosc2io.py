from copy import deepcopy
from typing import Union, Tuple
import numpy as np
import blosc2
import math


class Blosc2IO:
    @staticmethod
    def load(filepath: str, num_threads: int = 1, mode: str = "r"):
        """
        Load a Blosc2-compressed file.
        Args:
            filepath (str): Path to the Blosc2 file to be loaded. The filepath needs to have the extension '.b2nd'.
            num_threads (int): Number of threads to use for loading the file.
            mode (str): The Blosc2 open mode.
        Returns:
            Tuple[blosc2.ndarray, dict]: A tuple containing the loaded data and its metadata.
        """
        blosc2.set_nthreads(num_threads)
        dparams = {"nthreads": num_threads}
        data = blosc2.open(urlpath=filepath, dparams=dparams, mode=mode, mmap_mode=mode)
        metadata = dict(data.schunk.meta)
        del metadata["b2nd"]
        return data, metadata

    @staticmethod
    def save(
        data: np.ndarray,
        filepath: str,
        chunks: Tuple = None,
        blocks: Tuple = None,
        metadata: dict = None,
        clevel: int = 8,
        codec: blosc2.Codec = blosc2.Codec.ZSTD,
        num_threads: int = 1,
    ):
        """
        Save a numpy array to a Blosc2-compressed file.
        Args:
            data (np.ndarray): The array to be saved.
            filepath (str): Path to save the file, must end with ".b2nd".
            chunks (Tuple): Size of chunks for the saved file. Defaults to None.
            blocks (Tuple): Size of blocks for the saved file. Defaults to None.
            metadata (dict): Metadata to be saved with the file. Defaults to None.
            clevel (int): Compression level, ranging from 0 (no compression) to 9 (maximum compression). Defaults to 8.
            codec (blosc2.Codec): Compression codec to use. Defaults to blosc2.Codec.ZSTD.
            num_threads (int): Number of threads to use for saving the file.
        Raises:
            RuntimeError: If the file extension is not ".b2nd".
        Returns:
            None
        """
        blosc2.set_nthreads(num_threads)
        if not filepath.endswith(".b2nd"):
            raise RuntimeError("Blosc2 requires '.b2nd' as extension.")

        cparams = {
            "codec": codec,
            "clevel": clevel,
        }
        blosc2.asarray(
            np.ascontiguousarray(data),
            urlpath=filepath,
            chunks=chunks,
            blocks=blocks,
            cparams=cparams,
            mmap_mode="w+",
            meta=metadata,
        )

    @staticmethod
    def comp_blosc2_params(
        image_size: Tuple[int, int, int, int],
        patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
        bytes_per_pixel: int = 4,  # 4 byte are float32
        l1_cache_size_per_core_in_bytes: int = 32768,  # 1 Kibibyte (KiB) = 2^10 Byte;  32 KiB = 32768 Byte
        l3_cache_size_per_core_in_bytes: int = 1441792,  # 1 Mibibyte (MiB) = 2^20 Byte = 1.048.576 Byte; 1.375MiB = 1441792 Byte
        safety_factor: float = 0.8,  # we dont will the caches to the brim. 0.8 means we target 80% of the caches
    ):
        """
        Computes a recommended block and chunk size for saving arrays with Blosc v2.
        Blosc2 NDIM documentation:
        "Having a second partition allows for greater flexibility in fitting different partitions to different CPU cache levels.
        Typically, the first partition (also known as chunks) should be sized to fit within the L3 cache,
        while the second partition (also known as blocks) should be sized to fit within the L2 or L1 caches,
        depending on whether the priority is compression ratio or speed."
        (Source: https://www.blosc.org/posts/blosc2-ndim-intro/)
        Our approach is not fully optimized for this yet.
        Currently, we aim to fit the uncompressed block within the L1 cache, accepting that it might occasionally spill over into L2, which we consider acceptable.
        Note: This configuration is specifically optimized for nnU-Net data loading, where each read operation is performed by a single core, so multi-threading is not an option.
        The default cache values are based on an older Intel 4110 CPU with 32KB L1, 128KB L2, and 1408KB L3 cache per core.
        We haven't further optimized for modern CPUs with larger caches, as our data must still be compatible with the older systems.
        Args:
            image_size (Tuple[int, int, int, int]): The size of the image.
            patch_size (Union[Tuple[int, int], Tuple[int, int, int]]): Patch size, containing the spatial dimensions (x, y) or (x, y, z).
            bytes_per_pixel (int, optional): Number of bytes per element. Defaults to 4 (for float32).
            l1_cache_size_per_core_in_bytes (int, optional): Size of the L1 cache per core in bytes. Defaults to 32768.
            l3_cache_size_per_core_in_bytes (int, optional): Size of the L3 cache exclusively accessible by each core in bytes. Defaults to 1441792.
            safety_factor (float, optional): Safety factor to avoid filling caches completely. Defaults to 0.8.
        Returns:
            Tuple[Tuple[int, ...], Tuple[int, ...]]: Recommended block size and chunk size.
        """

        num_squeezes = 0

        if len(image_size) == 2:
            image_size = (1, 1, *image_size)
            num_squeezes = 2

        if len(image_size) == 3:
            image_size = (1, *image_size)
            num_squeezes = 1

        if len(image_size) != 4:
            raise RuntimeError("Image size must be 4D.")

        if not (len(patch_size) == 2 or len(patch_size) == 3):
            raise RuntimeError("Patch size must be 2D or 3D.")

        num_channels = image_size[0]
        if len(patch_size) == 2:
            patch_size = [1, *patch_size]
        patch_size = np.array(patch_size)
        block_size = np.array(
            (
                num_channels,
                *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size],
            )
        )

        # shrink the block size until it fits in L1
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (
            l1_cache_size_per_core_in_bytes * safety_factor
        ):
            # pick largest deviation from patch_size that is not 1
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            # now reduce that axis to the next lowest power of 2
            block_size[picked_axis + 1] = 2 ** (
                max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1)))
            )
            block_size[picked_axis + 1] = min(
                block_size[picked_axis + 1], image_size[picked_axis + 1]
            )
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

        block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

        # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
        # load data into L3
        # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (
            l3_cache_size_per_core_in_bytes * safety_factor
        ):
            if patch_size[0] == 1 and all(
                [i == j for i, j in zip(chunk_size[2:], image_size[2:])]
            ):
                break
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
            # find axis that deviates from block_size the most
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while (
                chunk_size[picked_axis + 1] == image_size[picked_axis + 1]
                or patch_size[picked_axis] == 1
            ):
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(
                chunk_size[picked_axis + 1], image_size[picked_axis + 1]
            )
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
                # chunk size should not exceed patch size * 1.5 on average
                chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
                break
        # better safe than sorry
        chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]

        block_size = block_size[num_squeezes:]
        chunk_size = chunk_size[num_squeezes:]

        return tuple(block_size), tuple(chunk_size)


if __name__ == "__main__":
    image_array = np.random.random((128, 512, 512))
    filepath = "tmp.b2nd"

    image_shape = image_array.shape
    patch_size = (64, 128, 128)
    print("image_shape: ", image_shape)
    print("patch_size: ", patch_size)
    block_size, chunk_size = Blosc2IO.comp_blosc2_params(image_shape, patch_size)
    print("block_size: ", block_size)
    print("chunk_size: ", chunk_size)

    Blosc2IO.save(image_array, filepath, chunk_size, block_size, metadata={"tmp": 5})

    image, metadata = Blosc2IO.load(filepath)
    image_array = image[...]
    print("metadata: ", metadata)
