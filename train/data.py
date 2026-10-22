"""Unpaired open-domain training data: image-caption URLs (FLUX) and captioned videos (Wan)."""
import hashlib
import os
import random
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

import imageio
import requests
import torch
import torchvision
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image

from diffsynth.trainers.unified_dataset import ImageCropAndResize


# ---------------------------------------------------------------------------
# Images: HF dataset of (url, caption), e.g. bghira/photo-concept-bucket
# ---------------------------------------------------------------------------
class URLImageDataset(torch.utils.data.Dataset):
    """Downloads images lazily from a URL column and caches them on disk.

    Images are resized to at most `max_pixels` (dimensions divisible by 16). Samples whose URL
    fails to load are replaced by a random other sample.
    """

    def __init__(self, dataset_name="bghira/photo-concept-bucket", url_field="url",
                 text_field="cogvlm_caption", cache_dir="./data/images", max_pixels=1920 * 1080,
                 repeat=1, max_samples=None, timeout=10):
        self.dataset = load_dataset(dataset_name, split="train")
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.url_field, self.text_field = url_field, text_field
        self.max_pixels = max_pixels
        self.repeat = repeat
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url):
        ext = url.split("?")[0].rsplit(".", 1)[-1].lower() if "." in url.split("?")[0] else "jpg"
        if ext not in ("jpg", "jpeg", "png", "webp", "gif", "bmp"):
            ext = "jpg"
        return self.cache_dir / f"{hashlib.md5(url.encode()).hexdigest()}.{ext}"

    def _download(self, url):
        path = self._cache_path(url)
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                path.unlink(missing_ok=True)
        try:
            response = requests.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load {url}: {e}")
            return None
        try:
            image.save(path, quality=95)
        except Exception as e:
            warnings.warn(f"Failed to cache {url}: {e}")
        return image

    def _resize(self, image):
        width, height = image.size
        if width * height > self.max_pixels:
            scale = (width * height / self.max_pixels) ** 0.5
            height, width = int(height / scale), int(width / scale)
        height, width = height // 16 * 16, width // 16 * 16
        scale = max(width / image.size[0], height / image.size[1])
        image = torchvision.transforms.functional.resize(
            image, (round(image.size[1] * scale), round(image.size[0] * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return torchvision.transforms.functional.center_crop(image, (height, width))

    def __getitem__(self, index):
        index = index % len(self.dataset)
        for _ in range(len(self.dataset)):
            item = self.dataset[index]
            url = item.get(self.url_field)
            image = self._download(url) if url else None
            if image is not None:
                return {"image": self._resize(image), "prompt": item.get(self.text_field) or ""}
            index = random.randint(0, len(self.dataset) - 1)
        raise RuntimeError("No loadable image found in the dataset.")

    def __len__(self):
        return len(self.dataset) * self.repeat


# ---------------------------------------------------------------------------
# Videos: HF WebDataset repo with videos/*.tar and annotations/captions.parquet,
# e.g. zengxianyu/open-sora-pexels-subset
# ---------------------------------------------------------------------------
def _decode_video(video_bytes, num_frames, frame_processor):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()
        reader = imageio.get_reader(tmp.name)
        n = min(num_frames, reader.count_frames())
        while n > 1 and n % 4 != 1:  # Wan requires 4k+1 frames
            n -= 1
        frames = [frame_processor(Image.fromarray(reader.get_data(i))) for i in range(n)]
        reader.close()
    finally:
        os.unlink(tmp.name)
    return frames


class _VideoDatasetMixin:
    def _process(self, sample):
        video_bytes = sample["mp4"]
        if isinstance(video_bytes, dict):
            video_bytes = video_bytes["bytes"]
        return {
            "video": _decode_video(video_bytes, self.num_frames, self.frame_processor),
            "prompt": self.captions.get(sample.get("__key__", ""), ""),
        }


class StreamingVideoDataset(_VideoDatasetMixin, torch.utils.data.IterableDataset):
    def __init__(self, video_ds, captions, num_frames, frame_processor):
        self.video_ds, self.captions = video_ds, captions
        self.num_frames, self.frame_processor = num_frames, frame_processor

    def __iter__(self):
        for sample in self.video_ds:
            yield self._process(sample)


class MapVideoDataset(_VideoDatasetMixin, torch.utils.data.Dataset):
    def __init__(self, video_ds, captions, num_frames, frame_processor):
        self.video_ds, self.captions = video_ds, captions
        self.num_frames, self.frame_processor = num_frames, frame_processor

    def __getitem__(self, index):
        return self._process(self.video_ds[index])

    def __len__(self):
        return len(self.video_ds)


def create_video_dataset(repo_id="zengxianyu/open-sora-pexels-subset", num_frames=49, height=None,
                         width=None, max_pixels=1280 * 720, streaming=True, max_samples=None):
    captions_ds = load_dataset(repo_id, data_files="annotations/captions.parquet", split="train")
    captions = {}
    for row in captions_ds.sort("source"):  # prefer the "513f" captions, which sort first
        captions.setdefault(row["sample_id"], row["caption"])

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    tar_urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}"
                for f in sorted(files) if f.startswith("videos/") and f.endswith(".tar")]
    video_ds = load_dataset("webdataset", data_files={"train": tar_urls}, split="train", streaming=streaming)
    if hasattr(video_ds, "decode"):  # keep raw mp4 bytes; newer `datasets` would require torchcodec
        video_ds = video_ds.decode(False)
    if max_samples is not None:
        if streaming:
            video_ds = video_ds.take(max_samples)
        else:
            video_ds = video_ds.select(range(min(max_samples, len(video_ds))))

    frame_processor = ImageCropAndResize(height, width, max_pixels, 16, 16)
    cls = StreamingVideoDataset if streaming else MapVideoDataset
    return cls(video_ds, captions, num_frames, frame_processor)
