"""Frame extraction for video and world-model outputs.

Video / world models return either a rendered clip per step or a directory of
frames. EgoGenEval scores the *target view* of each step, so exactly one frame
must be chosen per step. The paper's protocol ("boundary_fixed_step") takes the
last frame of each step's segment as the realised target view; a clip of
``frames_per_step`` frames therefore contributes its final frame.

This module keeps that policy explicit and dependency-light: still frames need
only Pillow; decoding an actual video file additionally needs an available video
backend (imageio-ffmpeg or OpenCV), which is imported lazily so the still-image
path never pays for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..errors import ValidationError

# Extensions treated as already-extracted still frames.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".ppm", ".tif", ".tiff"}
# Extensions treated as encoded video that must be decoded to frames.
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


@dataclass(frozen=True)
class FrameSelection:
    """Which frame to pull out of a per-step segment.

    policy:
        ``"last"`` — the final rendered frame of the segment (paper default;
        the target view the model commits to after executing the action).
        ``"index"`` — an explicit 0-based frame index.
        ``"ratio"`` — a fractional position in ``[0, 1]`` along the segment.
    """

    policy: str = "last"
    index: int = -1
    ratio: float = 1.0

    def resolve_index(self, frame_count: int) -> int:
        if frame_count <= 0:
            raise ValidationError("cannot select a frame from an empty segment")
        if self.policy == "last":
            return frame_count - 1
        if self.policy == "index":
            idx = self.index if self.index >= 0 else frame_count + self.index
            if not 0 <= idx < frame_count:
                raise ValidationError(
                    f"frame index {self.index} out of range for {frame_count} frames"
                )
            return idx
        if self.policy == "ratio":
            if not 0.0 <= self.ratio <= 1.0:
                raise ValidationError(f"frame ratio must be in [0, 1]: {self.ratio}")
            return min(frame_count - 1, round(self.ratio * (frame_count - 1)))
        raise ValidationError(f"unknown frame selection policy: {self.policy}")


def _decode_video_frame(video_path: Path, selection: FrameSelection):
    """Return a PIL image for the selected frame of an encoded video."""

    # Prefer imageio (ffmpeg) then fall back to OpenCV; both are optional.
    # ValidationError is deliberately *not* caught: a bad --frame-index or an
    # empty segment is a user error, and retrying under OpenCV would only hide
    # it behind a second, less specific failure.
    backend_errors: list[str] = []
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as error:
        backend_errors.append(f"imageio-ffmpeg: {error}")
    else:
        try:
            frames = iio.imread(video_path, index=None)  # (T, H, W, C)
        except ValidationError:
            raise
        except Exception as error:  # noqa: BLE001 - decode failure; try OpenCV
            backend_errors.append(f"imageio-ffmpeg: {error}")
        else:
            idx = selection.resolve_index(len(frames))
            from PIL import Image

            return Image.fromarray(frames[idx])

    try:
        import cv2  # type: ignore
    except ImportError as error:
        backend_errors.append(f"opencv: {error}")
        detail = "; ".join(backend_errors)
        raise ValidationError(
            f"decoding {video_path.name} needs a video backend and none worked "
            f"({detail}). Install with: pip install 'egogeneval[video]'"
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = selection.resolve_index(frame_count)
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = capture.read()
        if not ok:
            detail = "; ".join(backend_errors) or "no prior backend"
            raise ValidationError(
                f"failed to read frame {idx} from {video_path.name} ({detail})"
            )
        from PIL import Image

        return Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
    finally:
        capture.release()


def _select_from_frame_dir(directory: Path, selection: FrameSelection) -> Path:
    frames = sorted(
        child
        for child in directory.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise ValidationError(f"no frames found in segment directory: {directory}")
    return frames[selection.resolve_index(len(frames))]


def extract_step_frame(
    source: Path,
    destination: Path,
    *,
    selection: FrameSelection,
    save: Callable | None = None,
) -> Path:
    """Materialise the chosen frame for one step at ``destination``.

    ``source`` may be:
      - a single still image (already-extracted frame) — copied through;
      - an encoded video file — decoded, the selected frame saved;
      - a directory of per-frame stills — the selected frame copied.

    Returns ``destination``.
    """

    source = Path(source)
    if not source.exists():
        raise ValidationError(f"generated frame source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        chosen = _select_from_frame_dir(source, selection)
        _copy_image(chosen, destination)
        return destination

    suffix = source.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        # Already a single frame; the paper materialised these upstream.
        _copy_image(source, destination)
        return destination

    if suffix in VIDEO_SUFFIXES:
        image = _decode_video_frame(source, selection)
        saver = save or (lambda img, path: img.convert("RGB").save(path))
        saver(image, destination)
        return destination

    raise ValidationError(f"unsupported generated source type: {source.name}")


def _copy_image(source: Path, destination: Path) -> None:
    import shutil

    if source.suffix.lower() == destination.suffix.lower():
        # Byte copy, no decode -- so an empty source would sail through here and
        # only surface much later as an "unreadable image" naming the *copy*.
        # Network and FUSE mounts produce exactly this: a symlink that reads
        # back as a 0-byte regular file.
        if source.stat().st_size == 0:
            raise ValidationError(
                f"generated frame source is empty (0 bytes): {source}. "
                "Some mounts materialise symlinks as empty files -- check that "
                "the generations are real image bytes, not links."
            )
        shutil.copy2(source, destination)
        return
    # Re-encode to the destination container so the extension stays truthful.
    from PIL import Image

    with Image.open(source) as image:
        image.convert("RGB").save(destination)
