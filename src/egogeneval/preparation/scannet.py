"""Selective ScanNet .sens v4 extraction in EmbodiedScan's image format.

Format references (not vendored implementations): ScanNet's SensorData.py and
EmbodiedScan's generate_image_scannet.py; see docs/local_data.md. JPEGs are
decoded and re-encoded, as in EmbodiedScan, rather than copying the compressed
sensor payload. Fixed IJG 9e tools reproduce the frozen JPEG bytes; imageio is
retained for other exports. Only requested frames are decoded.
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from ..errors import ValidationError


def extract_frames(sensor: Path, frames: set[int], output: Path, *, jpeg_tools: Path | None = None) -> None:
    """Write selected five-digit JPG/PNG pairs to a caller-owned staging folder."""
    if not frames or min(frames) < 0:
        raise ValidationError(f"{sensor}: expected nonnegative source frame numbers")
    size = sensor.stat().st_size
    with sensor.open("rb") as stream:

        def read(count: int) -> bytes:
            if count < 0 or count > size - stream.tell():
                raise ValidationError(f"{sensor}: truncated or invalid .sens payload")
            payload = stream.read(count)
            if len(payload) != count:
                raise ValidationError(f"{sensor}: truncated .sens payload")
            return payload

        def unpack(fmt: str):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))

        if unpack("I")[0] != 4:
            raise ValidationError(
                f"{sensor}: only ScanNet .sens version 4 is supported"
            )
        read(unpack("Q")[0])  # sensor name
        read(4 * 16 * 4)  # color/depth intrinsics and extrinsics
        color_codec, depth_codec = unpack("ii")
        color_w, color_h, depth_w, depth_h = unpack("IIII")
        depth_shift = unpack("f")[0]
        count = unpack("Q")[0]
        if (color_codec, depth_codec) != (2, 1):
            raise ValidationError(
                f"{sensor}: expected JPEG color and zlib_ushort depth"
            )
        if depth_shift != 1000.0 or min(color_w, color_h, depth_w, depth_h) <= 0:
            raise ValidationError(
                f"{sensor}: unexpected depth scale or image dimensions"
            )
        if max(frames) >= count:
            raise ValidationError(
                f"{sensor}: requested frame {max(frames)} but only {count} exist"
            )
        output.mkdir(parents=True, exist_ok=True)
        for frame in range(max(frames) + 1):
            read(16 * 4 + 2 * 8)  # c2w and color/depth timestamps
            color_size, depth_size = unpack("QQ")
            if color_size + depth_size > size - stream.tell():
                raise ValidationError(f"{sensor}: truncated frame {frame}")
            if frame not in frames:
                stream.seek(color_size + depth_size, 1)
                continue
            color_payload, depth_payload = read(color_size), read(depth_size)
            try:
                color = None if jpeg_tools else imageio.imread(io.BytesIO(color_payload), format="JPEG")
                # A bounded inflater rejects corrupt dimensions or oversized payloads.
                expected = depth_w * depth_h * 2
                inflater = zlib.decompressobj()
                depth_bytes = inflater.decompress(depth_payload, expected + 1)
                if len(depth_bytes) != expected or not inflater.eof:
                    raise ValueError("invalid decompressed depth size")
                depth = np.frombuffer(depth_bytes, dtype="<u2").reshape(
                    depth_h, depth_w
                )
                if color is not None and color.shape[:2] != (color_h, color_w):
                    raise ValueError("color dimensions do not match header")
                if jpeg_tools:
                    from .jpeg import reencode_scannet

                    encoded = reencode_scannet(color_payload, jpeg_tools, (color_w, color_h))
                    (output / f"{frame:05d}.jpg").write_bytes(encoded)
                else:
                    imageio.imwrite(output / f"{frame:05d}.jpg", color)
                imageio.imwrite(output / f"{frame:05d}.png", depth)
            except (ValueError, OSError, zlib.error) as exc:
                raise ValidationError(
                    f"{sensor}: cannot decode frame {frame}: {exc}"
                ) from exc
