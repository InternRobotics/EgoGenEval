# Evaluation Dependencies

Evaluator code and checkpoints are obtained from their respective providers and
remain subject to their own licenses. Model weights are not included here.

## Evaluator checkpoints

Configure these four components in `configs/evaluator.local.yaml`:

| Component | Purpose | Paper-pinned identity |
| --- | --- | --- |
| Depth Anything 3 | CMG pose and generated-object depth | source commit `41736238f5bced4debf3f2a12375d2466874866d`; `DA3NESTED-GIANT-LARGE` weight SHA-256 `8899faf998dedbc230261ab736fa57015280727399429122d44d4f9e7aac2ddd` |
| Grounding DINO | prompt-conditioned boxes | `IDEA-Research/grounding-dino-base`, box/text threshold 0.25/0.25 |
| DINOv3 ViT-L/16 | crop identity and integrity | `dinov3-vitl16-pretrain-lvd1689m` |
| Qwen3-VL | correspondence and match/quality checking | `Qwen3-VL-30B-A3B-Instruct` |

Depth Anything 3 needs both its source checkout and checkpoint. Its additional
runtime dependencies are:

```bash
pip install omegaconf pycolmap evo 'moviepy<2'
```

Check the configured environment before scoring:

```bash
egogeneval doctor --full --evaluator-config configs/evaluator.local.yaml
```

## Benchmark inputs

The repository includes the benchmark manifest and frozen GT object labels.
Use the [local preparation guide](docs/local_data.md) to reconstruct RGB-D frames
and camera calibration from official source downloads.

The benchmark uses [Hypersim](https://github.com/apple/ml-hypersim),
[ScanNet](https://github.com/ScanNet/ScanNet),
[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), and
[Matterport3D](https://niessner.github.io/Matterport/). Their original licenses,
attribution requirements, and access terms continue to apply. The MIT license
on this code does not grant rights to redistribute source data or model weights.

## Local data preparation references

The local builder reads the annotation format documented by
[EmbodiedScan](https://github.com/InternRobotics/EmbodiedScan/tree/fe26e4bc3f3fb706fd7e33788766f61f8857fc3c)
and follows its ScanNet image extraction procedure. Its streaming `.sens` reader
uses the format documented by [ScanNet's SensorData.py](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py).
The raw Matterport3D camera reader follows the
[official pose and intrinsic formats](https://github.com/niessner/Matterport/blob/8cd7c81aff5824d578caa8cf5b79e2896bfe7f34/data_organization.md).
The Hypersim download helper reads selected files from the public scene ZIPs
listed by the [official downloader](https://github.com/apple-aiml-research/ml-hypersim/blob/main/code/python/tools/dataset_download_images.py).
Upstream source implementations were consulted rather than vendored into the
builder. EmbodiedScan annotations and original RGB-D files are not bundled with
these scripts; obtain them under their respective access terms.

Exact ScanNet JPEG reproduction uses [IJG libjpeg 9e](https://www.ijg.org/).
`scripts/install_scannet_jpeg.py` downloads and verifies the official source
archive, then builds isolated `djpeg` and `cjpeg` tools. IJG's source and tools
retain their own license; they are not included in this repository.
