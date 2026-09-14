# Prepare the benchmark locally

Obtain the official source datasets, then run **one preparation command** with
their paths. It creates a combined benchmark for both model generation and
scoring: all four datasets, 1,400 cases and 3,675 frame identities. Source files
remain unchanged, and preparation refuses to overwrite an existing output.
The providers' [data terms](../DATA_TERMS.md) still apply.

## Download the sources

From the repository root, install with Python 3.10–3.12 and create the download
plan. Full preparation requires Linux with working EGL/OpenGL:

```bash
pip install -e '.[prepare,scannetpp]'
egogeneval source-plan --output source-plan.json
```

The plan lists the required scene IDs, filenames and ScanNet++ download assets.
Obtain access from each provider and download/extract the original files:

| Source | Official access / download | Required files |
| --- | --- | --- |
| Hypersim | [Download instructions](https://github.com/apple-aiml-research/ml-hypersim#downloading-the-hypersim-dataset) | Preview RGB JPEGs, depth HDF5, camera and unit metadata |
| ScanNet v2 | [Access instructions](https://github.com/ScanNet/ScanNet#scannet-data) | `<scene>.sens` and `<scene>.txt` containing `axisAlignment` |
| Matterport3D | [Access instructions](https://github.com/niessner/Matterport#data) | `matterport_color_images` and `matterport_depth_images` archives |
| EmbodiedScan v1 | [Annotation request and instructions](https://github.com/InternRobotics/EmbodiedScan/blob/main/data/README.md) | Train, validation and test annotation PKLs; used only for Matterport3D cameras |
| ScanNet++ v2 | [Application and download](https://scannetpp.mlsg.cit.tum.de/scannetpp/) | Original 2 MP fisheye DSLR images, anonymization masks, Nerfstudio transforms and aligned scan meshes |

Use the EmbodiedScan v1 release with native Matterport UUID image paths. Preserve
original JPEGs; independently resized or re-encoded files will not match. For
ScanNet++, use **v2** with the official downloader's default version settings.
Set its `download_scenes` and `download_assets`
from `source-plan.json` (`dslr_resized_dir`, `dslr_resized_mask_dir`,
`dslr_nerfstudio_transform_path`, `scan_mesh_path`).

Example extracted layout:

```text
/datasets/
  scannet/scans/<scene>/
    <scene>.sens
    <scene>.txt
  matterport3d/scans/<house>/
    matterport_color_images/<uuid>_iX_Y.jpg
    matterport_depth_images/<uuid>_dX_Y.png
  embodiedscan/
    embodiedscan_infos_train.pkl
    embodiedscan_infos_val.pkl
    embodiedscan_infos_test.pkl
  hypersim/evermotion_dataset/scenes/<scene>/
    images/scene_cam_XX_final_preview/frame.IIII.color.jpg
    images/scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5
    _detail/cam_XX/camera_keyframe_positions.hdf5
    _detail/cam_XX/camera_keyframe_orientations.hdf5
    _detail/metadata_scene.csv
  scannetpp/data/<scene>/
    dslr/resized_images/DSCXXXXX.JPG
    dslr/resized_anon_masks/DSCXXXXX.png
    dslr/nerfstudio/transforms.json
    scans/mesh_aligned_0.05.ply
```

The optional `download-hypersim` and `unpack-sources` helpers can download
selected Hypersim files or extract selected Hypersim/Matterport archive members;
see their `--help`. They are not separate benchmark conversion stages.

## Prepare all datasets with one command

```bash
egogeneval prepare-benchmark \
  --hypersim /datasets/hypersim \
  --scannet /datasets/scannet \
  --matterport3d /datasets/matterport3d \
  --scannetpp /datasets/scannetpp \
  --embodiedscan /datasets/embodiedscan \
  --output prepared-data/egogeneval \
  --workers 4
```

The command checks all selected inputs first, reconstructs RGB/depth/cameras,
checks bundled reference fingerprints and writes one output directory. No
per-dataset conversion commands or manual output merging are needed. Add
`--check-only` to check inputs without producing assets or installing tools.

ScanNet's fixed IJG 9e JPEG tools are downloaded, checksum-verified and compiled
on first use; install a C compiler and `make`. Subsequent runs reuse the verified
tool cache. Use `--cache-dir` to move the cache, `--scannet-jpeg-tools` for an
existing IJG 9e installation, or `--jpeg-archive jpegsrc.v9e.tar.gz` for offline
compilation. `--workers` controls parallel ScanNet scene readers/extractors and
isolated ScanNet++ rendering processes (1–16); the selected ScanNet sensor
streams occupy about 108 GiB.

ScanNet++ uses Linux EGL with PyRender 0.1.45, PyOpenGL 3.1.0, Trimesh 5.1.0,
OpenCV 4.10.0 and Pillow 11.3.0. The reference renderer is Mesa 25.2.8 llvmpipe.
Other OpenGL implementations can change depth pixels; the command records the
actual renderer and rejects any asset fingerprint mismatch. A GPU is not
required for this reference rendering path.

The bundled camera conventions preserve the benchmark's historical pixel-center
choices for metadata and depth rendering separately. They are applied automatically
from official calibration; no processed ScanNet++ images or depth maps are needed.

The prepared assets use revision `scannetpp-depth-v2`, with ScanNet++ depth
rendered from the official aligned meshes. `build_report.json`, `run_config.json`
and `results.json` record the asset revision; scoring also records the asset-index
hash so results can be associated with the exact local ground truth.

| Output | Use |
| --- | --- |
| `generation_inputs.jsonl` | Initial input images and ordered instructions; excludes target images and numeric GT pose metadata |
| `manifest.jsonl` | Canonical cases passed to the scorer |
| `eval_frames.jsonl`, `images/`, `depth/` | Local RGB, depth and camera assets consumed directly by the evaluator |
| `build_report.json`, `reference_fingerprints.jsonl` | Build provenance and verification reference |

Verification runs during preparation. To check the completed bundle again:

```bash
egogeneval verify-data --data prepared-data/egogeneval
```

For a deliberate subset, repeat `--dataset`, for example `--dataset hypersim
--dataset scannet`; only those sources are required. Retain the generated
manifest and identify subset scores explicitly.

## Generate and evaluate

Read `generation_inputs.jsonl` and resolve its image paths relative to
`prepared-data/egogeneval`. Follow each case's instructions in step order. For
Chain and Cycle cases, feed each generated image into the next step and keep the
initial auxiliary views fixed. Save outputs as
`runs/mymodel/outputs/<sample_id>/step<N>.png`. See the
[model interfaces and output contract](../README.md#-evaluate-your-model) for
image, video and world models.

Install and configure the evaluator as described in the
[README](../README.md#1-install), then point it at the same prepared directory:

```bash
egogeneval score \
  --manifest prepared-data/egogeneval/manifest.jsonl \
  --generations runs/mymodel \
  --model-id my-model --model-type image \
  --eval-frames prepared-data/egogeneval \
  --evaluator-config configs/evaluator.local.yaml \
  --output runs/mymodel/evaluation
```

The scorer resolves target RGB, depth and camera parameters through
`eval_frames.jsonl`; no original dataset paths or additional conversion are
needed after preparation.
