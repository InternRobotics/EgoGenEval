# Data Terms

Original source code is licensed under [MIT](LICENSE), unless otherwise noted.
This license does not cover third-party datasets or model weights.

The benchmark uses Hypersim, Matterport3D, ScanNet, and ScanNet++. Their original
licenses and access terms apply; see [THIRD_PARTY.md](THIRD_PARTY.md).

The manifest and frozen GT object labels are included in this repository.
The reconstruction references contain only file and camera hashes,
dimensions and canonical frame identities; they contain no images, depth maps or
camera matrices. `prepare-benchmark` reconstructs these assets locally
from provider downloads and verifies them against that reference. The recommended
Matterport3D workflow obtains camera parameters from independently acquired
official EmbodiedScan v1 annotations; those PKLs are not bundled here and retain
their source access terms.
Source images, depth maps, and camera calibration remain subject to upstream terms.
`prepare-benchmark` creates local evaluation assets from files already obtained by
the user; see [local preparation](docs/local_data.md). Generated assets are local
research data, and running
these scripts does not grant permission to redistribute them.
