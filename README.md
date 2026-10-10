# vitrine

My photos land in a dated folder tree via [shelf](https://github.com/franzos/shelf), which moves and deduplicates them but doesn't help me look at them. vitrine is a local viewer over that tree: folders on the left, thumbnails in the middle, a single image on Enter. Face recognition is an optional filter on top, for "photos with these two people" or "photos with exactly one face".

vitrine never touches image files. It keeps a catalog database and a thumbnail cache, nothing else.

- Images: JPEG, PNG, WebP, GIF (first frame), TIFF, HEIC, AVIF, JXL, and RAW (DNG, Nikon, Sony, Panasonic, Olympus, Fujifilm) via the embedded preview.
- Videos show up as placeholders; no thumbnails, no face scanning.
- Images larger than the GPU's max texture size are drawn in tiles.

## Usage

```bash
vitrine                  # open the configured roots
vitrine ~/Pictures       # open one folder, configured or not
vitrine index            # index the configured roots, no GUI
vitrine index ~/Pictures # index one folder
vitrine status           # paths, catalog counts, last run's counters
vitrine faces fetch      # download the face models (see the license note below)
vitrine faces recluster  # regroup faces from scratch, keeping manual edits
```

The GUI indexes in the background on start and on "Rescan". Only one indexer runs at a time; if `vitrine index` is busy in a terminal, the GUI shows "indexing elsewhere" and waits.

| Key | Action |
| --- | --- |
| Enter, double-click | open the image |
| Left, Right | step through the current (filtered) set |
| Escape | back to the grid |
| `0` / `1` | fit / 100% |
| `O` | open the set (up to 500 images either side) in the external viewer |

The viewer can also reveal the image in the file manager (freedesktop `FileManager1` over D-Bus, else `xdg-open` on the folder).

A rerun over an unchanged tree is a stat walk. Moved files keep their thumbnail and faces, since both are keyed by content hash. Files that disappear are marked missing, not deleted, so an unmounted disk doesn't wipe the names you gave people.

## Config

`~/.config/vitrine/config.toml`. Every key is optional; these are the defaults:

```toml
roots = []                  # absolute paths, ~ is expanded

[faces]
enabled = false
device = "cpu"              # or "cuda", with a CUDA build of ONNX Runtime
cluster_threshold = 0.62    # max cosine distance for two faces to be neighbours
min_neighbours = 2          # neighbours a face needs to start a new person
merge_threshold = 0.25      # max cosine distance between two people's mean embeddings to merge them
min_score = 0.5             # min detection score
min_face_size = 40          # min shorter box side, in source pixels

[viewer]
external = ["oculante"]     # image paths are appended
```

The catalog and models live in `$XDG_DATA_HOME/vitrine/`, thumbnails in `$XDG_CACHE_HOME/vitrine/thumbs.db`. The thumbnail cache is safe to delete; it's rebuilt on demand.

## Faces

Off by default. To turn it on:

1. `vitrine faces fetch`
2. Set `faces.enabled = true`.
3. `vitrine index` (or open the GUI).

Detection is SCRFD (`det_10g.onnx`), embeddings are ArcFace R50 (`w600k_r50.onnx`), both from InsightFace's buffalo_l set, run through [ONNX Runtime](https://onnxruntime.ai/) via [ort](https://github.com/pykeio/ort). ONNX Runtime 1.28 or newer is loaded at runtime from `ORT_DYLIB_PATH`.

**License note:** the buffalo_l weights aren't mine and aren't licensed for commercial use. From the [InsightFace README](https://github.com/deepinsight/insightface#license): "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." So they're never bundled: `faces fetch` prints that note, downloads both files from a pinned Hugging Face revision and checks them against sha256 hashes of the official `buffalo_l.zip` files.

Faces are grouped into people automatically, much like [Immich](https://immich.app/) does it. One person with several looks (hair, beard, glasses) often comes out as several people, so each pass ends by merging people whose mean embeddings are within `merge_threshold`; two named people are never merged. In the people panel you can name, merge and ignore people, remove a face from a person, and mark a detection as "not a face". Manual edits always win; reclustering and later index runs never undo them.

The filter bar narrows the grid by face count (0, 1, 2, 3+), by people (images with everyone selected), or to unknown faces. The viewer steps through whatever the filter left.

Clustering isn't perfect. Small or blurry faces stay unassigned, and the same person across twenty years of photos sometimes ends up as two people you'll need to merge.

## shelf

To index new photos after each `shelf run`, add this to the shelf profile:

```toml
[post_run]
command = ["vitrine", "index"]
```

vitrine doesn't read shelf's database; it walks the configured roots again.

## Building

Needs Rust 1.95+, libheif 1.23+ and libjpeg-turbo 3 (via pkg-config), and at runtime ONNX Runtime 1.28+ for faces. On Guix, the manifest provides all of it:

```bash
guix shell -m manifest.scm -- cargo build --release
guix shell -m manifest.scm -- cargo test
```

Guix's Rust 1.95 ships without rustfmt, clippy and rustdoc, so doctests are off and `cargo fmt` / `cargo clippy` need a separate toolchain.

## License

GPL-3.0-only. The image loader and the tiled GL renderer are adapted from [oculante](https://github.com/woelper/oculante) (MIT); see `NOTICE`.
