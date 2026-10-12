# vitrine

My photos land in a dated folder tree via [shelf](https://github.com/franzos/shelf), which moves and deduplicates them but doesn't help me look at them. vitrine is a local viewer over that tree: folders on the left, thumbnails in the middle, a single image on Enter. On top of that it filters by place ("everything from Lisbon", "what haven't I placed yet") and, optionally, by face ("photos with these two people").

vitrine never touches image files. It keeps a catalog database, a thumbnail cache and a place-name database, nothing else. Apart from `geo fetch` and `faces fetch`, it doesn't use the network.

- Images: JPEG, PNG, WebP, GIF (first frame), TIFF, HEIC, AVIF, JXL, and RAW (DNG, Nikon, Sony, Panasonic, Olympus, Fujifilm) via the embedded preview.
- Videos get a thumbnail, a duration badge and their capture date; Enter in the viewer opens them in the configured player. This needs `ffmpeg` and `ffprobe` on `PATH`; without them, videos show as placeholders. Faces and locations don't apply to videos.

## Install

| Method | Command |
|--------|---------|
| Debian/Ubuntu | Download the [`.deb`](https://github.com/franzos/vitrine/releases) - `sudo apt install ./vitrine_*_amd64.deb` |
| Binary | Download [`vitrine-x86_64-unknown-linux-gnu.tar.gz`](https://github.com/franzos/vitrine/releases) and put `vitrine` on your `PATH` |
| Guix, Cargo | See [Building](#building) |

The release builds are Linux x86_64 only and need glibc 2.39 or newer (Ubuntu 24.04, Debian 13). The plain binary also needs libde265 and dav1d installed; the `.deb` pulls them in. For videos, install `ffmpeg`. For faces, you need ONNX Runtime, which isn't packaged (see [Faces](#faces)).

## Usage

```bash
vitrine                  # open the configured roots
vitrine ~/Pictures       # open one folder, configured or not
vitrine index            # index the configured roots, no GUI
vitrine index ~/Pictures # index one folder
vitrine status           # paths, catalog counts, last run's counters
vitrine geo fetch        # download the GeoNames place names (see Locations below)
vitrine faces fetch      # download the face models (see the license note below)
vitrine faces recluster  # regroup faces from scratch, keeping manual edits
```

The GUI indexes in the background on start and on "Rescan". Only one indexer runs at a time; if `vitrine index` is busy in a terminal, the GUI waits for it.

| Key | Action |
| --- | --- |
| Enter, double-click | open the image |
| Enter, in the viewer | play a video in the configured player |
| Ctrl-click, Shift-click | add one image to the selection, or a range |
| Ctrl-A | select every image in the current set |
| Left, Right | step through the current (filtered) set |
| Escape | back to the grid |
| `0` / `1` | fit / 100% |
| `O` | open the current set in the external viewer |

The viewer can also reveal the image in the file manager.

Rerunning the index over an unchanged tree is quick. Moved files keep their thumbnails and faces. Files that disappear are marked missing, not deleted, so an unmounted disk doesn't wipe the names you gave people.

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

[geo]
event_gap_hours = 6         # max gap between two capture times in one event

[viewer]
external = ["oculante"]     # image paths are appended
player = ["xdg-open"]       # the video path is appended
```

The catalog, the models and `geo.db` live in `$XDG_DATA_HOME/vitrine/`, thumbnails in `$XDG_CACHE_HOME/vitrine/thumbs.db`. The thumbnail cache is safe to delete; it's rebuilt on demand.

## Faces

Off by default. To turn it on:

1. Point `ORT_DYLIB_PATH` at [ONNX Runtime](https://onnxruntime.ai/) 1.28 or newer.
2. `vitrine faces fetch`
3. Set `faces.enabled = true`.
4. `vitrine index` (or open the GUI).

**License note:** the models are InsightFace's buffalo_l set. They aren't mine and aren't licensed for commercial use. From the [InsightFace README](https://github.com/deepinsight/insightface#license): "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." So they're never bundled: `faces fetch` prints that note, downloads them and checks their hashes.

Faces are grouped into people automatically, much like [Immich](https://immich.app/) does it. In the people panel you can name, merge and ignore people, remove a face from a person, and mark a detection as "not a face". Manual edits always win; reclustering and later index runs never undo them.

Clustering isn't perfect. Small or blurry faces stay unassigned, and one person with different looks (hair, beard, glasses, twenty years apart) often comes out as several people. vitrine merges people whose faces are close enough (`merge_threshold`), but never two named people; the rest you merge by hand.

With faces on, the filter bar narrows the grid by face count (0, 1, 2, 3+), by people (images with everyone selected), or to unknown faces. Face and place filters combine.

## Locations

Most of my photos have no GPS. A sample of 1,500 from my library had coordinates on 98, nearly all from old iPhones; the Fujifilm, Sony and Pixel shots carry none. So locations are mostly set by hand, at city or town level, and vitrine tries to keep that to a few clicks per trip.

`vitrine geo fetch` downloads the place names from GeoNames and builds `geo.db`. `--coverage` picks how much goes in:

| Coverage | Places | Download | `geo.db` |
|---|---|---|---|
| `cities` (default) | every place with 1,000+ people, about 165,000 | 10 MB | 60 MB |
| `islands` | `cities` plus inhabited islands with 1,000+ people | 430 MB | 60 MB |
| `full` | every populated place and inhabited island, about 5 million | 420 MB | 1.1 GB |

Pick `islands` if you've been to places like the Maldives: GeoNames files many inhabited islands as islands rather than towns, so `cities` leaves them out. A rerun replaces `geo.db` only once the new one is complete; restart the GUI to pick it up.

Place data is from [GeoNames](https://www.geonames.org/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `geo fetch` prints that attribution.

**From EXIF:** indexing reads each image's GPS position and resolves it to the nearest place. Positions read before the first `geo fetch` resolve on the next index run. There's no distance cap, so a photo taken at sea gets the nearest town, and one near a border can get the town across it.

**By hand:** in the viewer, the bar at the bottom shows the image's place, marked "(exif)" or "(manual)". Edit searches by name, including alternate names ("Lisboa", "München" and "Munich" all work), and lists recently used places. In the grid, "Set location" applies a place to the selection, or to the image under the cursor.

**Events:** after you set a place, vitrine finds the photos taken around the same time (no more than `event_gap_hours` apart, across all folders) and offers to apply the place to those that have none yet. Show lets you check them first. Photos without an EXIF capture time fall back to a `YYYY-MM-DD` file name prefix, as shelf writes it, and group by day. Nothing changes without your confirmation, and photos that already have a place are left alone.

Manual always wins: an index run never replaces a place you set, and clearing a place keeps the EXIF position from coming back. Deleting `geo.db` loses the search, not your tags. Locations are never written back to the files.

The filter bar narrows the grid by country, region and city, or to images with no location. `vitrine status` shows how many images are located, from EXIF and by hand, and how many still wait for `geo.db`.

## Map

The country and city filters work when I know where I'm looking. The map answers the other question: where did I take photos at all. "Map" in the top bar swaps the grid for a world map of whatever the grid would show, so all filters apply.

A place set by hand wins over EXIF, so a photo you set to Porto shows in Porto. Images marked "no location", or with a manual place that has no coordinates, stay off the map. Nearby images are grouped into clusters with a count and a thumbnail.

- **Drag** pans, **scroll** zooms around the pointer, **double-click** zooms in.
- **Click a cluster** to filter the grid to its area. The filter bar shows it as "Area"; "Remove" or "Clear filters" drops it. Back on the map, the view stays where you left it, with the area outlined.
- **Show on map** in the viewer's location bar opens the map centred on that image.

The map works offline. The basemap is [Natural Earth](https://www.naturalearthdata.com/) 1:50m (land, lakes, borders and country names), compiled into the binary. That's enough from the whole world down to a region; zoom in further and the coastlines get coarse (1:50m puts central Lisbon in the Tagus). City names come from `geo.db`; without it, the map works, just without them.

## shelf

To index new photos after each `shelf run`, add this to the shelf profile:

```toml
[post_run]
command = ["vitrine", "index"]
```

vitrine doesn't read shelf's database; it walks the configured roots again.

## Building

Needs Rust 1.95+, libheif 1.23+ and libjpeg-turbo 3 (via pkg-config), and at runtime ONNX Runtime 1.28+ for faces and `ffmpeg`/`ffprobe` for videos. On Guix, the manifest provides all of it:

```bash
guix shell -m manifest.scm -- cargo build --release
guix shell -m manifest.scm -- cargo test
```

If your distro ships older libheif or libjpeg-turbo, `cargo build --release --features vendored` builds both from source and links them statically. That needs `cmake` and `nasm`, plus the libde265 and dav1d headers for HEIC and AVIF. The release packages are built this way.

Guix's Rust 1.95 ships without rustfmt, clippy and rustdoc, so doctests are off and `cargo fmt` / `cargo clippy` need a separate toolchain.

## License

GPL-3.0-only. The image loader and the tiled GL renderer are adapted from [oculante](https://github.com/woelper/oculante) (MIT); see `NOTICE`. The basemap is made with [Natural Earth](https://www.naturalearthdata.com/) (public domain); `examples/pack_basemap.rs` builds `assets/basemap.vmap` from its 1:50m data.
