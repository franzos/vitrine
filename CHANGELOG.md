# Changelog

## [0.1.2] - 2026-10-25

### Added
- Menu entry and app icon
- Name, assign and ignore faces from the info panel

### Fixed
- Grid sorts by capture time, falling back to the file name date

## [0.1.1] - 2026-10-12

### Added
- Locations from EXIF GPS or set by hand, with GeoNames place search
- `geo fetch` with coverage for inhabited islands or all places
- Suggestion to apply a place to photos from the same event
- Country, region, city and area filters
- World map with clusters, city names and "Show on map"
- Viewer info panel with location, people and details
- Searchable people and place filters
- Grid multi-select with Ctrl, Shift and Ctrl-A
- Video thumbnails, duration badge and playback in the configured player
- Release packages: `.deb` and a Linux binary

### Fixed
- Stalled model and gazetteer downloads now time out
- Viewer stays on its image after a location change
- Index lock no longer fails while a forked child still holds it
