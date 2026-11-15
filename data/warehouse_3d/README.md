# Warehouse 3D Viewer

An independent, offline tool for understanding the warehouse layout, its storage locations and the
storage data. It reads the SQLite review copy of the raw dataset and does not depend on the ML pipeline.

**Open:** double-click `Open Warehouse 3D.cmd` (or open `index.html` in Edge or Chrome).
No internet connection or server is needed.

## What you see

* **Floor:** the L-shaped building, with picking aisles and the left, centre and right cross aisles as grey strips.
* **Boxes:** each of the 2,292 rack locations, stacked by rack level (1 = floor, up to 4).
* **Grey discs:** the 44 aisle-end navigation points (LC, CC, RC).
* **Teal cube:** the I/O point (depot) at CC-08. This is the point marked in the authors' drawings,
  where pickers start and finish.

| Control | What it does |
|---|---|
| Colour locations by | Pick frequency, ABC class of the most-picked product, storage-plan class, rack level or rack |
| Storage plan | As recorded, or the Random / Class-based / Dedicated / Hybrid plans. Drives the details panel, product search and plan colouring. |
| Show levels | Hide upper levels to look inside the racks |
| Find | A location code (`H-06-13`) shows its details. A product (`8N10W9`) highlights every location that stores it under the chosen plan, or every location it was picked from (as recorded). |
| Picking wave | Highlights the wave's pick locations with pins. Use ← → or Random to step through waves. The URL `index.html#wave=43175` opens a wave directly. |
| Floor plans (SVG) | Shows the authors' drawings `data/raw/dataset_Layout_Z1.0.svg` to `Z4.0.svg`, one per level |
| Mouse | Left-drag rotates, right-drag pans, scroll zooms. Hover shows a location code; click shows its details. |

## Files

| File | Purpose |
|---|---|
| `index.html` | The viewer (three.js r147, saved locally in `vendor/`, MIT licence) |
| `warehouse_data.js` | Generated data (about 3 MB). Do not edit it. |
| `build_viewer.py` | Rebuilds `warehouse_data.js` from `../review/smartroute_review.sqlite`, and builds that database first if it is missing |
| `Open Warehouse 3D.cmd` | Launcher |

Rebuild after the data changes: `python build_viewer.py` (a few seconds).

## Notes

* Coordinates are drawing units of the authors' CAD layout. The details panel also gives metres
  (1 unit = 0.07625 m, the same scale as SmartRoute).
* Only rack locations are drawn. Picks from staging or consolidation areas (RC-01, ZN-COR, EX-xx) have no
  coordinates; a wave's panel shows how many there are.
* Pick frequencies count order lines (unit rows grouped by wave, product, size and location),
  as in `v_order_line`.
