# RetinaReachKit

The phone-side half of RetinaReach, in Swift on Core ML's public API (the same
on iOS and macOS). It does what the app does, nothing more:

1. **Preprocess** a capture exactly as the training loader did: JPEG draft
   reduction (PIL's rule), crop to the field of view (`max(R,G,B) > 12`),
   antialiased bilinear resize to S × S, 8-bit.
2. **Calibrate** to the camera: run `RetinaReachCalibrator.mlpackage` on the
   first captures in batches of B and average (AdaBN's statistics, no labels,
   no gradients).
3. **Look the camera up** against the validated device profiles
   (`camera_distance`, envelope by N) and decide which targets to show.
4. **Screen**: `RetinaReach.mlpackage` with the stored statistics vector; a
   shown target is flagged at its shipped threshold.

Inputs are the three files `export_retinareach.py` writes to one directory:
`RetinaReach.mlpackage`, `RetinaReachCalibrator.mlpackage`,
`retinareach_profiles.json`.

## Build and run on a Mac

```bash
swift build -c release --package-path app/RetinaReachKit
app/RetinaReachKit/.build/release/retinareach-cli calibrate --models edge_export/retinareach \
    --captures <folder of captures> --n 64 --out device.json
app/RetinaReachKit/.build/release/retinareach-cli screen --models edge_export/retinareach \
    --device device.json image1.jpg image2.jpg
```

`calibrate` prints the report the phone would show (nearest validated camera,
inside its envelope or not, status per target) and stores the statistics;
`screen` prints REFER / no for each target shown on that camera. `--units ane`
keeps Core ML on the Neural Engine; `--prior N0` shrinks a small calibration set
toward the trained statistics (the camera lookup always uses the unshrunk
measurement).

## In an app

```swift
let rr = try await RetinaReach(directory: modelsURL)             // compiles both packages once
let captures = try urls.map { try Preprocess.prepare($0, size: rr.profiles.imageSize) }
let calibration = try rr.calibrate(captures)                     // store it per camera
let result = try rr.screen(try Preprocess.prepare(photoURL, size: rr.profiles.imageSize),
                           file: photoURL.lastPathComponent, calibration: calibration)
// calibration.report.shown -> the targets to display; result.flagged -> referrals
```

## Parity with the Python build

`export_retinareach.py --verify-images <dir> --swift-cli <built cli>` runs this
package and Python on the same captures (also the opt-in test
`RR_SWIFT=1 pytest tests/test_retinareach.py -k swift`). On a small trained
checkpoint: calibration statistics identical to PyTorch AdaBN (camera distance
6e-11), capability statuses identical, screening probabilities within 5e-6. Crop
and resize match the training pipeline to within one gray level on 0.3 % of
values; the remaining preprocessing difference is the JPEG decoder (Apple
ImageIO vs the libjpeg PIL uses: ~24 % of values differ, mean 0.4 levels),
which moves the calibrated statistics by ~1.5 % of the calibration shift
itself. Live camera frames never go through either decoder.
