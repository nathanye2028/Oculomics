// swift-tools-version:5.9
// RetinaReachKit: the phone-side half of RetinaReach -- preprocessing, on-device
// self-calibration, the capability lookup and screening -- on Core ML's public
// API, identical on macOS and iOS. `retinareach-cli` runs it on a Mac against the
// packages written by export_retinareach.py.
import PackageDescription

let package = Package(
    name: "RetinaReachKit",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "RetinaReachKit", targets: ["RetinaReachKit"]),
        .executable(name: "retinareach-cli", targets: ["retinareach-cli"]),
    ],
    targets: [
        .target(name: "RetinaReachKit"),
        .executableTarget(name: "retinareach-cli", dependencies: ["RetinaReachKit"]),
    ]
)
