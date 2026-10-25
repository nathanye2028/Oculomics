import Foundation

/// One row of a validated device's capability table (written by the gate in
/// `train_retinareach.py`, carried into `retinareach_profiles.json` by
/// `export_retinareach.py`).
public struct CapabilityRow: Codable {
    public let target: String
    public let protocolName: String?
    public let status: String
    public let reason: String?
    public let auroc: Double?
    public let delta: Double?
    public let sensitivity: Double?

    enum CodingKeys: String, CodingKey {
        case target, status, reason, auroc, delta, sensitivity
        case protocolName = "protocol"
    }
}

/// A validated camera: its calibrated BatchNorm statistics, the same-camera
/// distance envelope by calibration size, and its capability rows.
public struct DeviceProfile: Codable {
    public let dataset: String
    public let nPool: Int
    public let envelope: [String: Double]
    public let capability: [CapabilityRow]
    public let bnStats: [Float]

    enum CodingKeys: String, CodingKey {
        case dataset, envelope, capability
        case nPool = "n_pool"
        case bnStats = "bn_stats"
    }

    /// Envelope radius for an N-image calibration: the smallest tabulated N >= n
    /// (more images -> tighter), else the largest tabulated N (retinareach.DeviceProfile.radius).
    public func radius(n: Int) -> Double? {
        let table = envelope.compactMap { k, v in Int(k).map { ($0, v) } }.sorted { $0.0 < $1.0 }
        guard let last = table.last else { return nil }
        return (table.first { $0.0 >= n } ?? last).1
    }
}

public struct StatsLayout: Codable {
    public let order: [String]
    public let channels: [Int]
}

/// One source target's label-free threshold reference
/// (retinareach.ThresholdReference): fit once on the source validation outputs
/// of the calibrated protocol, shipped so the device can re-set its threshold
/// from unlabelled captures. `sourceSpec` is absent in references written
/// before the `rate` method existed.
public struct ThresholdReference: Codable {
    public let threshold: Double
    public let targetSens: Double
    public let prior: Double
    public let plattA: Double
    public let plattB: Double
    public let anchorQ: Double
    public let anchorLogit: Double
    public let sourceSpec: Double?

    enum CodingKeys: String, CodingKey {
        case threshold, prior
        case targetSens = "target_sens"
        case plattA = "platt_a"
        case plattB = "platt_b"
        case anchorQ = "anchor_q"
        case anchorLogit = "anchor_logit"
        case sourceSpec = "source_spec"
    }
}

/// How the device recalibrates (`calibration` in retinareach_profiles.json):
/// which profile is the source, and how many of the BatchNorm layers, from the
/// input, shallow mode takes from the camera.
public struct CalibrationSpec: Codable {
    public let sourceProfile: String?
    public let nBnLayers: Int
    public let shallowLayers: Int
    public let shallowFrac: Double?

    enum CodingKeys: String, CodingKey {
        case sourceProfile = "source_profile"
        case nBnLayers = "n_bn_layers"
        case shallowLayers = "shallow_layers"
        case shallowFrac = "shallow_frac"
    }
}

/// `retinareach_profiles.json`: the contract between the Python build and the app.
public struct ProfilesFile: Codable {
    public let targets: [String]
    public let sourceTargets: [String]
    public let probeTargets: [String]
    public let imageSize: Int
    public let calibBatch: Int
    public let statsLayout: StatsLayout
    public let trainedStats: [Float]
    public let probeBuild: Bool
    public let profiles: [String: DeviceProfile]
    public let thresholds: [String: [String: Double?]]
    /// Absent in files written before the device re-set its own threshold.
    public let thresholdReference: [String: ThresholdReference]?
    /// Absent in files written before shallow mode: full calibration only.
    public let calibration: CalibrationSpec?

    enum CodingKeys: String, CodingKey {
        case targets, profiles, thresholds, calibration
        case sourceTargets = "source_targets"
        case probeTargets = "probe_targets"
        case imageSize = "image_size"
        case calibBatch = "calib_batch"
        case statsLayout = "stats_layout"
        case trainedStats = "trained_stats"
        case probeBuild = "probe_build"
        case thresholdReference = "threshold_reference"
    }

    public static func load(_ url: URL) throws -> ProfilesFile {
        try JSONDecoder().decode(ProfilesFile.self, from: Data(contentsOf: url))
    }

    public var statsLength: Int { 2 * statsLayout.channels.reduce(0, +) }

    /// The shipped (self-calibrated protocol) threshold for a target, if one was fixed.
    public func threshold(_ target: String) -> Double? {
        (thresholds["calibrated"] ?? [:])[target] ?? nil
    }

    /// The source profile's calibrated statistics (the tabletop camera the model
    /// was built on): the layers shallow mode keeps, and the statistics the
    /// captures are scored under to estimate the clinic's prevalence.
    public var sourceStats: [Float]? {
        guard let name = calibration?.sourceProfile, let p = profiles[name] else { return nil }
        return p.bnStats
    }
}
