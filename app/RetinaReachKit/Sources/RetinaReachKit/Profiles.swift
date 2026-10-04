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

    enum CodingKeys: String, CodingKey {
        case targets, profiles, thresholds
        case sourceTargets = "source_targets"
        case probeTargets = "probe_targets"
        case imageSize = "image_size"
        case calibBatch = "calib_batch"
        case statsLayout = "stats_layout"
        case trainedStats = "trained_stats"
        case probeBuild = "probe_build"
    }

    public static func load(_ url: URL) throws -> ProfilesFile {
        try JSONDecoder().decode(ProfilesFile.self, from: Data(contentsOf: url))
    }

    public var statsLength: Int { 2 * statsLayout.channels.reduce(0, +) }

    /// The shipped (self-calibrated protocol) threshold for a target, if one was fixed.
    public func threshold(_ target: String) -> Double? {
        (thresholds["calibrated"] ?? [:])[target] ?? nil
    }
}
