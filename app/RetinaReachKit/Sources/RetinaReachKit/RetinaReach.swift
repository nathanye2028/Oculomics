import CoreML
import CoreVideo
import Foundation

public struct TargetStatus: Codable {
    public let target: String
    public let status: String
    public let reason: String
}

/// What the phone prints after calibrating on `nImages` captures
/// (retinareach.device_capability).
public struct DeviceReport: Codable {
    public let nearestProfile: String?
    public let distances: [String: Double]
    public let radius: Double?
    public let insideEnvelope: Bool
    public let nImages: Int
    public let shown: [String]
    public let targets: [TargetStatus]
}

/// The stored result of calibrating to one camera.
public struct Calibration: Codable {
    public let nImages: Int
    /// Measured statistics: drives the camera lookup (envelopes are unblended).
    public let measured: [Float]
    /// What every later screen passes to the model: `measured`, or blended with
    /// the trained statistics when `priorStrength` > 0.
    public let inference: [Float]
    public let priorStrength: Double
    public let report: DeviceReport
}

public struct ScreenResult: Codable {
    public let file: String
    public let probabilities: [String: Float]
    /// Only targets shown on this camera, flagged at their shipped threshold.
    public let flagged: [String: Bool]
}

public enum RetinaReachError: Error, CustomStringConvertible {
    case tooFewCaptures(have: Int, batch: Int)
    case badOutput(String)
    public var description: String {
        switch self {
        case .tooFewCaptures(let n, let b): return "calibration needs at least \(b) captures (one batch), got \(n)"
        case .badOutput(let s): return "unexpected model output: \(s)"
        }
    }
}

/// The phone-side model: two Core ML graphs (screen, calibrate) plus the profiles file.
public final class RetinaReach {
    public let profiles: ProfilesFile
    let screenModel: MLModel
    let calibModel: MLModel

    /// `directory` holds RetinaReach.mlpackage, RetinaReachCalibrator.mlpackage and
    /// retinareach_profiles.json as written by export_retinareach.py.
    public init(directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        profiles = try ProfilesFile.load(directory.appendingPathComponent("retinareach_profiles.json"))
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let screenURL = try await MLModel.compileModel(at: directory.appendingPathComponent("RetinaReach.mlpackage"))
        let calibURL = try await MLModel.compileModel(at: directory.appendingPathComponent("RetinaReachCalibrator.mlpackage"))
        screenModel = try MLModel(contentsOf: screenURL, configuration: cfg)
        calibModel = try MLModel(contentsOf: calibURL, configuration: cfg)
    }

    // MARK: self-calibration

    /// Average of the calibrator over full batches of `calibBatch` captures
    /// (equal weight per batch, as AdaBN). A trailing partial batch is not used.
    public func measure(_ images: [RGBImage]) throws -> (vector: [Float], nImages: Int) {
        let B = profiles.calibBatch, S = profiles.imageSize
        let nBatches = images.count / B
        guard nBatches > 0 else { throw RetinaReachError.tooFewCaptures(have: images.count, batch: B) }
        var acc = [Double](repeating: 0, count: profiles.statsLength)
        for b in 0..<nBatches {
            let arr = try MLMultiArray(shape: [B, 3, S, S].map { NSNumber(value: $0) }, dataType: .float32)
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: B * 3 * S * S)
            for (k, img) in images[(b * B)..<((b + 1) * B)].enumerated() {
                fillCHW(p + k * 3 * S * S, img, size: S)
            }
            let out = try calibModel.prediction(from: MLDictionaryFeatureProvider(
                dictionary: ["images": MLFeatureValue(multiArray: arr)]))
            guard let v = out.featureValue(for: "bn_stats")?.multiArrayValue else {
                throw RetinaReachError.badOutput("calibrator has no bn_stats")
            }
            let f = floats(v)
            guard f.count == acc.count else { throw RetinaReachError.badOutput("bn_stats has \(f.count) values") }
            for i in 0..<f.count { acc[i] += Double(f[i]) }
        }
        return (acc.map { Float($0 / Double(nBatches)) }, nBatches * B)
    }

    /// Schneider et al. (2020) shrinkage toward the trained statistics, pseudo-count N0.
    public func blend(_ measured: [Float], nImages: Int, priorStrength: Double) -> [Float] {
        guard priorStrength > 0 else { return measured }
        let a = Float(priorStrength / (priorStrength + Double(nImages)))
        return zip(profiles.trainedStats, measured).map { a * $0 + (1 - a) * $1 }
    }

    public func calibrate(_ images: [RGBImage], priorStrength: Double = 0) throws -> Calibration {
        let (m, n) = try measure(images)
        return Calibration(nImages: n, measured: m, inference: blend(m, nImages: n, priorStrength: priorStrength),
                           priorStrength: priorStrength, report: lookup(m, nImages: n))
    }

    // MARK: camera signature and capability lookup

    /// retinareach.camera_distance: per-channel symmetric KL of the BN Gaussians,
    /// variances floored at relFloor x the layer's (lower) median variance,
    /// (lower) median over channels, mean over layers.
    public func cameraDistance(_ a: [Float], _ b: [Float], relFloor: Double = 1e-3) -> Double {
        var perLayer: [Double] = []
        var i = 0
        for c in profiles.statsLayout.channels {
            let va = a[(i + c)..<(i + 2 * c)].map(Double.init), vb = b[(i + c)..<(i + 2 * c)].map(Double.init)
            let floor = relFloor * lowerMedian(va + vb) + 1e-12
            var skl = [Double](repeating: 0, count: c)
            for j in 0..<c {
                let x = max(va[j], floor), y = max(vb[j], floor)
                let dm = Double(a[i + j]) - Double(b[i + j])
                skl[j] = 0.5 * (x / y + y / x - 2.0) + 0.5 * dm * dm * (1.0 / x + 1.0 / y)
            }
            perLayer.append(lowerMedian(skl))
            i += 2 * c
        }
        return perLayer.isEmpty ? .nan : perLayer.reduce(0, +) / Double(perLayer.count)
    }

    /// retinareach.device_capability: nearest validated profile; inside its
    /// envelope -> that profile's calibrated-protocol statuses; outside every
    /// envelope -> NOT_VALIDATED for every target. Never promotes a status.
    public func lookup(_ measured: [Float], nImages: Int) -> DeviceReport {
        let targets = profiles.targets
        guard !profiles.profiles.isEmpty else {
            return DeviceReport(nearestProfile: nil, distances: [:], radius: nil, insideEnvelope: false,
                                nImages: nImages, shown: [],
                                targets: targets.map { TargetStatus(target: $0, status: "NOT_VALIDATED",
                                                                    reason: "no validated device profiles in this build") })
        }
        var dist: [String: Double] = [:]
        for (name, p) in profiles.profiles { dist[name] = cameraDistance(measured, p.bnStats) }
        let near = dist.min { $0.value < $1.value }!.key
        let prof = profiles.profiles[near]!
        let radius = prof.radius(n: nImages)
        let inside = radius.map { dist[near]! <= $0 } ?? false
        var rows: [String: CapabilityRow] = [:]
        for r in prof.capability where r.protocolName == "calibrated" && rows[r.target] == nil { rows[r.target] = r }
        let statuses: [TargetStatus] = targets.map { t in
            if !inside {
                let r = radius.map { String(format: "%.3g", $0) } ?? "n/a"
                return TargetStatus(target: t, status: "NOT_VALIDATED",
                                    reason: "camera outside every validated profile (nearest \(near): distance "
                                        + String(format: "%.3g", dist[near]!) + " > envelope \(r) at N=\(nImages))")
            }
            guard let row = rows[t] else {
                return TargetStatus(target: t, status: "NOT_VALIDATED", reason: "no evaluation of \(t) on \(near)")
            }
            return TargetStatus(target: t, status: row.status, reason: row.reason ?? "")
        }
        return DeviceReport(nearestProfile: near, distances: dist, radius: radius, insideEnvelope: inside,
                            nImages: nImages, shown: statuses.filter { $0.status == "SUPPORTED" }.map { $0.target },
                            targets: statuses)
    }

    // MARK: screening

    /// Per-target probabilities for one prepared S x S image under `stats`.
    public func probabilities(_ img: RGBImage, stats: [Float]) throws -> [Float] {
        let S = profiles.imageSize
        let statsArr = try MLMultiArray(shape: [NSNumber(value: stats.count)], dataType: .float32)
        let sp = statsArr.dataPointer.bindMemory(to: Float.self, capacity: stats.count)
        for i in 0..<stats.count { sp[i] = stats[i] }
        let imageFeature: MLFeatureValue
        if screenModel.modelDescription.inputDescriptionsByName["image"]?.type == .image {
            imageFeature = MLFeatureValue(pixelBuffer: try pixelBuffer(img))
        } else {
            let arr = try MLMultiArray(shape: [1, 3, S, S].map { NSNumber(value: $0) }, dataType: .float32)
            fillCHW(arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * S * S), img, size: S)
            imageFeature = MLFeatureValue(multiArray: arr)
        }
        let out = try screenModel.prediction(from: MLDictionaryFeatureProvider(
            dictionary: ["image": imageFeature, "bn_stats": MLFeatureValue(multiArray: statsArr)]))
        guard let p = out.featureValue(for: "probabilities")?.multiArrayValue else {
            throw RetinaReachError.badOutput("model has no probabilities")
        }
        return floats(p)
    }

    public func screen(_ img: RGBImage, file: String, calibration: Calibration?) throws -> ScreenResult {
        let stats = calibration?.inference ?? profiles.trainedStats
        let p = try probabilities(img, stats: stats)
        var probs: [String: Float] = [:], flags: [String: Bool] = [:]
        for (t, v) in zip(profiles.targets, p) { probs[t] = v }
        for t in calibration?.report.shown ?? [] {
            if let thr = profiles.threshold(t), let v = probs[t] { flags[t] = Double(v) >= thr }
        }
        return ScreenResult(file: file, probabilities: probs, flagged: flags)
    }
}

// MARK: helpers

func lowerMedian(_ x: [Double]) -> Double {
    guard !x.isEmpty else { return .nan }
    let s = x.sorted()
    return s[(s.count - 1) / 2]                 // torch.median: the lower of the two middle values
}

func fillCHW(_ p: UnsafeMutablePointer<Float>, _ img: RGBImage, size S: Int) {
    for y in 0..<S {
        for x in 0..<S {
            let i = 3 * (y * S + x)
            p[y * S + x] = Float(img.pixels[i])
            p[S * S + y * S + x] = Float(img.pixels[i + 1])
            p[2 * S * S + y * S + x] = Float(img.pixels[i + 2])
        }
    }
}

func floats(_ a: MLMultiArray) -> [Float] {
    switch a.dataType {
    case .float32:
        return Array(UnsafeBufferPointer(start: a.dataPointer.bindMemory(to: Float.self, capacity: a.count), count: a.count))
    case .float16:
        let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count)
        return (0..<a.count).map { Float(p[$0]) }
    default:
        return (0..<a.count).map { a[$0].floatValue }
    }
}

func pixelBuffer(_ img: RGBImage) throws -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                 kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
    guard CVPixelBufferCreate(kCFAllocatorDefault, img.width, img.height, kCVPixelFormatType_32BGRA,
                              attrs, &pb) == kCVReturnSuccess, let buf = pb else {
        throw RetinaReachError.badOutput("cannot allocate a pixel buffer")
    }
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    let base = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
    let bpr = CVPixelBufferGetBytesPerRow(buf)
    for y in 0..<img.height {
        for x in 0..<img.width {
            let i = 3 * (y * img.width + x), o = y * bpr + 4 * x
            base[o] = img.pixels[i + 2]; base[o + 1] = img.pixels[i + 1]
            base[o + 2] = img.pixels[i]; base[o + 3] = 255
        }
    }
    return buf
}
