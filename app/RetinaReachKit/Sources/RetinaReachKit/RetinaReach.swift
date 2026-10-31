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

/// Which BatchNorm layers the camera's measured statistics replace at screening time.
public enum CalibrationMode: String, Codable {
    /// Every layer (AdaBN). On BRSET -> mBRSET this raised the AUROC and cut the
    /// sensitivity at the shipped threshold (REPORT 10.6-10.8).
    case full
    /// The first `calibration.shallow_layers` layers only; the deeper, more
    /// disease-specific statistics stay the source profile's (REPORT 10.8).
    case shallow
}

/// How the per-target threshold is set after calibration (retinareach.label_free_threshold).
public enum ThresholdMethod: String, Codable {
    /// The threshold fixed on the source camera's validation split, unchanged.
    case shipped
    /// Prevalence by EM on the captures scored under the source profile, posteriors
    /// carried to the same captures' calibrated scores, threshold by expected sensitivity.
    case em2
    /// Rank only: flag the top fraction expected from the estimated prevalence and the
    /// source's false-alarm rate.
    case rate
}

/// The stored result of calibrating to one camera.
public struct Calibration: Codable {
    public let nImages: Int
    /// Measured statistics: drives the camera lookup (envelopes are unblended).
    public let measured: [Float]
    /// What every later screen passes to the model: `measured` (full), its first
    /// `shallow_layers` layers over the source profile (shallow), or `measured`
    /// blended with the trained statistics when `priorStrength` > 0.
    public let inference: [Float]
    public let priorStrength: Double
    public let report: DeviceReport
    /// `full` or `shallow`; absent in calibrations stored before shallow mode (full).
    public let mode: String?
    /// The threshold method the stored thresholds came from; absent = shipped.
    public let thresholdMethod: String?
    /// Per source target: the threshold this camera screens at when it was re-set.
    public let thresholds: [String: Double]?
    /// Per source target: the clinic's prevalence estimated from the captures.
    public let estimatedPrevalence: [String: Double]?
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

    /// Shallow recalibration (retinareach.calibrate_blended with `adapt` = the first K
    /// layers): the first `calibration.shallow_layers` layers' entries come from the
    /// measured statistics, the rest from the source profile. The collector needs no
    /// change: layer k's batch statistics depend only on the layers below it, so the
    /// measured vector's first K entries are exactly the shallow calibration's.
    public func shallowVector(_ measured: [Float]) -> [Float] {
        guard let spec = profiles.calibration, let src = profiles.sourceStats, src.count == measured.count
        else { return measured }
        var out = src
        var i = 0
        for (layer, c) in profiles.statsLayout.channels.enumerated() {
            if layer < spec.shallowLayers { for j in i..<(i + 2 * c) { out[j] = measured[j] } }
            i += 2 * c
        }
        return out
    }

    public func calibrate(_ images: [RGBImage], mode: CalibrationMode = .full, priorStrength: Double = 0,
                          thresholdMethod: ThresholdMethod = .shipped) throws -> Calibration {
        let (m, n) = try measure(images)
        let inference: [Float]
        switch mode {
        case .full: inference = blend(m, nImages: n, priorStrength: priorStrength)
        case .shallow: inference = shallowVector(m)
        }
        var thresholds: [String: Double] = [:], prevalence: [String: Double] = [:]
        if thresholdMethod != .shipped, let refs = profiles.thresholdReference, let src = profiles.sourceStats {
            // The same captures scored under the source profile (where the Platt map was
            // fit: the prevalence estimate) and under the calibrated statistics (what the
            // device will act on: where the threshold is placed).
            let qRef = try images.map { try probabilities($0, stats: src) }
            let qCal = try images.map { try probabilities($0, stats: inference) }
            for (t, ref) in refs {
                guard let k = profiles.targets.firstIndex(of: t) else { continue }
                let (thr, pi) = labelFreeThreshold(qCal.map { Double($0[k]) }, reference: qRef.map { Double($0[k]) },
                                                   ref: ref, method: thresholdMethod)
                if let thr = thr { thresholds[t] = thr; prevalence[t] = pi }
            }
        }
        return Calibration(nImages: n, measured: m, inference: inference, priorStrength: priorStrength,
                           report: lookup(m, nImages: n), mode: mode.rawValue,
                           thresholdMethod: thresholdMethod.rawValue, thresholds: thresholds,
                           estimatedPrevalence: prevalence)
    }

    // MARK: label-free threshold (retinareach.label_free_threshold, em2 / rate)

    /// Saerens, Latinne & Decaestecker (2002): EM re-estimate of the class prior from
    /// posteriors `pCal` calibrated at `prior` (retinareach.em_prevalence, same arithmetic).
    public func emPrevalence(_ pCal: [Double], prior: Double, iters: Int = 200, tol: Double = 1e-7) -> (Double, [Double]) {
        var pi = prior, post = pCal
        for _ in 0..<iters {
            post = pCal.map { p in
                let a = (pi / prior) * p, b = ((1 - pi) / (1 - prior)) * (1 - p)
                return a / max(a + b, 1e-12)
            }
            let new = post.reduce(0, +) / Double(max(post.count, 1))
            let done = abs(new - pi) < tol
            pi = new
            if done { break }
        }
        return (pi, post)
    }

    /// Highest threshold whose expected sensitivity under soft labels reaches `target`
    /// (retinareach.expected_sensitivity_threshold).
    public func expectedSensitivityThreshold(_ scores: [Double], soft: [Double], target: Double) -> Double {
        let order = scores.indices.sorted { scores[$0] > scores[$1] }
        let total = max(soft.reduce(0, +), 1e-12)
        var cum = 0.0
        for i in order {
            cum += soft[i] / total
            if cum >= target { return scores[i] }
        }
        return scores[order.last!]
    }

    /// The threshold the device applies after scoring its captures under the calibrated
    /// statistics (`qCal`) and under the source profile (`qRef`, same captures, same order).
    /// Returns (threshold or nil when the method cannot run, estimated prevalence).
    public func labelFreeThreshold(_ qCal: [Double], reference qRef: [Double], ref: ThresholdReference,
                                   method: ThresholdMethod) -> (Double?, Double) {
        guard !qCal.isEmpty, qCal.count == qRef.count else { return (nil, .nan) }
        let logit = { (p: Double) -> Double in let c = min(max(p, 1e-6), 1 - 1e-6); return log(c / (1 - c)) }
        let pCal = qRef.map { 1 / (1 + exp(-(ref.plattA * logit($0) + ref.plattB))) }
        let (pi, post) = emPrevalence(pCal, prior: ref.prior)
        switch method {
        case .shipped:
            return (ref.threshold, pi)
        case .em2:
            return (expectedSensitivityThreshold(qCal, soft: post, target: ref.targetSens), pi)
        case .rate:
            guard let spec = ref.sourceSpec else { return (nil, pi) }
            let f = min(max(pi * ref.targetSens + (1 - pi) * (1 - spec), 0), 1)
            let k = min(max(Int(ceil(f * Double(qCal.count))), 1), qCal.count)
            return (qCal.sorted(by: >)[k - 1], pi)
        }
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
            // the camera's own re-set threshold when the calibration carries one, else the shipped one
            if let thr = calibration?.thresholds?[t] ?? profiles.threshold(t), let v = probs[t] {
                flags[t] = Double(v) >= thr
            }
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
