// retinareach-cli: the phone-side procedure on a Mac.
//
//   retinareach-cli calibrate --models <export dir> --captures <dir> [--n N] [--prior N0]
//                             [--units all|ane|cpu] [--preprocessed] --out device.json
//   retinareach-cli screen    --models <export dir> [--device device.json]
//                             [--units ...] [--preprocessed] IMAGE...
//   retinareach-cli preprocess --size S [--preprocessed] --out <dir> IMAGE...
//
// calibrate: preprocess the captures (sorted by file name, first N, full batches
// only), measure the camera's BatchNorm statistics with the calibrator, look the
// camera up against the validated profiles, print the capability report and
// store it. screen: per-target probabilities under the stored statistics, with a
// flag at the shipped threshold for the targets the report shows.
// preprocess: dump the prepared S x S bytes (for the parity check).
import CoreML
import Foundation
import ImageIO
import RetinaReachKit
import UniformTypeIdentifiers

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(("error: " + msg + "\n").data(using: .utf8)!)
    exit(2)
}

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    fail("usage: retinareach-cli calibrate|screen|preprocess ... (see the header of main.swift)")
}
args.removeFirst()
var opts: [String: String] = [:]
var flags = Set<String>()
var positional: [String] = []
var i = 0
while i < args.count {
    let a = args[i]
    if a == "--preprocessed" {
        flags.insert(a)
    } else if a.hasPrefix("--") {
        guard i + 1 < args.count else { fail("\(a) needs a value") }
        opts[a] = args[i + 1]
        i += 1
    } else {
        positional.append(a)
    }
    i += 1
}

func units() -> MLComputeUnits {
    switch opts["--units"] ?? "all" {
    case "ane": return .cpuAndNeuralEngine
    case "cpu": return .cpuOnly
    default: return .all
    }
}

func imageFiles(in dir: URL) -> [URL] {
    let exts: Set<String> = ["jpg", "jpeg", "png", "tif", "tiff"]
    let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
    return items.filter { exts.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    enc.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
    try enc.encode(value).write(to: url)
}

func savePNG(_ img: RGBImage, to url: URL) throws {
    var rgba = [UInt8](repeating: 255, count: img.width * img.height * 4)
    for p in 0..<(img.width * img.height) {
        rgba[4 * p] = img.pixels[3 * p]; rgba[4 * p + 1] = img.pixels[3 * p + 1]; rgba[4 * p + 2] = img.pixels[3 * p + 2]
    }
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let cg = CGImage(width: img.width, height: img.height, bitsPerComponent: 8, bitsPerPixel: 32,
                     bytesPerRow: img.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                     bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                     provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { fail("cannot write \(url.path)") }
}

do {
    switch command {
    case "preprocess":
        guard let s = opts["--size"].flatMap(Int.init), let out = opts["--out"] else { fail("preprocess needs --size and --out") }
        let dir = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in positional {
            let u = URL(fileURLWithPath: f)
            let img = try Preprocess.prepare(u, size: s, preprocessed: flags.contains("--preprocessed"))
            try savePNG(img, to: dir.appendingPathComponent(u.deletingPathExtension().lastPathComponent + ".png"))
        }

    case "calibrate":
        guard let models = opts["--models"], let caps = opts["--captures"], let out = opts["--out"] else {
            fail("calibrate needs --models, --captures and --out")
        }
        let rr = try await RetinaReach(directory: URL(fileURLWithPath: models), computeUnits: units())
        var files = imageFiles(in: URL(fileURLWithPath: caps))
        if let n = opts["--n"].flatMap(Int.init) { files = Array(files.prefix(n)) }
        let S = rr.profiles.imageSize
        let t0 = Date()
        let images = try files.map { try Preprocess.prepare($0, size: S, preprocessed: flags.contains("--preprocessed")) }
        let t1 = Date()
        let cal = try rr.calibrate(images, priorStrength: Double(opts["--prior"] ?? "0") ?? 0)
        let t2 = Date()
        try writeJSON(cal, to: URL(fileURLWithPath: out))
        let r = cal.report
        print("calibrated on \(cal.nImages) captures (preprocess \(String(format: "%.2f", t1.timeIntervalSince(t0))) s, "
              + "calibrator \(String(format: "%.2f", t2.timeIntervalSince(t1))) s)")
        print("nearest profile: \(r.nearestProfile ?? "none")  inside envelope: \(r.insideEnvelope)"
              + (r.radius.map { "  (radius \(String(format: "%.3g", $0)))" } ?? ""))
        for (k, v) in r.distances.sorted(by: { $0.key < $1.key }) { print("  distance to \(k): \(String(format: "%.4g", v))") }
        print("shown on this camera: \(r.shown.isEmpty ? "none" : r.shown.joined(separator: ", "))")
        for t in r.targets {
            print("  \(t.target.padding(toLength: 22, withPad: " ", startingAt: 0)) "
                  + "\(t.status.padding(toLength: 18, withPad: " ", startingAt: 0)) \(t.reason)")
        }

    case "screen":
        guard let models = opts["--models"] else { fail("screen needs --models") }
        let rr = try await RetinaReach(directory: URL(fileURLWithPath: models), computeUnits: units())
        var cal: Calibration? = nil
        if let d = opts["--device"] {
            cal = try JSONDecoder().decode(Calibration.self, from: Data(contentsOf: URL(fileURLWithPath: d)))
        }
        var results: [ScreenResult] = []
        for f in positional {
            let u = URL(fileURLWithPath: f)
            let img = try Preprocess.prepare(u, size: rr.profiles.imageSize, preprocessed: flags.contains("--preprocessed"))
            results.append(try rr.screen(img, file: u.lastPathComponent, calibration: cal))
        }
        if let out = opts["--out"] { try writeJSON(results, to: URL(fileURLWithPath: out)) }
        for r in results {
            let shown = r.flagged.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value ? "REFER" : "no")" }
            print("\(r.file): \(shown.isEmpty ? "(no target shown on this camera)" : shown.joined(separator: " "))")
        }

    default:
        fail("unknown command \(command)")
    }
} catch {
    fail("\(error)")
}
