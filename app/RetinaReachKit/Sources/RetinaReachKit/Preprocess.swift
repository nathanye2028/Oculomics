import CoreGraphics
import Foundation
import ImageIO

/// 8-bit RGB, row-major, interleaved.
public struct RGBImage {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3, "RGBImage: pixel count mismatch")
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

public enum PreprocessError: Error, CustomStringConvertible {
    case unreadable(URL)
    public var description: String {
        switch self { case .unreadable(let u): return "cannot decode image \(u.path)" }
    }
}

/// The training loader's preprocessing, step for step (dataset.MBRSETDataset in
/// eval mode + export_coreml.preprocess_for_verify):
///
///  1. decode; for a JPEG, emulate PIL's `draft("RGB", (S, S))`: the largest
///     power-of-two reduction s in {8, 4, 2, 1} with s <= min(w / S, h / S),
///     output ceil(w / s) x ceil(h / s) (libjpeg's scaled IDCT is approximated
///     by an s x s box average);
///  2. crop to the field of view: the bounding box of pixels whose
///     max(R, G, B) > 12 (fundus_utils.fov_bbox);
///  3. full-frame resize to S x S with the antialiased bilinear filter
///     torchvision / PIL use (triangle kernel, support scaled by the reduction);
///  4. round to 8 bits -- the bytes Core ML is given.
public enum Preprocess {
    public static let fovTolerance: UInt8 = 12

    public static func decode(_ url: URL) throws -> (RGBImage, isJPEG: Bool) {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw PreprocessError.unreadable(url)
        }
        let type = (CGImageSourceGetType(src) as String?) ?? ""
        let w = cg.width, h = cg.height
        // Draw in the image's own RGB colour space: no colour matching, so the
        // bytes are the decoded values PIL would see.
        let space = (cg.colorSpace?.model == .rgb ? cg.colorSpace : nil)
            ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgbx = [UInt8](repeating: 0, count: w * h * 4)
        let ok: Bool = rgbx.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw PreprocessError.unreadable(url) }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[3 * i] = rgbx[4 * i]
            rgb[3 * i + 1] = rgbx[4 * i + 1]
            rgb[3 * i + 2] = rgbx[4 * i + 2]
        }
        return (RGBImage(width: w, height: h, pixels: rgb), type == "public.jpeg")
    }

    /// PIL JpegImageFile.draft's reduction for a requested (size, size).
    public static func draftScale(width: Int, height: Int, size: Int) -> Int {
        let scale = min(width / size, height / size)
        return [8, 4, 2, 1].first { scale >= $0 } ?? 1
    }

    public static func boxReduce(_ img: RGBImage, by s: Int) -> RGBImage {
        guard s > 1 else { return img }
        let ow = (img.width + s - 1) / s, oh = (img.height + s - 1) / s
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        for oy in 0..<oh {
            let y0 = oy * s, y1 = min(y0 + s, img.height)
            for ox in 0..<ow {
                let x0 = ox * s, x1 = min(x0 + s, img.width)
                var acc = (0, 0, 0)
                for y in y0..<y1 {
                    for x in x0..<x1 {
                        let i = 3 * (y * img.width + x)
                        acc.0 += Int(img.pixels[i]); acc.1 += Int(img.pixels[i + 1]); acc.2 += Int(img.pixels[i + 2])
                    }
                }
                let n = Double((y1 - y0) * (x1 - x0))
                let o = 3 * (oy * ow + ox)
                out[o] = UInt8((Double(acc.0) / n).rounded())
                out[o + 1] = UInt8((Double(acc.1) / n).rounded())
                out[o + 2] = UInt8((Double(acc.2) / n).rounded())
            }
        }
        return RGBImage(width: ow, height: oh, pixels: out)
    }

    /// fundus_utils.fov_bbox + crop_to_fov.
    public static func cropToFOV(_ img: RGBImage, tolerance: UInt8 = fovTolerance) -> RGBImage {
        var rowHit = [Bool](repeating: false, count: img.height)
        var colHit = [Bool](repeating: false, count: img.width)
        for y in 0..<img.height {
            for x in 0..<img.width {
                let i = 3 * (y * img.width + x)
                if max(img.pixels[i], img.pixels[i + 1], img.pixels[i + 2]) > tolerance {
                    rowHit[y] = true
                    colHit[x] = true
                }
            }
        }
        guard let r0 = rowHit.firstIndex(of: true), let r1 = rowHit.lastIndex(of: true),
              let c0 = colHit.firstIndex(of: true), let c1 = colHit.lastIndex(of: true) else { return img }
        let w = c1 - c0 + 1, h = r1 - r0 + 1
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let src = 3 * ((y + r0) * img.width + c0)
            out.replaceSubrange((3 * y * w)..<(3 * (y + 1) * w), with: img.pixels[src..<(src + 3 * w)])
        }
        return RGBImage(width: w, height: h, pixels: out)
    }

    /// Antialiased bilinear coefficients (PIL precompute_coeffs / ATen
    /// _compute_indices_min_size_weights_aa).
    static func coefficients(input: Int, output: Int) -> [(start: Int, weights: [Double])] {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1.0)
        let support = filterScale                      // bilinear: interp_size / 2 = 1
        let invScale = 1.0 / filterScale
        return (0..<output).map { i in
            let center = (Double(i) + 0.5) * scale
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), input)
            var w = (xmin..<xmax).map { x -> Double in
                let t = abs((Double(x) - center + 0.5) * invScale)
                return t < 1.0 ? 1.0 - t : 0.0
            }
            let total = w.reduce(0, +)
            if total > 0 { w = w.map { $0 / total } }
            return (xmin, w)
        }
    }

    /// Separable antialiased bilinear resize to size x size; returns rounded bytes.
    public static func resize(_ img: RGBImage, to size: Int) -> RGBImage {
        let cx = coefficients(input: img.width, output: size)
        let cy = coefficients(input: img.height, output: size)
        var horiz = [Double](repeating: 0, count: img.height * size * 3)
        for y in 0..<img.height {
            for (ox, c) in cx.enumerated() {
                var acc = (0.0, 0.0, 0.0)
                for (k, wk) in c.weights.enumerated() {
                    let i = 3 * (y * img.width + c.start + k)
                    acc.0 += wk * Double(img.pixels[i]); acc.1 += wk * Double(img.pixels[i + 1])
                    acc.2 += wk * Double(img.pixels[i + 2])
                }
                let o = 3 * (y * size + ox)
                horiz[o] = acc.0; horiz[o + 1] = acc.1; horiz[o + 2] = acc.2
            }
        }
        var out = [UInt8](repeating: 0, count: size * size * 3)
        for (oy, c) in cy.enumerated() {
            for x in 0..<size {
                var acc = (0.0, 0.0, 0.0)
                for (k, wk) in c.weights.enumerated() {
                    let i = 3 * ((c.start + k) * size + x)
                    acc.0 += wk * horiz[i]; acc.1 += wk * horiz[i + 1]; acc.2 += wk * horiz[i + 2]
                }
                let o = 3 * (oy * size + x)
                out[o] = UInt8(min(max(acc.0.rounded(), 0), 255))
                out[o + 1] = UInt8(min(max(acc.1.rounded(), 0), 255))
                out[o + 2] = UInt8(min(max(acc.2.rounded(), 0), 255))
            }
        }
        return RGBImage(width: size, height: size, pixels: out)
    }

    /// The full chain. `preprocessed`: the file is already an S x S model input
    /// (e.g. written by the Python parity check) and is used as is.
    public static func prepare(_ url: URL, size: Int, preprocessed: Bool = false) throws -> RGBImage {
        var (img, isJPEG) = try decode(url)
        if preprocessed {
            precondition(img.width == size && img.height == size, "preprocessed input must be \(size)x\(size)")
            return img
        }
        if isJPEG {
            img = boxReduce(img, by: draftScale(width: img.width, height: img.height, size: size))
        }
        return resize(cropToFOV(img), to: size)
    }
}
