import Foundation
import Vision

var results: [String: [[String: Any]]] = [:]
for file in CommandLine.arguments.dropFirst() {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    request.usesLanguageCorrection = false
    request.minimumTextHeight = 0.006
    try VNImageRequestHandler(url: URL(fileURLWithPath: file)).perform([request])
    results[URL(fileURLWithPath: file).lastPathComponent] = (request.results ?? []).compactMap { entry in
        guard let candidate = entry.topCandidates(1).first else { return nil }
        let box = entry.boundingBox
        return ["text": candidate.string, "confidence": candidate.confidence, "box": [box.minX, box.minY, box.width, box.height]]
    }
}
let payload = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(payload)
