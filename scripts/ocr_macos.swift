#!/usr/bin/env swift
import Foundation
import Vision

struct OCRRecord: Codable {
    let hash: String
    let text: String
    let observations: Int
    let elapsedMilliseconds: Int
}

func usage() -> Never {
    fputs("Usage: ocr_macos.swift IMAGE_DIR OUTPUT_JSONL [LIMIT] [fast|accurate]\n", stderr)
    exit(2)
}

guard CommandLine.arguments.count >= 3 else { usage() }
let imageDirectory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let limit = CommandLine.arguments.count >= 4 ? Int(CommandLine.arguments[3]) ?? 0 : 0
let mode = CommandLine.arguments.count >= 5 ? CommandLine.arguments[4] : "fast"
guard mode == "fast" || mode == "accurate" else { usage() }

let manager = FileManager.default
let files = try manager.contentsOfDirectory(
    at: imageDirectory,
    includingPropertiesForKeys: nil,
    options: [.skipsHiddenFiles]
).filter { $0.pathExtension.lowercased() == "jpg" }.sorted { $0.lastPathComponent < $1.lastPathComponent }

var completed = Set<String>()
if manager.fileExists(atPath: outputURL.path),
   let existing = try? String(contentsOf: outputURL, encoding: .utf8) {
    for line in existing.split(separator: "\n") {
        if let data = line.data(using: .utf8),
           let record = try? JSONDecoder().decode(OCRRecord.self, from: data) {
            completed.insert(record.hash)
        }
    }
} else {
    manager.createFile(atPath: outputURL.path, contents: nil)
}

let handle = try FileHandle(forWritingTo: outputURL)
try handle.seekToEnd()
defer { try? handle.close() }
let encoder = JSONEncoder()
var processed = 0

for file in files {
    let digest = file.deletingPathExtension().lastPathComponent
    if completed.contains(digest) { continue }
    if limit > 0 && processed >= limit { break }
    let started = DispatchTime.now().uptimeNanoseconds
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = mode == "accurate" ? .accurate : .fast
    request.usesLanguageCorrection = mode == "accurate"
    request.recognitionLanguages = ["en-US"]
    request.minimumTextHeight = 0.006
    let handler = VNImageRequestHandler(url: file, options: [:])
    try handler.perform([request])
    let observations = (request.results ?? []).sorted {
        if abs($0.boundingBox.midY - $1.boundingBox.midY) > 0.015 {
            return $0.boundingBox.midY > $1.boundingBox.midY
        }
        return $0.boundingBox.minX < $1.boundingBox.minX
    }
    let strings = observations.compactMap { $0.topCandidates(1).first?.string }
    let elapsed = Int((DispatchTime.now().uptimeNanoseconds - started) / 1_000_000)
    let record = OCRRecord(
        hash: digest,
        text: strings.joined(separator: " "),
        observations: observations.count,
        elapsedMilliseconds: elapsed
    )
    var data = try encoder.encode(record)
    data.append(0x0A)
    try handle.write(contentsOf: data)
    processed += 1
    if processed % 100 == 0 {
        fputs("ocr \(processed) new images\n", stderr)
    }
}

fputs("completed \(processed) new images\n", stderr)
