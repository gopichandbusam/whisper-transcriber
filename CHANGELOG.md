# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Timestamped transcript output with MM:SS format
- Audio preprocessing with .npy caching for faster subsequent processing
- Universal ETA calculations for all job statuses (preparing, queued, ready, processing)
- GPU detection with optimized performance estimates
- Multi-job dashboard with real-time progress tracking
- Segment-level progress mode for finer granularity
- Process-based transcription with cancellation support
- Comprehensive logging and status tracking
- Settings persistence in localStorage
- Drag & drop file upload interface

### Changed
- Enhanced progress tracking with multiple modes (estimate, chunks, segments)
- Improved UI with modern design and real-time updates
- Better error handling and validation
- Optimized transcription worker with cached audio support

### Fixed
- Status update issues in preparing phase
- Progress calculation accuracy
- Memory management for large audio files
- Cross-browser compatibility issues

## [1.0.0] - 2025-01-XX

### Added
- Initial release with basic Whisper transcription functionality
- Flask web interface for audio upload and transcription
- Support for multiple Whisper model sizes
- Management script for easy deployment
- Real-time progress tracking
- Live log streaming
- Download functionality for completed transcripts

### Features
- **Audio Support**: mp3, wav, m4a, mp4, aac, flac, ogg, wma, webm
- **Models**: tiny, base, small, medium, large, large-v2, large-v3
- **Decoding Options**: temperature, beam_size, best_of, language selection
- **Progress Modes**: heuristic estimation and chunk-based tracking
- **GPU Acceleration**: Automatic detection and optimized performance
- **Timestamped Output**: Both plain text and timestamped transcripts

---

**Author**: [Gopichand Busam](https://gopichand.me)
