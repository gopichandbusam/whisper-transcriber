## Gemini Usage Guide

This document provides guidelines for using Google's Gemini models to assist with the development of the Whisper Transcriber application.

### Understanding the Codebase

To get an overview of the project structure and key files, you can ask Gemini:

> "Analyze the project structure and explain the purpose of `app.py`, `templates/index.html`, and `static/style.css`."

### Modifying and Improving Code

When you need to make changes, you can provide high-level instructions. For example, to refactor the transcription process:

> "Refactor the transcription logic in `app.py`. The new process should take the uploaded audio file, transcribe it using the Whisper model upon user request, and generate a downloadable text file with timestamps. Ensure the code is simplified and follows a clear, single-process flow."

### Adding New Features

To add a new feature, describe what you want to achieve. For instance, to add support for more output formats:

> "Extend the application to allow users to download the transcript in SRT or VTT format in addition to the existing TXT format."

### Best Practices

- **Be Specific:** Provide clear and detailed requests. Instead of "fix the bug," describe the bug's behavior and the expected outcome.
- **Provide Context:** Mention the relevant files or functions you want to modify.
- **Iterate:** Start with a high-level goal and refine it based on Gemini's responses and suggestions.
