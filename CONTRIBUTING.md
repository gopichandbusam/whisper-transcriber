# Contributing to Whisper Transcriber

Thank you for your interest in contributing to Whisper Transcriber! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/yourusername/whisper-transcriber.git
   cd whisper-transcriber
   ```
3. **Set up** the development environment:
   ```bash
   ./manage.sh setup
   ```

## 💻 Development Workflow

### Setting Up Your Development Environment

```bash
# Create and activate virtual environment
./manage.sh shell

# Install dependencies in development mode
pip install -e .

# Run the application in debug mode
FLASK_DEBUG=1 python app.py
```

### Code Style and Standards

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and small
- Add type hints where appropriate

### Testing Your Changes

Before submitting a pull request:

1. **Test the application manually**:
   ```bash
   ./manage.sh run
   # Test upload, transcription, and download functionality
   ```

2. **Check for syntax errors**:
   ```bash
   python -m py_compile app.py
   ```

3. **Test with different audio formats** (mp3, wav, m4a, etc.)
4. **Test with different Whisper models** (tiny, base, small, etc.)

## 📝 Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, descriptive commits:
   ```bash
   git commit -m "Add feature: describe what you added"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Screenshots if UI changes were made
   - Reference any related issues

### Pull Request Guidelines

- **One feature per PR**: Keep changes focused and atomic
- **Update documentation**: Update README.md if needed
- **Test thoroughly**: Ensure your changes don't break existing functionality
- **Follow existing patterns**: Match the coding style of the existing codebase

## 🐛 Reporting Issues

When reporting bugs or requesting features:

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** if available
3. **Provide detailed information**:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages or logs
   - Audio file details (format, duration, size)

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Upload file '...'
2. Select model '...'
3. Click on '...'
4. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. macOS 12.0, Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Browser: [e.g. Chrome 95, Safari 15]
- Audio file: [format, duration, size]

**Additional context**
Any other context about the problem.
```

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### 🔧 Features
- New audio format support
- Additional transcription options
- Performance optimizations
- UI/UX improvements
- Mobile responsiveness
- Batch processing capabilities

### 📚 Documentation
- Code comments and docstrings
- Tutorial videos or blog posts
- Translation to other languages
- API documentation

### 🧪 Testing
- Unit tests
- Integration tests
- Performance benchmarks
- Cross-platform testing

### 🐛 Bug Fixes
- Error handling improvements
- Edge case handling
- Memory leak fixes
- Cross-browser compatibility

## 🤔 Questions?

If you have questions about contributing:

1. **Check the README** for basic setup and usage
2. **Search existing issues** for similar questions
3. **Open a new issue** with the "question" label
4. **Contact the maintainer**: [Gopichand Busam](https://gopichand.me)

## 📜 Code of Conduct

This project follows a simple code of conduct:

- **Be respectful** and inclusive
- **Be collaborative** and helpful
- **Be patient** with newcomers
- **Focus on constructive feedback**
- **Respect different viewpoints** and experiences

## 🙏 Recognition

Contributors will be recognized in:
- GitHub contributors page
- Release notes for significant contributions
- README acknowledgments section

Thank you for contributing to Whisper Transcriber! 🎉

---

**Questions?** Reach out to [Gopichand Busam](https://gopichand.me)
