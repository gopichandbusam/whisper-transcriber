# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 🔒 For Security Issues

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues privately by:

1. **Email**: Contact [Gopichand Busam](https://gopichand.me) directly
2. **Include**: 
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### 📋 What to Include

When reporting a security vulnerability, please include:

- **Vulnerability Description**: Clear explanation of the issue
- **Impact Assessment**: What could an attacker accomplish?
- **Reproduction Steps**: How to reproduce the vulnerability
- **Affected Components**: Which parts of the code are affected
- **Suggested Mitigation**: If you have ideas for fixes

### ⏱️ Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity and complexity

### 🛡️ Security Considerations

This application processes user-uploaded audio files. Key security areas:

- **File Upload Validation**: Ensuring only safe audio formats
- **Path Traversal**: Preventing malicious file paths
- **Resource Limits**: Preventing DoS through large files
- **Input Sanitization**: Validating all user inputs
- **Process Isolation**: Transcription runs in separate processes

### 🔧 General Security Best Practices

When deploying this application:

- Use HTTPS in production
- Implement rate limiting
- Set appropriate file size limits
- Use a reverse proxy (nginx, caddy)
- Regular security updates for dependencies
- Monitor resource usage
- Implement proper logging

## Acknowledgments

We appreciate responsible disclosure of security vulnerabilities and will acknowledge contributors (with their permission) in our security advisories.

---

**Contact**: [Gopichand Busam](https://gopichand.me)
