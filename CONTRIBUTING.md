# Contributing to DERL

We welcome contributions to the DERL project! This document provides guidelines for contributing.

## How to Contribute

1. **Fork the Repository**
   - Fork the project on GitHub
   - Clone your fork locally

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clear, commented code
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   python -m unittest discover tests/
   ```

5. **Commit Your Changes**
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Describe your changes

## Code Style

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

- Add unit tests for new features
- Ensure all tests pass before submitting PR
- Test on multiple environments when possible

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions and classes
- Update examples if changing API

## Reporting Issues

- Use GitHub Issues to report bugs
- Include minimal reproducible example
- Specify Python version and dependencies
- Describe expected vs actual behavior

## Feature Requests

- Open a GitHub Issue with "Feature Request" label
- Describe the feature and its use case
- Discuss implementation approach

Thank you for contributing to DERL!
