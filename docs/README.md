# Project Documentation

This folder contains comprehensive documentation for the Blockchain P2P Network project.

## Contents

- `architecture.md` - System architecture and component interactions
- `api/` - API documentation for each module
- `user_guides/` - End-user documentation for installation and usage
- `development/` - Developer guidelines and contribution information

## Generating Documentation

To generate HTML documentation from these sources:

1. Install Sphinx:
   ```
   pip install sphinx sphinx-rtd-theme
   ```

2. Build the documentation:
   ```
   cd docs
   make html
   ```

3. View documentation:
   ```
   open _build/html/index.html
   ```

