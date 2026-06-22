```markdown
# ppocr_pdf Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you how to contribute to the `ppocr_pdf` Python codebase, which focuses on OCR processing for PDFs. You'll learn the project's coding conventions, commit patterns, and the structured workflows used for adding features, fixing bugs, planning designs, refactoring, updating tests, and managing dependencies. The repository emphasizes clarity, modularity, and maintainability, using conventional commits and well-defined development processes.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for all Python files and modules.  
  *Example:*  
  ```
  ocr_pdf.py
  download_models.py
  device_utils.py
  ```

- **Import Style:**  
  Use **relative imports** within the package.  
  *Example:*  
  ```python
  from .utils import download_file
  from .routers import output_format
  ```

- **Export Style:**  
  Use **named exports** (explicit function/class definitions).  
  *Example:*  
  ```python
  def run_ocr(...):
      ...

  class PDFProcessor:
      ...
  ```

- **Commit Messages:**  
  Follow the **conventional commit** format with these prefixes:  
  `fix`, `chore`, `docs`, `feat`, `test`, `refactor`  
  *Example:*  
  ```
  feat: add support for new OCR model in ocr_pdf.py
  fix: handle empty PDF pages in api.py
  ```

## Workflows

### Feature Development with Tests and Docs
**Trigger:** When adding a significant new feature or capability (e.g., new model support, output format, or endpoint).  
**Command:** `/new-feature`

1. Update or extend core implementation files (e.g., `ocr_pdf.py`, `api.py`, `download_models.py`).
2. Add or update CLI/API parameters as needed.
3. Add or update test files in `tests/` to cover the new feature.
4. Update documentation files (`README.md`, `Docker.md`, or design/plans in `thoughts/`).
5. Optionally, update `.gitignore` or `requirements.txt` if new outputs or dependencies are introduced.

*Example:*
```python
# ocr_pdf.py
def run_ocr_with_new_model(...):
    ...
```
```markdown
# README.md
## New Model Support
Describe usage and parameters.
```

---

### Bugfix or Hotfix Across Core Logic
**Trigger:** When fixing a bug or regression in core functionality.  
**Command:** `/fix-bug`

1. Edit core implementation files to fix the bug (e.g., `ocr_pdf.py`, `api.py`, `download_models.py`, `device_utils.py`).
2. Optionally update or add tests to verify the fix.
3. Optionally update documentation or add notes to design/ledger files.

*Example:*
```python
# api.py
def process_pdf(...):
    if not pages:
        raise ValueError("No pages found in PDF")
```

---

### Design and Implementation Planning
**Trigger:** When planning a significant refactor, feature, or architectural change.  
**Command:** `/new-design`

1. Create or update design documents in `thoughts/shared/designs/`.
2. Create or update implementation plans in `thoughts/shared/plans/`.
3. Optionally, validate design with baseline tests or notes.

*Example:*
```markdown
# thoughts/shared/designs/new_architecture.md
## Motivation
...
## Proposed Changes
...
```

---

### Refactor with Module Extraction
**Trigger:** When modularizing or cleaning up code, often as part of a feature or to enable future changes.  
**Command:** `/refactor-module`

1. Extract logic from an existing file (e.g., `api.py`) into a new module (e.g., `routers/output_format.py`).
2. Update imports and references in the original and related files.
3. Add or update constants/shared utilities as needed.
4. Maintain backward compatibility with try/except or fallback logic.

*Example:*
```python
# routers/output_format.py
def format_output(...):
    ...

# api.py
from .routers import output_format
```

---

### Test Suite Update or Migration
**Trigger:** When core logic or supported models change, requiring test coverage updates.  
**Command:** `/update-tests`

1. Update, rewrite, or remove outdated test files in `tests/`.
2. Add new test files for new features or models.
3. Update test fixtures or mocks to match new output structures.

*Example:*
```python
# tests/ocr_pdf_test.py
def test_new_model_support():
    ...
```

---

### Dependency or Infrastructure Update
**Trigger:** When a new dependency is needed, or when infrastructure/configuration changes are required.  
**Command:** `/add-dependency`

1. Add or update dependencies in `requirements.txt`.
2. Add or update configuration files (e.g., `pytest.ini`, `.gitignore`).
3. Optionally, update documentation to reflect the new dependency or config.

*Example:*
```
# requirements.txt
paddlepaddle>=2.0.0
```
```
# .gitignore
*.pdf
```

---

## Testing Patterns

- **Test Framework:**  
  Not explicitly specified; use standard Python testing frameworks (e.g., `pytest` or `unittest`).

- **Test File Naming:**  
  All test files use the pattern `*_test.py` and are located in the `tests/` directory.

- **Test Coverage:**  
  Tests are updated or added with every significant feature, bugfix, or refactor.

*Example:*
```python
# tests/download_models_test.py
def test_download_success():
    ...
```

## Commands

| Command         | Purpose                                                      |
|-----------------|--------------------------------------------------------------|
| /new-feature    | Start a new feature with tests and documentation             |
| /fix-bug        | Apply a bugfix or hotfix across core logic                   |
| /new-design     | Add or update design documents and implementation plans       |
| /refactor-module| Refactor code by extracting logic into new modules           |
| /update-tests   | Update or migrate test suite for new/refactored functionality|
| /add-dependency | Add or update dependencies or infrastructure/configuration    |
```
