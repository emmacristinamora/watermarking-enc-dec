# Hard Requirements from the Professor
The code should be:
- clearly structured into logical modules;
- easy to understand and reproduce;
- documented with meaningful comments where necessary;
- consistent with the experiments and claims reported in the paper.
3.2. Code Checklist
□ Include a README.md with setup instructions and a short project description.
□ Separate data processing, model definition, training, and evaluation.
□ Use clear naming conventions and maintain a clean folder structure.
□ Comment non-trivial parts of the code.
□ Store key plots, qualitative results, and tables in a dedicated results folder.
□ Ensure that the experiments described in the report can be reproduced from the repository.

- **coding style prompt**
    
    ## Code Style & Architecture Guide
    
    ### Project Structure
    
    - Standard layout: `config/`, `data/`, `src/` (or scripts at root), plus a `.gitignore` and `requirements.txt` — no conda envs, no setup.py unless packaging is needed
    - Dependencies are always pinned in `requirements.txt`; the environment is fully reproducible from it
    - Logics are decoupled: data generation, training, evaluation, and orchestration each live in separate scripts — no monolithic files
    
    ### Per-Script Structure
    
    - First line of every script: `# path/script_name.py` (relative to repo root)
    - Scripts are divided into named sections with two blank lines before each header:Typical order: `IMPORTS` → `CONFIG` → `HELPERS` → `MAIN`
        
        `# === SECTION NAME ===`
        
    - Every script ends with `if __name__ == "__main__": main()`
    
    ### Comments & Docstrings
    
    - Inline comments are lowercase, short, and placed only where they add real value — not narrating obvious code
        
        `# skip empty transcripts`
        
    - Long or non-trivial functions get a structured docstring:Simple utility functions don't need one
        
        `"""
        Summary sentence.
        Args:
        Returns:
        Logic:
        """`
        
    
    ### CLI & Configuration
    
    - Every script is fully driven by CLI arguments via `argparse`, with a dedicated `parse_args()` function — no hardcoded paths or parameters inside function bodies
    - Static specs (prompts, personas, attributes) live in YAML files under `config/`; runtime parameters come from CLI
    - Dataclasses (`@dataclass`) are used for structured config objects when grouping related parameters
    
    ### Data & I/O
    
    - JSONL for datasets, logs, and per-example results; JSON for summaries and config snapshots
    - Paths are always `pathlib.Path` objects, never raw strings
    - IO helpers (`load_jsonl`, `save_json`, `append_jsonl`) are defined explicitly and kept consistent
    
    ### Code Style
    
    - Type hints on all function signatures, including return types
    - Descriptive names over abbreviations — the name should make the purpose obvious without needing a comment
    - Constants and defaults belong in argparse defaults or dataclass fields, not buried in logic
    - Validation is early and loud: raise `ValueError` / `FileNotFoundError` with clear messages as soon as bad input is detected — no silent failures
    
    ### What We Don't Do
    
    - No Jupyter notebooks — everything is a standalone runnable script
    - No broad `try/except` for normal control flow
    - No speculative abstractions — don't build utilities for hypothetical reuse; factor things out only when there's actual repetition