# VI-AMR Claude Code Context

## Project Overview
This repository implements **Adaptive Mesh Refinement (AMR) for Variational Inequalities (VIs)**. The main focus is targeted refinement near computed free boundaries with the ability to measure location errors.

## Key Components
- **Core class**: `VIAMR` in `viamr/viamr.py`
- **Main algorithms**: UDO (Unstructured Dilation Operator), VCD (Variable Coefficient Diffusion)
- **Dependencies**: Firedrake, animate, PETSc, netgen
- **Testing**: `pytest .` for serial tests, `mpiexec -n 3 python3 tests/test_parallel.py` for parallel

## Development Guidelines

### Code Standards
- Follow existing Firedrake/PETSc conventions
- Import structure: firedrake imports, then local viamr imports
- No underscores in filenames (prefer camelCase or single words)
- Use `VTKFile` for output, not `File`
- Output files should follow `result_*.pvd` naming convention
- All text files (.py, .md, etc.) should end with newlines
- No whitespace-only lines (blank lines should be actual newlines)
- Active set fields should use DG0 function space
- VI problems should have compatible boundary conditions: `psi ≤ g ≤ ub` on boundary
- Functions should be pure when possible (separate computation from I/O)
- Use double quotes for strings (not single quotes)
- Trailing commas in dictionaries and lists
- Two blank lines between top-level functions
- Long function calls split across multiple lines with proper indentation

### Proposed Improvements
- **UDO Performance**: Optimization with caching and vectorized operations for better scalability

### Testing Commands
**IMPORTANT: Must activate Firedrake virtual environment first:**
```bash
# Activate Firedrake venv (adjust path as needed)
source ~/venv-firedrake/bin/activate

# Serial tests
pytest .

# Parallel tests
cd tests/
mpiexec -n 3 python3 test_parallel.py

# Coverage report
pytest --cov-report html --cov=viamr tests/
```

### Known Limitations
1. **2D only**: SBR refinement limited to 2D (PETSc DMPlex limitation)
2. **Parallel issues**: #37 (adaptaveragedmetric), #38 (vcdmark) have serial/parallel differences
3. **Distribution parameters**: Required for parallel UDO marking
4. **Dependencies**: Complex installation chain (Firedrake → ngsPETSc → animate)

### Common Tasks
- **Run examples**: `cd examples/ && python3 sphere.py` or `python3 spiral.py`
- **View results**: Use Paraview to open `result_*.pvd` files
- **Build**: No specific build process, uses Python setuptools

### Architecture Notes
- Uses tag-and-refine mesh refinement strategy
- Supports both element marking (UDO/VCD) and metric-based approaches
- Integrates with PETSc DMPlex for mesh operations
- Supports both Netgen and Firedrake utility meshes

## File Structure Context
- `viamr/viamr.py`: Main VIAMR class (2000+ lines)
- `examples/`: Working examples (sphere, spiral, glacier applications)
- `tests/`: Basic and parallel test suites
- `paper/`: Research paper artifacts and numerical results
- `NumericalResults/`: Parameter exploration and load balancing studies
