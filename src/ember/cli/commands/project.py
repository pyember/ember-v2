"""Project command implementation."""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from ember.core.utils.output import print_header, print_info, print_error, print_success
from ember.core.utils.verbosity import get_verbosity


def register(subparsers) -> argparse.ArgumentParser:
    """Add project command to subparsers."""
    parser = subparsers.add_parser(
        "project",
        help="Create and manage Ember projects",
        description="Commands for creating new projects and managing project structure"
    )
    
    subcommands = parser.add_subparsers(dest="subcommand", help="Project subcommands")
    
    # Init command
    init_parser = subcommands.add_parser("init", help="Initialize a new Ember project")
    init_parser.add_argument(
        "name",
        nargs="?",
        default=".",
        help="Project name (default: current directory)"
    )
    init_parser.add_argument(
        "--template",
        choices=["basic", "research", "production"],
        default="basic",
        help="Project template to use"
    )
    init_parser.add_argument(
        "--no-git",
        action="store_true",
        help="Don't initialize git repository"
    )
    
    # Structure command
    structure_parser = subcommands.add_parser(
        "structure",
        help="Show recommended project structure"
    )
    
    parser.set_defaults(func=execute)
    return parser


def execute(args: argparse.Namespace) -> int:
    """Execute project command."""
    if not args.subcommand:
        print_error("No subcommand specified. Use 'ember project --help' for usage.")
        return 1
    
    if args.subcommand == "init":
        return init_project(args)
    elif args.subcommand == "structure":
        return show_structure(args)
    else:
        print_error(f"Unknown subcommand '{args.subcommand}'")
        return 1


def init_project(args: argparse.Namespace) -> int:
    """Initialize a new Ember project."""
    # Determine project path
    if args.name == ".":
        project_path = Path.cwd()
        project_name = project_path.name
    else:
        project_path = Path(args.name)
        project_name = args.name
    
    print_header(f"Initializing Ember project: {project_name}")
    
    try:
        # Create project directory if it doesn't exist
        if args.name != "." and not project_path.exists():
            project_path.mkdir(parents=True)
            print_success(f"Created directory: {project_path}")
        
        # Create project structure based on template
        if args.template == "basic":
            create_basic_structure(project_path)
        elif args.template == "research":
            create_research_structure(project_path)
        elif args.template == "production":
            create_production_structure(project_path)
        
        # Initialize git repository
        if not args.no_git:
            os.system(f"cd {project_path} && git init > /dev/null 2>&1")
            print_success("Initialized git repository")
        
        print_success(f"Project '{project_name}' initialized successfully!")
        print_info(f"Next steps:")
        print(f"  1. cd {project_path}")
        print(f"  2. pip install ember-ai")
        print(f"  3. ember model list")
        
        return 0
        
    except Exception as e:
        print_error(f"Error initializing project: {e}")
        return 1


def create_basic_structure(project_path: Path) -> None:
    """Create basic project structure."""
    # Create directories
    dirs = [
        "src",
        "data",
        "outputs",
        "notebooks"]
    
    for dir_name in dirs:
        dir_path = project_path / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            if get_verbosity() >= 1:
                print(f"  Created {dir_name}/")
    
    # Create files
    files = {
        "README.md": f"""# {project_path.name}

An Ember AI project.

## Setup

```bash
pip install ember-ai
```

## Usage

```python
from ember.api import models

# List available models
print(models.available())

# Use a model
model = models("gpt-4")
response = model("Hello, world!")
print(response)
```
""",
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Ember
outputs/
.ember/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
""",
        "src/__init__.py": "",
        "src/main.py": """\"\"\"Main entry point for the project.\"\"\"

from ember.api import models


def main():
    \"\"\"Run the main application.\"\"\"
    # Example: List available models
    print("Available models:")
    for model in models.available():
        print(f"  - {model}")


if __name__ == "__main__":
    main()
""",
    }
    
    for file_path, content in files.items():
        full_path = project_path / file_path
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            if get_verbosity() >= 1:
                print(f"  Created {file_path}")


def create_research_structure(project_path: Path) -> None:
    """Create research project structure."""
    # Start with basic structure
    create_basic_structure(project_path)
    
    # Add research-specific directories
    dirs = [
        "experiments",
        "results",
        "configs",
        "papers"]
    
    for dir_name in dirs:
        dir_path = project_path / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            if get_verbosity() >= 1:
                print(f"  Created {dir_name}/")
    
    # Add research-specific files
    files = {
        "experiments/example_experiment.py": """\"\"\"Example experiment script.\"\"\"

from ember.api import models, datasets, evaluators
from ember.core.utils.output import print_header, print_table


def run_experiment():
    \"\"\"Run an example experiment.\"\"\"
    print_header("Example Experiment")
    
    # Configure experiment
    model_name = "gpt-3.5-turbo"
    dataset_name = "mmlu"
    
    # Load components
    model = models(model_name)
    dataset = datasets(dataset_name)().take(10)  # Small sample
    evaluator = evaluators("accuracy")
    
    # Run evaluation
    results = []
    for example in dataset:
        response = model(example["prompt"])
        score = evaluator(response, example["answer"])
        results.append({
            "prompt": example["prompt"][:50] + "...",
            "score": score
        })
    
    # Display results
    print_table(results)


if __name__ == "__main__":
    run_experiment()
""",
        "configs/default.yaml": """# Default experiment configuration

model:
  name: gpt-3.5-turbo
  temperature: 0.7
  max_tokens: 1000

dataset:
  name: mmlu
  subset: abstract_algebra
  limit: 100

evaluation:
  metrics:
    - accuracy
    - f1_score
  
output:
  save_predictions: true
  results_dir: results/
""",
    }
    
    for file_path, content in files.items():
        full_path = project_path / file_path
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            if get_verbosity() >= 1:
                print(f"  Created {file_path}")


def create_production_structure(project_path: Path) -> None:
    """Create production project structure."""
    # Start with basic structure
    create_basic_structure(project_path)
    
    # Add production-specific directories
    dirs = [
        "src/api",
        "src/models",
        "src/services",
        "tests",
        "docker",
        "deploy"]
    
    for dir_name in dirs:
        dir_path = project_path / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            if get_verbosity() >= 1:
                print(f"  Created {dir_name}/")
    
    # Add production-specific files
    files = {
        "requirements.txt": """ember-ai>=0.1.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pytest>=7.0.0
""",
        "Dockerfile": """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
        "src/api/__init__.py": "",
        "src/api/main.py": """\"\"\"FastAPI application for serving Ember models.\"\"\"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from ember.api import models


app = FastAPI(title="Ember API", version="0.1.0")


class InvokeRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


class InvokeResponse(BaseModel):
    model: str
    response: str


@app.get("/models")
async def list_models():
    \"\"\"List available models.\"\"\"
    return {"models": models.available()}


@app.post("/invoke", response_model=InvokeResponse)
async def invoke_model(request: InvokeRequest):
    \"\"\"Invoke a model with a prompt.\"\"\"
    try:
        model = models(request.model)
        kwargs = {"temperature": request.temperature}
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        
        response = model(request.prompt, **kwargs)
        
        return InvokeResponse(
            model=request.model,
            response=response
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    \"\"\"Health check endpoint.\"\"\"
    return {"status": "healthy"}
""",
        "tests/test_api.py": """\"\"\"API tests.\"\"\"

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_health_check():
    \"\"\"Test health check endpoint.\"\"\"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_list_models():
    \"\"\"Test model listing.\"\"\"
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert isinstance(response.json()["models"], list)
""",
    }
    
    for file_path, content in files.items():
        full_path = project_path / file_path
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            if get_verbosity() >= 1:
                print(f"  Created {file_path}")


def show_structure(args: argparse.Namespace) -> int:
    """Show recommended project structure."""
    print_header("Recommended Ember Project Structure")
    
    structure = """
project/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
│
├── src/                  # Source code
│   ├── __init__.py
│   ├── main.py          # Main entry point
│   ├── models/          # Model definitions
│   ├── operators/       # Custom operators
│   └── utils/           # Utility functions
│
├── data/                 # Data directory
│   ├── raw/             # Raw data
│   ├── processed/       # Processed data
│   └── cache/           # Cached data
│
├── configs/              # Configuration files
│   ├── default.yaml     # Default config
│   └── experiments/     # Experiment configs
│
├── notebooks/            # Jupyter notebooks
│   ├── exploration/     # Data exploration
│   └── analysis/        # Results analysis
│
├── experiments/          # Experiment scripts
│   ├── baseline.py      # Baseline experiments
│   └── ablations/       # Ablation studies
│
├── results/              # Experiment results
│   ├── metrics/         # Evaluation metrics
│   └── predictions/     # Model predictions
│
├── tests/                # Test files
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
│
└── outputs/              # Generated outputs
    ├── logs/            # Log files
    └── models/          # Saved models
"""
    
    print(structure)
    
    print_info("Tips:")
    print("  - Keep experiments reproducible with config files")
    print("  - Use notebooks for exploration, scripts for production")
    print("  - Version control data references, not data itself")
    print("  - Write tests for custom operators and utilities")
    
    return 0