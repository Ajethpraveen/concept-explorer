# 🌳 Concept explorer
A terminal-based tool that visually maps the connections between ideas by exploring diverse related concepts. Starting from a single root concept, it builds an expanding tree of interconnected ideas that span across different domains and intellectual territories.

## Features

- **Interactive visualization**: Watch in real-time as the concept web grows with a dynamic ASCII tree display
- **Contextual exploration**: Each new concept is generated with awareness of the full path that led to it
- **Domain diversity**: Encourages cross-disciplinary connections across philosophy, science, art, technology, and more
- **Customizable parameters**: Control the exploration depth, diversity bias, and model used
- **Exportable results**: Save the final concept web as a text file for later reference

## Requirements

- Python 3.6+
- [Ollama](https://ollama.ai/) running locally with models like Llama3, Qwen, etc.
- Required Python packages:
    - requests
    - networkx
    - colorama
    - shutil

## Installation

1. Clone the repo (or just copy the `explorer.py` code):

    ```bash
    git clone [https://gist.github.com/0be0dd20b0200fa4482e4fa795e40550.git](https://github.com/UdaraJay/concept-explorer.git)
    cd concept-explorer
    ```

2. Create and activate a virtual environment, then install the required dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install requests networkx colorama
    ```

3. You can now run `explorer.py` while the virtual environment is active:

    ```bash
    python explorer.py
    ```

3. Ensure Ollama is installed and running with at least one model:

```bash
# Install Ollama from https://ollama.ai/
# Pull a model
ollama pull llama3
```

## Usage

Run the concept explorer with default settings:

```bash
python explorer.py
```

Specify a different root concept:

```bash
python explorer.py "Time"
```

Customize multiple parameters:

```bash
python explorer.py --root="Emergence" --model="llama3" --diversity=0.9 --depth=100
```

### Command line options

- `--root=CONCEPT`: The starting concept (default: "Consciousness")
- `--model=MODEL`: The Ollama model to use (default: "llama3")
- `--diversity=FLOAT`: Diversity bias between 0.0-1.0 (default: 0.8)
- `--depth=INT`: Maximum exploration depth (default: 3)

## How it works

1. The tool starts with a root concept (e.g., "Consciousness")
2. For each concept, it queries a local LLM via Ollama to generate diverse related concepts
3. The LLM is given the full path context to generate meaningful connections
4. New concepts are added to the tree and visualized in real-time
5. The process continues until the maximum depth is reached
6. The final concept web is saved to a text file

## Examples

Starting with "Consciousness" might lead to branches like:

- Consciousness → Qualia → Dance as Embodied Knowledge → Cultural Memory
- Consciousness → Self-Awareness → Machine Sentience → Silicon Ethics
- Consciousness → Altered States → Synesthesia → Multimedia Art

## Terminal controls

- Press `Ctrl+C` to interrupt the exploration and save the current state
- Type `reset` if your terminal display becomes corrupted after the program exits

## Customization

You can modify the prompt template in the `get_related_concepts` method to adjust how the LLM generates connections.
