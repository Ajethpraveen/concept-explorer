import argparse
import json
import os
import random
import shutil
import sys
import textwrap
import time
from collections import deque

import networkx as nx
import requests
from colorama import Back, Fore, Style, init

# Initialize colorama for colored terminal output.
init(autoreset=True)

# Default configuration constants
DEFAULT_MODEL = "llama3"
DEFAULT_ROOT_CONCEPT = "Consciousness"
DEFAULT_DIVERSITY_BIAS = 0.8
DEFAULT_MAX_DEPTH = 3
SLEEP_DURATION = 0.5  # seconds

def clear_terminal():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

class ConceptExplorer:
    def __init__(self, model=DEFAULT_MODEL):
        self.graph = nx.DiGraph()
        self.seen_concepts = set()
        self.last_added = None
        self.current_concept = None
        self.model = model
        self.term_width, self.term_height = shutil.get_terminal_size((80, 24))
        self.last_tree_update_time = 0  # For throttling live updates
        self.MIN_UPDATE_INTERVAL = 0.5  # Minimum time between live tree updates (in seconds)

    def get_available_models(self):
        """Fetch available models from Ollama."""
        url = "http://localhost:11434/api/tags"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except Exception as e:
            print(f"{Fore.RED}Error connecting to Ollama: {str(e)}{Style.RESET_ALL}")
            return []

    def check_model_availability(self):
        """Check if the current model is available. If not found exactly, try a prefix match."""
        available_models = self.get_available_models()
        if not available_models:
            return False

        if self.model in available_models:
            return True

        for model_name in available_models:
            if model_name.startswith(f"{self.model}:"):
                self.model = model_name
                return True
        return False

    def _update_thinking_block(self, text, state):
        """
        Update the live 'thinking' display.
        Parameters:
          text (str): The current accumulated thinking text.
          state (dict): Holds tracking variables such as 'printed_lines', 
                        'printed_brain_emoji' and 'last_printed_block'.
        """
        wrapped = textwrap.wrap(text.strip(), width=self.term_width)
        if not state["printed_brain_emoji"]:
            wrapped.insert(0, "🧠")
            state["printed_brain_emoji"] = True
        if len(wrapped) > 6:
            wrapped = wrapped[:6]
            wrapped[-1] = wrapped[-1] + "..."
        if wrapped == state["last_printed_block"]:
            return
        state["last_printed_block"] = wrapped.copy()
        for _ in range(state["printed_lines"]):
            sys.stdout.write("\033[F\033[K")
        sys.stdout.flush()
        for line_out in wrapped:
            print(f"{Fore.LIGHTBLACK_EX}{line_out}{Style.RESET_ALL}")
        state["printed_lines"] = len(wrapped)

    def query_ollama_stream(self, prompt):
        """
        Streams a response from Ollama in real-time using a streaming POST request.
        Only live <think> blocks are shown. The final response is JSON-parsed internally.
        """
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {"model": self.model, "prompt": prompt, "stream": True}

        if not self.check_model_availability():
            print(f"{Fore.RED}Error: Model '{self.model}' is not available in Ollama.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please pull the model first with: 'ollama pull {self.model}'{Style.RESET_ALL}")
            return "[]"
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()

            full_response = ""
            in_think_mode = False
            think_buffer = ""
            state = {
                "printed_lines": 0,
                "printed_brain_emoji": False,
                "last_printed_block": []
            }
            for line in response.iter_lines():
                # Allow keyboard interrupts inside the streaming loop.
                try:
                    if not line:
                        continue
                    try:
                        data_chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    chunk = data_chunk.get("response", "")
                    full_response += chunk

                    idx = 0
                    while idx < len(chunk):
                        if not in_think_mode:
                            start_tag_index = chunk.find("<think>", idx)
                            if start_tag_index == -1:
                                break
                            idx = start_tag_index + len("<think>")
                            in_think_mode = True
                            state["printed_brain_emoji"] = False
                            state["last_printed_block"] = []
                        else:
                            end_tag_index = chunk.find("</think>", idx)
                            if end_tag_index == -1:
                                think_buffer += chunk[idx:]
                                self._update_thinking_block(think_buffer, state)
                                break
                            else:
                                think_buffer += chunk[idx:end_tag_index]
                                self._update_thinking_block(think_buffer, state)
                                in_think_mode = False
                                think_buffer = ""
                                idx = end_tag_index + len("</think>")
                    if data_chunk.get("done", False):
                        break
                except KeyboardInterrupt:
                    raise
            clean_response = self.strip_thinking_tags(full_response)
            return clean_response
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"{Fore.RED}Error streaming from Ollama: {str(e)}{Style.RESET_ALL}")
            return "[]"

    def strip_thinking_tags(self, response):
        """Remove all <think> tags from the final response."""
        return response.replace("<think>", "").replace("</think>", "")

    def get_related_concepts(self, concept, depth=0, path=None):
        """
        Query Ollama for related concepts given a concept and an exploration path.
        Returns a filtered list of unique, concise related concept strings.
        """
        if concept in self.seen_concepts or depth > 5:
            return []

        self.seen_concepts.add(concept)
        self.current_concept = concept
        full_path = (path or []) + [concept]

        prompt = textwrap.dedent(f"""
            Starting with the concept: "{concept}", generate 4-5 fascinating and unexpected related concepts.

            Context: We're building a concept web and have followed this path to get here:
            {' → '.join(full_path)}

            Guidelines:
            1. Seek maximum intellectual diversity - span across domains like science, art, philosophy, technology, culture, etc.
            2. Each concept should be expressed in 1-5 words (shorter is better).
            3. Avoid obvious associations - prefer surprising or thought-provoking connections.
            4. Consider how your suggested concepts relate to BOTH:
               - The immediate parent concept "{concept}"
               - The overall path context: {' → '.join(full_path)}
            5. Consider these different types of relationships:
               - Metaphorical parallels
               - Contrasting opposites
               - Historical connections
               - Philosophical implications
               - Cross-disciplinary applications

            Avoid any concepts already in the path. Be creative but maintain meaningful connections.

            You may use <think> tags to show your reasoning process.

            Return ONLY a JSON array of strings, with no explanation or additional text.
            Example: ["Related concept 1", "Related concept 2", "Related concept 3", "Related concept 4"]
        """).strip()

        print(f"\n{Fore.CYAN}🔍 Exploring concepts related to: {Fore.YELLOW}{concept}{Style.RESET_ALL}")
        if path:
            print(f"{Fore.CYAN}📜 Path context: {Fore.YELLOW}{' → '.join(path)} → {concept}{Style.RESET_ALL}")

        response = self.query_ollama_stream(prompt)
        try:
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                related_concepts = json.loads(json_str)

                filtered_concepts = []
                for rc in related_concepts:
                    if len(rc) > self.term_width // 3:
                        rc = rc[:self.term_width // 3 - 3] + "..."
                    if not rc.strip() or rc.lower() in (c.lower() for c in self.seen_concepts):
                        print(f"{Fore.RED}✗ Rejected concept: {rc}{Style.RESET_ALL}")
                    else:
                        filtered_concepts.append(rc)
                print(f"{Fore.GREEN}✓ Found {len(filtered_concepts)} valid related concepts{Style.RESET_ALL}")
                return filtered_concepts
            else:
                print(f"{Fore.RED}✗ No valid JSON found in response{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Response: {response}{Style.RESET_ALL}")
                return []
        except Exception as e:
            print(f"{Fore.RED}✗ Error parsing response: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Response: {response}{Style.RESET_ALL}")
            return []

    def _color_node(self, node, prefix, is_last, current_depth):
        """
        Color the node based on its role:
          - Current concept: special background
          - Last added: another color
          - Root and deeper nodes: different hues
        """
        connector = "└── " if is_last else "├── "
        available_width = self.term_width - len(prefix) - len(connector) - 5
        if len(node) > available_width:
            node = node[:available_width-3] + "..."
        if node == self.current_concept:
            return f"{prefix}{Fore.CYAN}{connector}{Back.BLUE}{Fore.WHITE}{node}{Style.RESET_ALL}"
        elif node == self.last_added:
            return f"{prefix}{Fore.CYAN}{connector}{Back.GREEN}{Fore.BLACK}{node}{Style.RESET_ALL}"
        elif current_depth == 0:
            return f"{prefix}{Fore.CYAN}{connector}{Fore.MAGENTA}{Style.BRIGHT}{node}{Style.RESET_ALL}"
        else:
            colors = [Fore.YELLOW, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.RED, Fore.WHITE]
            color = colors[min(current_depth, len(colors)-1)]
            return f"{prefix}{Fore.CYAN}{connector}{color}{node}{Style.RESET_ALL}"

    def update_live_tree(self, focus_node=None, max_display_depth=None):
        """
        Clear the screen and print the current ASCII representation of the concept web.
        This method is throttled so it doesn't refresh too frequently.
        """
        current_time = time.time()
        if current_time - self.last_tree_update_time < self.MIN_UPDATE_INTERVAL:
            return  # Skip update if called too soon.
        self.last_tree_update_time = current_time

        self.term_width, self.term_height = shutil.get_terminal_size((80, 24))
        clear_terminal()

        header = [
            f"{Fore.GREEN}🌳 {Fore.YELLOW}CONCEPT {Fore.GREEN}EXPLORER {Fore.GREEN}🌳",
            f"{Fore.CYAN}{'═' * min(50, self.term_width - 2)}",
            ""
        ]
        for line in header:
            print(line)

        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        if not roots:
            print(f"{Fore.RED}No root nodes found yet{Style.RESET_ALL}")
            return

        path_to_highlight = []
        if focus_node:
            current = focus_node
            while current:
                path_to_highlight.append(current)
                predecessors = list(self.graph.predecessors(current))
                current = predecessors[0] if predecessors else None

        available_height = self.term_height - 10
        if focus_node:
            path_depth = len(path_to_highlight)
            if max_display_depth is None or max_display_depth < path_depth:
                max_display_depth = path_depth + 1
        else:
            if max_display_depth is None:
                max_display_depth = max(2, min(5, available_height // 3))

        tree_text = self._generate_ascii_tree(
            roots[0],
            focus_paths=path_to_highlight,
            max_depth=max_display_depth,
            available_height=available_height
        )
        print(tree_text)

        print(f"\n{Fore.CYAN}{'═' * min(50, self.term_width - 2)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 Concepts: {len(self.graph.nodes)} | Connections: {len(self.graph.edges)}{Style.RESET_ALL}")
        if self.current_concept:
            current_display = self.current_concept
            if len(current_display) > self.term_width - 25:
                current_display = current_display[:self.term_width - 28] + "..."
            print(f"{Fore.CYAN}🔍 Exploring: {Fore.YELLOW}{current_display}{Style.RESET_ALL}")

    def _generate_ascii_tree(self, node, prefix="", is_last=True, visited=None, focus_paths=None,
                             max_depth=None, current_depth=0, available_height=24, lines_used=0):
        """
        Recursively generate an ASCII tree representation of the concept graph.
        Stops when reaching the available height or maximum allowed depth.
        """
        if visited is None:
            visited = set()
        if focus_paths is None:
            focus_paths = []

        if lines_used >= available_height:
            return f"{prefix}{Fore.CYAN}{'└── ' if is_last else '├── '}{Fore.RED}(...more...){Style.RESET_ALL}\n"
        if node in visited or (max_depth is not None and current_depth > max_depth):
            return f"{self._color_node(node, prefix, is_last, current_depth)} {Fore.RED}(...){Style.RESET_ALL}\n"
        visited.add(node)

        result = f"{self._color_node(node, prefix, is_last, current_depth)}\n"
        lines_used += 1
        children = list(self.graph.successors(node))
        if not children or lines_used >= available_height:
            return result
        next_prefix = prefix + ("    " if is_last else "│   ")
        if focus_paths:
            children.sort(key=lambda x: x not in focus_paths)
        remaining_height = available_height - lines_used
        has_more = False
        if len(children) > remaining_height:
            focus_children = [c for c in children if c in focus_paths]
            non_focus = [c for c in children if c not in focus_paths]
            sample_size = max(0, remaining_height - len(focus_children) - 1)
            if sample_size:
                if len(non_focus) <= sample_size:
                    sampled = non_focus
                else:
                    third = max(1, sample_size // 3)
                    sampled = (non_focus[:third] +
                               non_focus[len(non_focus)//2 - third//2: len(non_focus)//2 + third//2] +
                               non_focus[-third:])
                    sampled = list(dict.fromkeys(sampled))[:sample_size]
            else:
                sampled = []
            children = focus_children + sampled
            has_more = (len(focus_children) + len(non_focus)) > len(children)
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1 and not has_more)
            child_tree = self._generate_ascii_tree(
                child,
                next_prefix,
                is_last_child,
                visited.copy(),
                focus_paths,
                max_depth,
                current_depth + 1,
                available_height,
                lines_used
            )
            result += child_tree
            lines_used += child_tree.count('\n')
            if lines_used >= available_height:
                break
        if has_more and lines_used < available_height:
            result += f"{next_prefix}{Fore.CYAN}└── {Fore.RED}(...more nodes...){Style.RESET_ALL}\n"
        return result

    def build_concept_web(self, root_concept, max_depth=DEFAULT_MAX_DEPTH, diversity_bias=DEFAULT_DIVERSITY_BIAS):
        """Build the concept web starting from the root concept using breadth‑first search."""
        self.graph.add_node(root_concept)
        self.update_live_tree()
        queue = deque([(root_concept, 0, [])])  # (concept, depth, path)

        try:
            while queue:
                # Check frequently for keyboard interrupt.
                concept, depth, path = queue.popleft()
                if depth >= max_depth:
                    continue
                display_depth = min(3, max_depth)
                self.update_live_tree(focus_node=concept, max_display_depth=display_depth)
                related_concepts = self.get_related_concepts(concept, depth, path)
                if diversity_bias > 0 and related_concepts and random.random() < diversity_bias:
                    related_concepts.sort(key=lambda x: self._diversity_score(x, self.seen_concepts))
                for rel_concept in related_concepts:
                    if rel_concept not in self.graph:
                        self.graph.add_node(rel_concept)
                        self.last_added = rel_concept
                    self.graph.add_edge(concept, rel_concept)
                    new_path = path + [concept]
                    queue.append((rel_concept, depth + 1, new_path))
                    self.update_live_tree(focus_node=rel_concept, max_display_depth=display_depth)
                    time.sleep(SLEEP_DURATION)
                time.sleep(SLEEP_DURATION)
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Exploration interrupted by user.{Style.RESET_ALL}")
            return  # Exit the exploration loop, keeping the partial graph.

        self.current_concept = None
        self.last_added = None
        self.update_live_tree()
        print(f"\n{Fore.GREEN}🎉 Concept exploration complete!{Style.RESET_ALL}")

    def _diversity_score(self, concept, existing_concepts):
        """Compute a diversity score based on common words between concepts."""
        score = 0
        for existing in existing_concepts:
            shared_words = set(concept.lower().split()) & set(existing.lower().split())
            if not shared_words:
                score += 1
        return score

    def export_ascii_tree(self, output_file="concept_web.txt"):
        """Export a plain ASCII tree of the concept graph to a text file."""
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        if not roots:
            print(f"{Fore.RED}No root nodes found{Style.RESET_ALL}")
            return

        def _plain_ascii_tree(node, prefix="", is_last=True, visited=None):
            if visited is None:
                visited = set()
            if node in visited:
                return f"{prefix}{'└── ' if is_last else '├── '}{node} (...)\n"
            visited.add(node)
            result = f"{prefix}{'└── ' if is_last else '├── '}{node}\n"
            children = list(self.graph.successors(node))
            if not children:
                return result
            next_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                result += _plain_ascii_tree(child, next_prefix, is_last_child, visited.copy())
            return result

        tree_text = _plain_ascii_tree(roots[0])
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(tree_text)
        print(f"{Fore.GREEN}📝 ASCII tree exported to {output_file}{Style.RESET_ALL}")

def parse_arguments():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(description="Explore diverse concepts and their connections.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Name of the model to use (default: %(default)s)")
    parser.add_argument("--root", default=DEFAULT_ROOT_CONCEPT, help="Root concept to start with (default: %(default)s)")
    parser.add_argument("--diversity", type=float, default=DEFAULT_DIVERSITY_BIAS, help="Diversity bias (default: %(default)s)")
    parser.add_argument("--depth", "--max-depth", dest="depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help="Maximum exploration depth (default: %(default)s)")
    return parser.parse_args()

def main():
    clear_terminal()
    print(f"{Fore.GREEN}{'=' * 50}")
    print(f"{Fore.YELLOW}🌳 CONCEPT EXPLORER 🌳")
    print(f"{Fore.GREEN}{'=' * 50}")
    print(f"{Fore.CYAN}Discovering diverse concepts and connections...{Style.RESET_ALL}\n")
    
    args = parse_arguments()
    print(f"{Fore.YELLOW}Starting concept: {Fore.WHITE}{args.root}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Using model: {Fore.WHITE}{args.model}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Diversity bias: {Fore.WHITE}{args.diversity}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Max depth: {Fore.WHITE}{args.depth}{Style.RESET_ALL}")

    explorer = ConceptExplorer(model=args.model)
    if not explorer.check_model_availability():
        print(f"{Fore.RED}Error: Model '{args.model}' is not available in Ollama.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please pull the model first with: 'ollama pull {args.model}'{Style.RESET_ALL}")
        available_models = explorer.get_available_models()
        if available_models:
            print(f"{Fore.GREEN}Available models:{Style.RESET_ALL}")
            for i, available_model in enumerate(available_models, 1):
                print(f"{Fore.CYAN}{i}. {available_model}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Try using one of these models with: python explorer.py --model=<model_name>{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No models available. Please pull a model with: 'ollama pull <model_name>'{Style.RESET_ALL}")
        sys.exit(1)

    try:
        explorer.build_concept_web(args.root, max_depth=args.depth, diversity_bias=args.diversity)
        output_file = f"{args.root.lower()}_concept_web.txt"
        explorer.export_ascii_tree(output_file)
        print(f"\n{Fore.GREEN}✨ Exploration complete! {Fore.YELLOW}Generated concept web with {len(explorer.graph.nodes)} concepts and {len(explorer.graph.edges)} connections.{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Exploration interrupted by user.{Style.RESET_ALL}")
        output_file = f"{args.root.lower()}_concept_web.txt"
        explorer.export_ascii_tree(output_file)
        print(f"{Fore.GREEN}Partial concept web saved with {len(explorer.graph.nodes)} nodes.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    finally:
        print(Style.RESET_ALL)
        print("\nExploration ended. Type 'reset' if your terminal displays incorrectly.")

if __name__ == "__main__":
    main()
