import requests
import json
import time
from collections import deque
import os
import sys
import random
import networkx as nx
from colorama import Fore, Back, Style, init
import shutil 

# Initialize colorama
init(autoreset=True)

class ConceptExplorer:
    def __init__(self, model="qwen2:7b"):
        self.graph = nx.DiGraph()
        self.seen_concepts = set()
        self.last_added = None
        self.current_concept = None
        self.model = model
        
        # terminal dimensions
        self.term_width, self.term_height = shutil.get_terminal_size((80, 24))
        
    def query_ollama(self, prompt):
        """Query Ollama using the generate API."""
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            print(f"{Fore.CYAN}Querying Ollama generate API...{Style.RESET_ALL}")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json().get("response", "")
            return result
        except Exception as e:
            print(f"{Fore.RED}Error querying Ollama: {str(e)}{Style.RESET_ALL}")
            return "[]"  
    
    def get_related_concepts(self, concept, depth=0, path=None):
        """Get diverse related concepts for a given starting concept."""
        if concept in self.seen_concepts or depth > 5:  # Prevent loops and limit depth
            return []
        
        self.seen_concepts.add(concept)
        self.current_concept = concept
        
        if path is None:
            path = []
        
        # Full path to current concept, including the concept itself
        full_path = path + [concept]
        
        # Prompt 
        prompt = f"""
Starting with the concept: "{concept}", generate 4-5 fascinating and unexpected related concepts.

Context: We're building a concept web and have followed this path to get here:
{' → '.join(full_path)}

Guidelines:
1. Seek maximum intellectual diversity - span across domains like science, art, philosophy, technology, culture, etc.
2. Each concept should be expressed in 1-5 words (shorter is better)
3. Avoid obvious associations - prefer surprising or thought-provoking connections
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

Return ONLY a JSON array of strings, with no explanation or additional text.
Example: ["Related concept 1", "Related concept 2", "Related concept 3", "Related concept 4"]
        """
        
        print(f"\n{Fore.CYAN}🔍 Exploring concepts related to: {Fore.YELLOW}{concept}{Style.RESET_ALL}")
        if path:
            print(f"{Fore.CYAN}📜 Path context: {Fore.YELLOW}{' → '.join(path)} → {concept}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}⏳ Thinking...{Style.RESET_ALL}")
        response = self.query_ollama(prompt)
        
        try:
            # Extract JSON array from the response
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                related_concepts = json.loads(json_str)
                
                # Validate concepts - reject overly generic ones
                filtered_concepts = []
                for rc in related_concepts:
                    # Truncate concept if it's too long for display
                    if len(rc) > self.term_width // 3:
                        rc = rc[:self.term_width // 3 - 3] + "..."
                        
                    if not rc.strip() or rc.lower() in [c.lower() for c in self.seen_concepts]:
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
        """Apply appropriate colors to nodes in the tree."""
        connector = "└── " if is_last else "├── "
        
        # Truncate node text if it would exceed terminal width
        available_width = self.term_width - len(prefix) - len(connector) - 5  # 5 for safety margin
        if len(node) > available_width:
            node = node[:available_width-3] + "..."
        
        if node == self.current_concept:
            # Currently being explored
            return f"{prefix}{Fore.CYAN}{connector}{Back.BLUE}{Fore.WHITE}{node}{Style.RESET_ALL}"
        elif node == self.last_added:
            # Just added
            return f"{prefix}{Fore.CYAN}{connector}{Back.GREEN}{Fore.BLACK}{node}{Style.RESET_ALL}"
        elif current_depth == 0:
            # Root node
            return f"{prefix}{Fore.CYAN}{connector}{Fore.MAGENTA}{Style.BRIGHT}{node}{Style.RESET_ALL}"
        else:
            # Regular nodes with colors based on depth
            colors = [Fore.YELLOW, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.RED, Fore.WHITE]
            color = colors[min(current_depth, len(colors)-1)]
            return f"{prefix}{Fore.CYAN}{connector}{color}{node}{Style.RESET_ALL}"
    
    def update_live_tree(self, focus_node=None, max_display_depth=None):
        """Generate and display the current ASCII tree with focus on recently added nodes."""
        # Update terminal size in case it changed
        self.term_width, self.term_height = shutil.get_terminal_size((80, 24))
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Fancy header
        header = [
            f"{Fore.GREEN}🌳 {Fore.YELLOW}C{Fore.GREEN}O{Fore.BLUE}N{Fore.MAGENTA}C{Fore.RED}E{Fore.YELLOW}P{Fore.GREEN}T {Fore.BLUE}E{Fore.MAGENTA}X{Fore.RED}P{Fore.YELLOW}L{Fore.GREEN}O{Fore.BLUE}R{Fore.MAGENTA}E{Fore.RED}R {Fore.GREEN}🌳",
            f"{Fore.CYAN}{'═' * min(50, self.term_width - 2)}",
            ""
        ]
        
        for line in header:
            print(line)
        
        # Find root nodes
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        
        if not roots:
            print(f"{Fore.RED}No root nodes found yet{Style.RESET_ALL}")
            return
        
        # If focus node is specified, show path to that node
        path_to_highlight = []
        if focus_node:
            current = focus_node
            while current:
                path_to_highlight.append(current)
                predecessors = list(self.graph.predecessors(current))
                current = predecessors[0] if predecessors else None
        
        # Calculate available height for tree display
        # Header (3 lines) + Stats footer (3 lines) + Current node (2 lines) + margins (2 lines)
        available_height = self.term_height - 10
        
        # If we have a focus node, display its path with adequate depth
        if focus_node:
            # We want to see at least the path to the focus node
            path_depth = len(path_to_highlight)
            if max_display_depth is None or max_display_depth < path_depth:
                max_display_depth = path_depth + 1  # +1 to see children of focus node
        else:
            # If no focus, adapt to available height (rough estimate)
            if max_display_depth is None:
                # Each level might have ~3 nodes on average, estimate how many levels we can display
                max_display_depth = max(2, min(5, available_height // 3))
        
        # Generate and print the tree
        tree_text = self._generate_ascii_tree(
            roots[0], 
            focus_paths=path_to_highlight, 
            max_depth=max_display_depth,
            available_height=available_height
        )
        print(tree_text)
        
        # Stats footer
        print(f"\n{Fore.CYAN}{'═' * min(50, self.term_width - 2)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 Concepts: {len(self.graph.nodes)} | Connections: {len(self.graph.edges)} | Display depth: {max_display_depth}{Style.RESET_ALL}")
        
        if self.current_concept:
            current_display = self.current_concept
            if len(current_display) > self.term_width - 25:
                current_display = current_display[:self.term_width - 28] + "..."
            print(f"{Fore.CYAN}🔍 Exploring: {Fore.YELLOW}{current_display}{Style.RESET_ALL}")
    
    def _generate_ascii_tree(self, node, prefix="", is_last=True, visited=None, focus_paths=None, max_depth=None, current_depth=0, available_height=24, lines_used=0):
        """Generate ASCII tree representation with colors and focus, respecting terminal height."""
        if visited is None:
            visited = set()
            
        if focus_paths is None:
            focus_paths = []
            
        # Stop rendering if we exceed available height
        if lines_used >= available_height:
            return f"{prefix}{Fore.CYAN}{'└── ' if is_last else '├── '}{Fore.RED}(...more...){Style.RESET_ALL}\n"
            
        # Handle cycles or max depth
        if node in visited or (max_depth is not None and current_depth > max_depth):
            return f"{self._color_node(node, prefix, is_last, current_depth)} {Fore.RED}(...){Style.RESET_ALL}\n"
            
        visited.add(node)
        
        # Color the node based on status
        result = f"{self._color_node(node, prefix, is_last, current_depth)}\n"
        lines_used += 1
        
        children = list(self.graph.successors(node))
        if not children or lines_used >= available_height:
            return result
            
        next_prefix = prefix + ("    " if is_last else "│   ")
        
        # Sort children - put focus path nodes first if applicable
        if focus_paths:
            children.sort(key=lambda x: x not in focus_paths)
        
        # If we need to limit display for space, prioritize focus path
        # and then select a representative sample of nodes
        remaining_height = available_height - lines_used
        if len(children) > remaining_height:
            # Always include focus path nodes
            focus_children = [c for c in children if c in focus_paths]
            non_focus = [c for c in children if c not in focus_paths]
            
            # Take a sample of non-focus nodes
            if len(focus_children) < remaining_height:
                # Evenly sample from beginning, middle and end for better representation
                sample_size = remaining_height - len(focus_children) - 1  # -1 for "more" indicator
                if sample_size > 0:
                    if len(non_focus) <= sample_size:
                        sampled = non_focus
                    else:
                        # Take some from start, middle and end
                        third = max(1, sample_size // 3)
                        sampled = (
                            non_focus[:third] + 
                            non_focus[len(non_focus)//2 - third//2:len(non_focus)//2 + third//2] + 
                            non_focus[-third:]
                        )
                        # Deduplicate
                        sampled = list(dict.fromkeys(sampled))
                        sampled = sampled[:sample_size]
                else:
                    sampled = []
                
                children = focus_children + sampled
                has_more = len(focus_children) + len(non_focus) > len(children)
            else:
                # Just take focus children
                children = focus_children[:remaining_height - 1]  # -1 for "more" indicator
                has_more = len(focus_children) > len(children) or non_focus
        else:
            has_more = False
        
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1 and not has_more
            
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
            
            # Stop if we've reached display limit
            if lines_used >= available_height:
                break
        
        # Show indication that there are more nodes
        if has_more and lines_used < available_height:
            result += f"{next_prefix}{Fore.CYAN}└── {Fore.RED}(...more nodes...){Style.RESET_ALL}\n"
            
        return result
    
    def build_concept_web(self, root_concept, max_depth=3, diversity_bias=0.8):
        """Build the concept web using BFS with enhanced diversity."""
        self.graph.add_node(root_concept)
        self.update_live_tree()
        
        queue = deque([(root_concept, 0, [])]) # (concept, depth, path)
        
        while queue:
            concept, depth, path = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Focus visualization on current part of the tree
            # For wider trees, set a lower max display depth to keep it visible
            display_depth = min(3, max_depth)
            self.update_live_tree(focus_node=concept, max_display_depth=display_depth)
            
            # Get related concepts with path context
            related_concepts = self.get_related_concepts(concept, depth, path)
            
            # Apply diversity bias - occasionally explore less obvious paths
            if diversity_bias > 0 and related_concepts and random.random() < diversity_bias:
                # Prioritize concepts that are most different from what we've seen
                related_concepts.sort(key=lambda x: self._diversity_score(x, self.seen_concepts))
            
            # Add new related concepts to the tree
            for rel_concept in related_concepts:
                if rel_concept not in self.graph:
                    self.graph.add_node(rel_concept)
                    self.last_added = rel_concept
                self.graph.add_edge(concept, rel_concept)
                
                # Each new concept gets the full path to its parent
                new_path = path + [concept]
                queue.append((rel_concept, depth + 1, new_path))
                
                # Flash each new addition with a brief pause
                self.update_live_tree(focus_node=rel_concept, max_display_depth=display_depth)
                time.sleep(0.5)
            
            # Rate limiting for Ollama
            time.sleep(0.5)
        
        # Final full tree display
        self.current_concept = None
        self.last_added = None
        self.update_live_tree()
        print(f"\n{Fore.GREEN}🎉 Concept exploration complete!{Style.RESET_ALL}")
    
    def _diversity_score(self, concept, existing_concepts):
        """Calculate how diverse a concept is compared to existing ones.
        Higher score = more diverse/different from what we've seen."""
        # This is a simple implementation - could be enhanced with embedding distance
        score = 0
        for existing in existing_concepts:
            # Increase score for concepts that don't share words with existing concepts
            shared_words = set(concept.lower().split()) & set(existing.lower().split())
            if not shared_words:
                score += 1  # More diverse = higher score
        return score
    
    def export_ascii_tree(self, output_file="concept_web.txt"):
        """Export the concept web as ASCII text (without colors)."""
        # Find root nodes
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        
        if not roots:
            print(f"{Fore.RED}No root nodes found{Style.RESET_ALL}")
            return
        
        # Generate plain ASCII tree for file export
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
                is_last_child = i == len(children) - 1
                result += _plain_ascii_tree(child, next_prefix, is_last_child, visited.copy())
                
            return result
            
        tree_text = _plain_ascii_tree(roots[0])
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(tree_text)
            
        print(f"{Fore.GREEN}📝 ASCII tree exported to {output_file}{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        # Colorful intro
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Fore.GREEN}{'=' * 50}")
        print(f"{Fore.YELLOW}🌳 CONCEPT EXPLORER 🌳")
        print(f"{Fore.GREEN}{'=' * 50}")
        print(f"{Fore.CYAN}Discovering diverse concepts and connections...{Style.RESET_ALL}")
        print()
        
        # Set the root concept and model
        root_concept = "Consciousness"  # Default root concept
        model = "llama3"  # Default model
        diversity_level = 0.8  # Default diversity bias
        max_depth = 300  # Default exploration depth
        
        # Parse command line arguments
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg.startswith("--model="):
                model = arg.split("=")[1]
            elif arg.startswith("--diversity="):
                try:
                    diversity_level = float(arg.split("=")[1])
                except ValueError:
                    print(f"{Fore.RED}Invalid diversity value, using default 0.8{Style.RESET_ALL}")
            elif arg.startswith("--root="):
                root_concept = arg.split("=")[1]
            elif arg.startswith("--depth="):
                try:
                    max_depth = int(arg.split("=")[1])
                except ValueError:
                    print(f"{Fore.RED}Invalid depth value, using default 3{Style.RESET_ALL}")
            elif i == 1 and not arg.startswith("--"):
                root_concept = arg
        
        print(f"{Fore.YELLOW}Starting concept: {Fore.WHITE}{root_concept}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Using model: {Fore.WHITE}{model}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Diversity bias: {Fore.WHITE}{diversity_level}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Max depth: {Fore.WHITE}{max_depth}{Style.RESET_ALL}")
        
        # Create the concept explorer
        explorer = ConceptExplorer(model=model)
        
        # Build the concept web with enhanced diversity
        try:
            explorer.build_concept_web(root_concept, max_depth=max_depth, diversity_bias=diversity_level)
            
            # Export ASCII tree to file
            explorer.export_ascii_tree(f"{root_concept.lower()}_concept_web.txt")
            
            print(f"\n{Fore.GREEN}✨ Exploration complete! {Fore.YELLOW}Generated concept web with {len(explorer.graph.nodes)} concepts and {len(explorer.graph.edges)} connections.{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Exploration interrupted by user.{Style.RESET_ALL}")
            explorer.export_ascii_tree(f"{root_concept.lower()}_concept_web.txt")
            print(f"{Fore.GREEN}Partial concept web saved with {len(explorer.graph.nodes)} nodes.{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"\n{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Make sure terminal is reset properly
        print(Style.RESET_ALL)
        print("\nExploration ended. Type 'reset' if your terminal displays incorrectly.")
