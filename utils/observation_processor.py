from typing import Dict, List, Any
import re

class ObservationProcessor:
    """Process WebArena observations into structured data"""
    
    def process(self, observation: str) -> Dict[str, Any]:
        """Convert raw observation to structured format"""
        if isinstance(observation, str) and observation.startswith("Error"):
            return self._process_error(observation)
            
        # Process accessibility tree format
        return self._process_accessibility_tree(observation)
        
    def _process_accessibility_tree(self, obs: str) -> Dict[str, Any]:
        """Parse accessibility tree observation"""
        elements = []
        lines = obs.strip().split('\n') if isinstance(obs, str) else []
        
        for line in lines:
            element = self._parse_element_line(line)
            if element:
                elements.append(element)
                
        return {
            'type': 'accessibility_tree',
            'elements': elements,
            'num_elements': len(elements),
            'has_form': any(e['tag'] in ['input', 'textarea', 'select'] for e in elements),
            'has_button': any(e['tag'] == 'button' or 'button' in e.get('text', '').lower() for e in elements),
            'raw': obs
        }
        
    def _parse_element_line(self, line: str) -> Dict[str, Any]:
        """Parse a single element from accessibility tree"""
        # Basic parsing - adapt based on actual WebArena format
        element = {}
        
        # Extract element ID [number]
        id_match = re.search(r'\[(\d+)\]', line)
        if id_match:
            element['id'] = int(id_match.group(1))
            
        # Extract tag type
        tag_match = re.search(r'<(\w+)', line)
        if tag_match:
            element['tag'] = tag_match.group(1).lower()
            
        # Extract text content
        text_match = re.search(r'text="([^"]*)"', line)
        if text_match:
            element['text'] = text_match.group(1)
            
        # Extract other attributes
        element['clickable'] = 'clickable' in line.lower()
        type_match = re.search(r'type="([^"]*)"', line)
        element['type'] = type_match.group(1) if type_match else None
        
        return element if element else None
        
    def _process_error(self, error_obs: str) -> Dict[str, Any]:
        """Process error observations"""
        return {
            'type': 'error',
            'error_message': error_obs,
            'elements': [],
            'num_elements': 0
        }