import json
import hashlib
import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

class DAGCacheManager:
    """Manages caching of successful DAGs for reuse"""
    
    def __init__(self, cache_dir: str = "dag_cache", max_cache_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for fast access
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Load existing cache from disk
        self._load_cache_from_disk()
    
    def _load_cache_from_disk(self):
        """Load cache entries from disk into memory"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            
            # Sort by modification time to keep most recent
            cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Load up to max_cache_size entries
            for cache_file in cache_files[:self.max_cache_size]:
                try:
                    with open(cache_file, 'r') as f:
                        cache_entry = json.load(f)
                    
                    # Extract context hash from filename
                    context_hash = cache_file.stem
                    self.memory_cache[context_hash] = cache_entry
                    
                except Exception as e:
                    print(f"Warning: Failed to load cache file {cache_file}: {e}")
                    # Remove corrupted cache file
                    try:
                        cache_file.unlink()
                    except:
                        pass
            
            print(f"Loaded {len(self.memory_cache)} DAG cache entries")
            
        except Exception as e:
            print(f"Warning: Failed to load DAG cache: {e}")
    
    def _compute_context_hash(self, context: Dict) -> str:
        """Compute a hash for the context to use as cache key"""
        # Extract relevant parts for caching
        cache_key_data = {
            'task_intent': context.get('task', {}).get('intent', ''),
            'task_sites': context.get('task', {}).get('sites', []),
            'task_difficulty': context.get('task', {}).get('difficulty', ''),
            'page_type': context.get('current_state', {}).get('page_type', ''),
            'elements_available': sorted(context.get('current_state', {}).get('elements_available', [])),
            'budget_remaining': round(context.get('constraints', {}).get('budget', 1.0) - 
                                   context.get('constraints', {}).get('cost_spent', 0), 2),
            'method': context.get('constraints', {}).get('current_method', 'p3o')
        }
        
        # Create deterministic hash
        cache_key_str = json.dumps(cache_key_data, sort_keys=True)
        return hashlib.md5(cache_key_str.encode()).hexdigest()
    
    def get_cached_dag(self, context: Dict) -> Optional[Dict]:
        """Retrieve a cached DAG if available"""
        context_hash = self._compute_context_hash(context)
        
        if context_hash in self.memory_cache:
            cache_entry = self.memory_cache[context_hash]
            
            # Check if cache entry is still valid
            if self._is_cache_valid(cache_entry, context):
                # Update access time
                cache_entry['last_accessed'] = time.time()
                self._save_cache_entry(context_hash, cache_entry)
                
                self.cache_stats['hits'] += 1
                return cache_entry['dag']
        
        self.cache_stats['misses'] += 1
        return None
    
    def cache_successful_dag(self, context: Dict, dag: Dict, metrics: Dict):
        """Cache a successful DAG for future reuse"""
        context_hash = self._compute_context_hash(context)
        
        cache_entry = {
            'dag': dag,
            'context_summary': {
                'task_intent': context.get('task', {}).get('intent', ''),
                'task_sites': context.get('task', {}).get('sites', []),
                'page_type': context.get('current_state', {}).get('page_type', ''),
                'budget_used': metrics.get('cost', 0),
                'method': context.get('constraints', {}).get('current_method', 'p3o')
            },
            'metrics': metrics,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'access_count': 1
        }
        
        # Add to memory cache
        self.memory_cache[context_hash] = cache_entry
        
        # Save to disk
        self._save_cache_entry(context_hash, cache_entry)
        
        # Manage cache size
        self._evict_if_needed()
    
    def _is_cache_valid(self, cache_entry: Dict, context: Dict) -> bool:
        """Check if a cached entry is still valid for the given context"""
        # Check age (expire after 24 hours)
        age_hours = (time.time() - cache_entry['created_at']) / 3600
        if age_hours > 24:
            return False
        
        # Check if success rate is good
        metrics = cache_entry.get('metrics', {})
        success_rate = metrics.get('success', False)
        if not success_rate:
            return False
        
        # Check if cost is still reasonable
        cached_cost = metrics.get('cost', 0)
        current_budget = context.get('constraints', {}).get('budget', 1.0) - \
                        context.get('constraints', {}).get('cost_spent', 0)
        
        if cached_cost > current_budget * 0.8:  # Don't use if it would consume >80% of budget
            return False
        
        return True
    
    def _save_cache_entry(self, context_hash: str, cache_entry: Dict):
        """Save a cache entry to disk"""
        try:
            cache_file = self.cache_dir / f"{context_hash}.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache entry: {e}")
    
    def _evict_if_needed(self):
        """Evict old cache entries if cache is too large"""
        if len(self.memory_cache) <= self.max_cache_size:
            return
        
        # Sort by last access time and remove oldest entries
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        entries_to_remove = len(self.memory_cache) - self.max_cache_size
        
        for i in range(entries_to_remove):
            context_hash = sorted_entries[i][0]
            
            # Remove from memory
            del self.memory_cache[context_hash]
            
            # Remove from disk
            cache_file = self.cache_dir / f"{context_hash}.json"
            try:
                cache_file.unlink()
            except:
                pass
            
            self.cache_stats['evictions'] += 1
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / max(total_requests, 1)
        
        return {
            'total_entries': len(self.memory_cache),
            'total_requests': total_requests,
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.cache_stats['evictions']
        }
    
    def clear_cache(self):
        """Clear all cache entries"""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clear disk cache: {e}")
        
        # Reset stats
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def get_similar_dags(self, context: Dict, limit: int = 5) -> List[Tuple[Dict, float]]:
        """Get similar DAGs from cache with similarity scores"""
        context_hash = self._compute_context_hash(context)
        current_task = context.get('task', {})
        current_intent = current_task.get('intent', '')
        current_sites = set(current_task.get('sites', []))
        
        similar_dags = []
        
        for cached_hash, cache_entry in self.memory_cache.items():
            if cached_hash == context_hash:
                continue  # Skip exact match
            
            # Calculate similarity
            similarity = self._calculate_similarity(context, cache_entry['context_summary'])
            
            if similarity > 0.3:  # Only include reasonably similar entries
                similar_dags.append((cache_entry['dag'], similarity))
        
        # Sort by similarity and return top results
        similar_dags.sort(key=lambda x: x[1], reverse=True)
        return similar_dags[:limit]
    
    def _calculate_similarity(self, context: Dict, cached_context: Dict) -> float:
        """Calculate similarity between current context and cached context"""
        score = 0.0
        
        # Task intent similarity
        current_intent = context.get('task', {}).get('intent', '')
        cached_intent = cached_context.get('task_intent', '')
        if current_intent and cached_intent:
            # Simple word overlap
            current_words = set(current_intent.lower().split())
            cached_words = set(cached_intent.lower().split())
            if current_words & cached_words:
                score += 0.4
        
        # Site similarity
        current_sites = set(context.get('task', {}).get('sites', []))
        cached_sites = set(cached_context.get('task_sites', []))
        if current_sites & cached_sites:
            score += 0.3
        
        # Page type similarity
        current_page = context.get('current_state', {}).get('page_type', '')
        cached_page = cached_context.get('page_type', '')
        if current_page == cached_page and current_page:
            score += 0.2
        
        # Method similarity
        current_method = context.get('constraints', {}).get('current_method', 'p3o')
        cached_method = cached_context.get('method', 'p3o')
        if current_method == cached_method:
            score += 0.1
        
        return score