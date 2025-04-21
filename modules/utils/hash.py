import hashlib
import threading
import os
import json
from collections import OrderedDict
from functools import lru_cache

from ..config import NODE_CACHE_DIR

CACHE_FILE = os.path.join(NODE_CACHE_DIR, "model_hash_cache.json")
CACHE_SIZE_LIMIT = 100


cache_model_hash = OrderedDict()
_disk_cache = {}
_disk_cache_dirty = False
_cache_lock = threading.Lock()

# Load cache from file on startup
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _disk_cache = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load cache file {CACHE_FILE}: {e}")
        _disk_cache = {}

@lru_cache(maxsize=100)  # Cache up to 100 file modification times
def get_file_mod_time(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0

def trim_disk_cache():
    global _disk_cache
    if len(_disk_cache) > CACHE_SIZE_LIMIT:
        _disk_cache = dict(list(_disk_cache.items())[-CACHE_SIZE_LIMIT:])

def save_disk_cache():
    global _disk_cache_dirty
    if not _disk_cache_dirty:
        return  # Skip write if unchanged
    try:
        trim_disk_cache()
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        temp_file = CACHE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(_disk_cache, f, indent=2)
        os.replace(temp_file, CACHE_FILE)  # Atomic write
        _disk_cache_dirty = False  # Reset flag after successful write
    except Exception as e:
        print(f"[ERROR] Failed to write cache to {CACHE_FILE}: {e}")

def calc_hash(filename, use_only_filename=True):
    global _disk_cache_dirty
    key = os.path.basename(filename) if use_only_filename else filename
    current_mod_time = get_file_mod_time(filename)

    with _cache_lock:
        # Check in-memory cache first
        if key in cache_model_hash:
            return cache_model_hash[key]

        # Check disk cache if not found in memory
        record = _disk_cache.get(key)
        if record and record.get("file_modification_date") == current_mod_time:
            # Update in-memory cache from disk cache
            cache_model_hash[key] = record["file_hash"]
            # Maintain LRU order
            cache_model_hash.move_to_end(key)
            return record["file_hash"]

    try:
        # Calculate hash if not found in any cache
        sha256_hash = hashlib.sha256()
        with open(filename, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        model_hash = sha256_hash.hexdigest()[:10]

        with _cache_lock:
            # Update in-memory cache with size limit
            if len(cache_model_hash) >= CACHE_SIZE_LIMIT:
                cache_model_hash.popitem(last=False)  # Remove oldest item
            cache_model_hash[key] = model_hash

            # Update disk cache only if necessary
            if key not in _disk_cache or _disk_cache[key].get("file_modification_date") != current_mod_time:
                _disk_cache[key] = {
                    "file_hash": model_hash,
                    "file_modification_date": current_mod_time
                }
                _disk_cache_dirty = True  # Mark cache as changed

            # Save disk cache only if dirty
            if _disk_cache_dirty:
                save_disk_cache()

        return model_hash
    except Exception as e:
        print(f"[ERROR] Failed to calculate hash for {filename}: {e}")
        return ""
