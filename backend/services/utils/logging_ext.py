"""
Structured logging utilities for compact JSON events.
"""

import json
import logging

def log_json(logger, event: str, **fields):
    """
    Log a structured JSON event alongside a brief human-readable message.
    
    Args:
        logger: Logger instance
        event: Event name (e.g., "DYNAMIC", "FINALS", "VIRALITY", "GATES")
        **fields: Additional fields to include in the JSON
    """
    try:
        # Create structured JSON log
        log_data = {"event": event, **fields}
        json_str = json.dumps(log_data, ensure_ascii=False)
        logger.info(json_str)
        
        # Also log a brief human-readable summary
        if event == "DYNAMIC":
            mode = fields.get("mode", "unknown")
            peaks = fields.get("peaks", 0)
            kept = fields.get("kept", 0)
            logger.info(f"DYNAMIC_DISCOVERY: mode={mode}, peaks={peaks}, kept={kept}")
            
        elif event == "FINALS":
            n = fields.get("n", 0)
            min_finals = fields.get("min_finals", 0)
            enforced = fields.get("enforced", False)
            logger.info(f"FINALS: n={n}, min_finals={min_finals}, enforced={enforced}")
            
        elif event == "VIRALITY":
            scaler = fields.get("scaler", "unknown")
            range_val = fields.get("range", 0.0)
            logger.info(f"VIRALITY: scaler={scaler}, range={range_val:.3f}")
            
        elif event == "GATES":
            finished_required = fields.get("finished_required", False)
            authoritative = fields.get("authoritative", False)
            dropped_for = fields.get("dropped_for", "")
            logger.info(f"GATES: finished_required={finished_required}, authoritative={authoritative}, dropped_for={dropped_for}")
            
        else:
            # Generic fallback
            logger.info(f"{event}: {fields}")
            
    except Exception as e:
        # Fallback to simple logging if JSON serialization fails
        logger.info(f"{event} {fields}")
