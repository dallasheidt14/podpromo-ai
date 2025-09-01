"""
Paddle payment configuration and product definitions
"""

import os
from typing import Dict, Optional

# Paddle API Configuration
PADDLE_CONFIG = {
    "environment": os.getenv("PADDLE_ENVIRONMENT", "sandbox"),  # sandbox or production
    "api_key": os.getenv("PADDLE_API_KEY"),
    "webhook_secret": os.getenv("PADDLE_WEBHOOK_SECRET"),
}

# Product IDs from Paddle Dashboard
PADDLE_PRODUCTS = {
    "pro": {
        "id": os.getenv("PADDLE_PRO_PRODUCT_ID", "pro_01hq8q8q8q8q8q8q8q8q8q8q"),  # Replace with actual ID
        "name": "Pro Plan",
        "price": 12.99,
        "billing": "monthly",
        "features": ["Unlimited uploads", "Priority processing", "Advanced analytics"]
    }
}

# Usage limits per plan
USAGE_LIMITS = {
    "free": {
        "uploads_per_month": 2,
        "description": "Free tier - 2 uploads per month"
    },
    "pro": {
        "uploads_per_month": -1,  # -1 means unlimited
        "description": "Pro tier - unlimited uploads"
    }
}

def get_product_by_id(product_id: str) -> Optional[Dict]:
    """Get product details by ID"""
    for plan, details in PADDLE_PRODUCTS.items():
        if details["id"] == product_id:
            return {"plan": plan, **details}
    return None

def get_usage_limits(plan: str) -> Dict:
    """Get usage limits for a plan"""
    return USAGE_LIMITS.get(plan, USAGE_LIMITS["free"])

def get_product_url(plan: str) -> Optional[str]:
    """Get Paddle checkout URL for a plan"""
    if plan not in PADDLE_PRODUCTS:
        return None
    
    # This will be generated dynamically when creating checkout sessions
    return None  # Will be set by PaddleService.create_checkout_url
