"""
Paddle payment service for handling subscriptions and webhooks
"""

import logging
import hmac
import hashlib
import httpx
from typing import Dict, Optional
from datetime import datetime

from config.paddle_config import PADDLE_CONFIG, PADDLE_PRODUCTS, get_product_by_id
from services.auth_service import AuthService

logger = logging.getLogger(__name__)

class PaddleService:
    """Service for handling Paddle payments and subscriptions"""
    
    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service
        self.environment = PADDLE_CONFIG["environment"]
        self.api_key = PADDLE_CONFIG["api_key"]
        self.webhook_secret = PADDLE_CONFIG["webhook_secret"]
        
        # Base URLs for Paddle API
        if self.environment == "sandbox":
            self.base_url = "https://sandbox-vendors.paddle.com"
            self.checkout_url = "https://sandbox-checkout.paddle.com"
        else:
            self.base_url = "https://vendors.paddle.com"
            self.checkout_url = "https://checkout.paddle.com"
    
    def verify_webhook_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature from Paddle"""
        try:
            if not self.webhook_secret:
                logger.warning("No webhook secret configured, skipping signature verification")
                return True
            
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {e}")
            return False
    
    async def process_webhook(self, event_type: str, data: Dict) -> bool:
        """Process Paddle webhook events"""
        try:
            logger.info(f"Processing Paddle webhook: {event_type}")
            
            if event_type == "subscription.created":
                return await self._handle_subscription_created(data)
            elif event_type == "subscription.updated":
                return await self._handle_subscription_updated(data)
            elif event_type == "subscription.cancelled":
                return await self._handle_subscription_cancelled(data)
            elif event_type == "payment.succeeded":
                return await self._handle_payment_succeeded(data)
            elif event_type == "payment.failed":
                return await self._handle_payment_failed(data)
            else:
                logger.info(f"Unhandled webhook event: {event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Webhook processing failed: {e}")
            return False
    
    async def _handle_subscription_created(self, data: Dict) -> bool:
        """Handle new subscription creation"""
        try:
            customer_id = data.get("customer_id")
            subscription_id = data.get("subscription_id")
            product_id = data.get("product_id")
            
            if not all([customer_id, subscription_id, product_id]):
                logger.error("Missing required fields in subscription.created webhook")
                return False
            
            # Get product details
            product = get_product_by_id(product_id)
            if not product:
                logger.error(f"Unknown product ID: {product_id}")
                return False
            
            plan = product["plan"]
            
            # Update user membership
            await self.auth_service.update_membership(
                user_id=customer_id,
                plan=plan,
                subscription_id=subscription_id,
                status="active"
            )
            
            logger.info(f"Subscription created for user {customer_id}: {plan} plan")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle subscription.created: {e}")
            return False
    
    async def _handle_subscription_updated(self, data: Dict) -> bool:
        """Handle subscription updates"""
        try:
            customer_id = data.get("customer_id")
            subscription_id = data.get("subscription_id")
            status = data.get("status")
            
            if not all([customer_id, subscription_id, status]):
                logger.error("Missing required fields in subscription.updated webhook")
                return False
            
            # Update user membership status
            await self.auth_service.update_membership(
                user_id=customer_id,
                subscription_id=subscription_id,
                status=status
            )
            
            logger.info(f"Subscription updated for user {customer_id}: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle subscription.updated: {e}")
            return False
    
    async def _handle_subscription_cancelled(self, data: Dict) -> bool:
        """Handle subscription cancellation"""
        try:
            customer_id = data.get("customer_id")
            subscription_id = data.get("subscription_id")
            
            if not all([customer_id, subscription_id]):
                logger.error("Missing required fields in subscription.cancelled webhook")
                return False
            
            # Downgrade user to free plan
            await self.auth_service.update_membership(
                user_id=customer_id,
                plan="free",
                subscription_id=None,
                status="cancelled"
            )
            
            logger.info(f"Subscription cancelled for user {customer_id}, downgraded to free")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle subscription.cancelled: {e}")
            return False
    
    async def _handle_payment_succeeded(self, data: Dict) -> bool:
        """Handle successful payment"""
        try:
            customer_id = data.get("customer_id")
            subscription_id = data.get("subscription_id")
            amount = data.get("amount")
            
            logger.info(f"Payment succeeded for user {customer_id}: ${amount}")
            
            # Log usage for analytics
            await self.auth_service.log_usage(
                user_id=customer_id,
                action="payment_succeeded",
                details={"amount": amount, "subscription_id": subscription_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle payment.succeeded: {e}")
            return False
    
    async def _handle_payment_failed(self, data: Dict) -> bool:
        """Handle failed payment"""
        try:
            customer_id = data.get("customer_id")
            subscription_id = data.get("subscription_id")
            
            logger.warning(f"Payment failed for user {customer_id}")
            
            # Log usage for analytics
            await self.auth_service.log_usage(
                user_id=customer_id,
                action="payment_failed",
                details={"subscription_id": subscription_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle payment.failed: {e}")
            return False
    
    async def create_checkout_url(self, user_id: str, plan: str = "pro") -> Optional[str]:
        """Create Paddle checkout URL for subscription"""
        try:
            if plan not in PADDLE_PRODUCTS:
                logger.error(f"Invalid plan: {plan}")
                return None
            
            product = PADDLE_PRODUCTS[plan]
            
            # Create checkout session using Paddle API
            checkout_data = {
                "customer_id": user_id,
                "product_id": product["id"],
                "quantity": 1,
                "success_url": f"{self.checkout_url}/success?user_id={user_id}",
                "cancel_url": f"{self.checkout_url}/cancel?user_id={user_id}",
            }
            
            # For now, return a simple checkout URL
            # In production, you'd use Paddle's API to create a proper checkout session
            checkout_url = f"{self.checkout_url}/checkout/{product['id']}?customer_id={user_id}"
            
            logger.info(f"Created checkout URL for user {user_id}: {plan} plan")
            return checkout_url
            
        except Exception as e:
            logger.error(f"Failed to create checkout URL: {e}")
            return None
    
    def get_available_plans(self) -> Dict:
        """Get available subscription plans"""
        return {
            "free": {
                "name": "Free",
                "price": 0,
                "billing": "monthly",
                "features": ["2 uploads per month", "Basic AI scoring", "Standard processing"],
                "description": "Perfect for getting started"
            },
            "pro": {
                "name": "Pro",
                "price": 12.99,
                "billing": "monthly",
                "features": ["Unlimited uploads", "Priority processing", "Advanced analytics", "Premium support"],
                "description": "For serious content creators"
            }
        }
