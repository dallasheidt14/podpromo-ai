"""
Authentication service for user management and membership
"""

import logging
import uuid
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from supabase import create_client, Client

from config.paddle_config import get_usage_limits

logger = logging.getLogger(__name__)

class AuthService:
    """Service for user authentication and membership management"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.usage_cache: Dict[str, Dict] = {}  # Cache usage data
    
    async def signup_user(self, email: str, password: str, name: str) -> Dict:
        """Create new user account"""
        try:
            # Create user in Supabase Auth
            auth_response = self.supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if auth_response.user is None:
                raise Exception("Failed to create user account")
            
            user_id = auth_response.user.id
            
            # Create user profile
            profile_data = {
                "id": user_id,
                "email": email,
                "name": name,
                "plan": "free",  # Default to free plan
                "subscription_id": None,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Insert profile into profiles table
            profile_response = self.supabase.table("profiles").insert(profile_data).execute()
            
            if profile_response.data is None:
                # If profile creation fails, clean up the auth user
                await self._cleanup_failed_signup(user_id)
                raise Exception("Failed to create user profile")
            
            # Create initial membership record
            membership_data = {
                "user_id": user_id,
                "plan": "free",
                "subscription_id": None,
                "status": "active",
                "start_date": datetime.now().isoformat(),
                "end_date": None
            }
            
            membership_response = self.supabase.table("user_memberships").insert(membership_data).execute()
            
            if membership_response.data is None:
                logger.warning(f"Failed to create membership record for user {user_id}")
            
            logger.info(f"User account created successfully: {user_id}")
            
            return {
                "user_id": user_id,
                "email": email,
                "name": name,
                "plan": "free",
                "message": "Account created successfully"
            }
            
        except Exception as e:
            logger.error(f"User signup failed: {e}")
            raise
    
    async def login_user(self, email: str, password: str) -> Dict:
        """Authenticate user login"""
        try:
            # Authenticate with Supabase
            auth_response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if auth_response.user is None:
                raise Exception("Invalid email or password")
            
            user_id = auth_response.user.id
            
            # Get user profile
            profile = await self.get_user_profile(user_id)
            if not profile:
                raise Exception("User profile not found")
            
            logger.info(f"User login successful: {user_id}")
            
            return {
                "user_id": user_id,
                "email": profile["email"],
                "name": profile["name"],
                "plan": profile["plan"],
                "access_token": auth_response.session.access_token
            }
            
        except Exception as e:
            logger.error(f"User login failed: {e}")
            raise
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile information"""
        try:
            response = self.supabase.table("profiles").select("*").eq("id", user_id).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user profile {user_id}: {e}")
            return None
    
    async def get_user_membership(self, user_id: str) -> Optional[Dict]:
        """Get user's current membership"""
        try:
            response = self.supabase.table("user_memberships").select("*").eq("user_id", user_id).order("start_date", desc=True).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user membership {user_id}: {e}")
            return None
    
    async def update_membership(self, user_id: str, plan: str, subscription_id: Optional[str] = None, status: str = "active") -> bool:
        """Update user membership plan"""
        try:
            # Update profile
            profile_update = {
                "plan": plan,
                "subscription_id": subscription_id,
                "updated_at": datetime.now().isoformat()
            }
            
            profile_response = self.supabase.table("profiles").update(profile_update).eq("id", user_id).execute()
            
            if profile_response.data is None:
                logger.error(f"Failed to update profile for user {user_id}")
                return False
            
            # Create new membership record
            membership_data = {
                "user_id": user_id,
                "plan": plan,
                "subscription_id": subscription_id,
                "status": status,
                "start_date": datetime.now().isoformat(),
                "end_date": None
            }
            
            # If cancelling, set end date for previous membership
            if status == "cancelled":
                await self._end_current_membership(user_id)
            
            membership_response = self.supabase.table("user_memberships").insert(membership_data).execute()
            
            if membership_response.data is None:
                logger.error(f"Failed to create membership record for user {user_id}")
                return False
            
            # Clear usage cache
            if user_id in self.usage_cache:
                del self.usage_cache[user_id]
            
            logger.info(f"Membership updated for user {user_id}: {plan} plan")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update membership for user {user_id}: {e}")
            return False
    
    async def check_usage_limits(self, user_id: str) -> Dict:
        """Check if user can upload based on their plan"""
        try:
            # Get user profile
            profile = await self.get_user_profile(user_id)
            if not profile:
                return {"can_upload": False, "reason": "User profile not found"}
            
            plan = profile.get("plan", "free")
            limits = get_usage_limits(plan)
            
            # Pro users have unlimited uploads
            if plan == "pro":
                return {"can_upload": True, "plan": plan, "remaining": -1}
            
            # Free users: check monthly limit
            if plan == "free":
                current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                
                # Get uploads this month
                response = self.supabase.table("usage_logs").select("count").eq("user_id", user_id).eq("action", "file_upload").gte("created_at", current_month.isoformat()).execute()
                
                uploads_this_month = len(response.data) if response.data else 0
                max_uploads = limits["uploads_per_month"]
                
                if uploads_this_month >= max_uploads:
                    return {
                        "can_upload": False,
                        "plan": plan,
                        "reason": f"Monthly limit reached ({max_uploads} uploads)",
                        "used": uploads_this_month,
                        "limit": max_uploads
                    }
                
                return {
                    "can_upload": True,
                    "plan": plan,
                    "remaining": max_uploads - uploads_this_month,
                    "used": uploads_this_month,
                    "limit": max_uploads
                }
            
            return {"can_upload": False, "reason": "Unknown plan"}
            
        except Exception as e:
            logger.error(f"Failed to check usage limits for user {user_id}: {e}")
            return {"can_upload": False, "reason": "Error checking limits"}
    
    async def log_usage(self, user_id: str, action: str, details: Optional[Dict] = None) -> bool:
        """Log user action for analytics and billing"""
        try:
            usage_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "action": action,
                "details": details or {},
                "created_at": datetime.now().isoformat()
            }
            
            response = self.supabase.table("usage_logs").insert(usage_data).execute()
            
            if response.data is None:
                logger.error(f"Failed to log usage for user {user_id}")
                return False
            
            # Clear usage cache
            if user_id in self.usage_cache:
                del self.usage_cache[user_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log usage for user {user_id}: {e}")
            return False
    
    async def get_usage_summary(self, user_id: str) -> Dict:
        """Get user's usage summary"""
        try:
            # Check cache first
            if user_id in self.usage_cache:
                return self.usage_cache[user_id]
            
            # Get current month usage
            current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            response = self.supabase.table("usage_logs").select("*").eq("user_id", user_id).gte("created_at", current_month.isoformat()).execute()
            
            usage_logs = response.data if response.data else []
            
            # Count different actions
            uploads = len([log for log in usage_logs if log["action"] == "file_upload"])
            clips_generated = len([log for log in usage_logs if log["action"] == "clip_generation"])
            
            # Get user profile for plan info
            profile = await self.get_user_profile(user_id)
            plan = profile.get("plan", "free") if profile else "free"
            limits = get_usage_limits(plan)
            
            summary = {
                "user_id": user_id,
                "plan": plan,
                "current_month": {
                    "uploads": uploads,
                    "clips_generated": clips_generated
                },
                "limits": limits,
                "can_upload": True
            }
            
            # Check if free user has reached limit
            if plan == "free":
                max_uploads = limits["uploads_per_month"]
                summary["can_upload"] = uploads < max_uploads
                summary["remaining_uploads"] = max(0, max_uploads - uploads)
            
            # Cache the result
            self.usage_cache[user_id] = summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get usage summary for user {user_id}: {e}")
            return {"error": str(e)}
    
    async def _cleanup_failed_signup(self, user_id: str):
        """Clean up failed user signup"""
        try:
            # Delete user from Supabase Auth
            self.supabase.auth.admin.delete_user(user_id)
            logger.info(f"Cleaned up failed signup for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup failed signup for user {user_id}: {e}")
    
    async def _end_current_membership(self, user_id: str):
        """End current membership by setting end date"""
        try:
            # Find current active membership
            response = self.supabase.table("user_memberships").select("*").eq("user_id", user_id).eq("status", "active").order("start_date", desc=True).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                current_membership = response.data[0]
                
                # Update end date
                update_data = {
                    "end_date": datetime.now().isoformat(),
                    "status": "ended"
                }
                
                self.supabase.table("user_memberships").update(update_data).eq("id", current_membership["id"]).execute()
                
                logger.info(f"Ended current membership for user {user_id}")
        
        except Exception as e:
            logger.error(f"Failed to end current membership for user {user_id}: {e}")
