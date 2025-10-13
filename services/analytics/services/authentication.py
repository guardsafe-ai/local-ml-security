"""
Authentication and RBAC Service
Implements JWT-based authentication and role-based access control
"""

import logging
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(Enum):
    """System permissions"""
    # Data permissions
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    UPLOAD_DATA = "upload_data"
    
    # Model permissions
    TRAIN_MODEL = "train_model"
    DEPLOY_MODEL = "deploy_model"
    PREDICT_MODEL = "predict_model"
    DELETE_MODEL = "delete_model"
    
    # Analytics permissions
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_ANALYTICS = "export_analytics"
    MANAGE_DRIFT = "manage_drift"
    
    # System permissions
    MANAGE_USERS = "manage_users"
    VIEW_LOGS = "view_logs"
    MANAGE_SYSTEM = "manage_system"

@dataclass
class User:
    """User model"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: Set[Permission]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    failed_login_attempts: int = 0
    locked_until: datetime = None

@dataclass
class Session:
    """User session"""
    session_id: str
    user_id: str
    token: str
    expires_at: datetime
    ip_address: str
    user_agent: str
    created_at: datetime

class AuthenticationService:
    """Handles user authentication and session management"""
    
    def __init__(self, secret_key: str = None, db_manager=None):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "default-secret-key")
        self.db_manager = db_manager
        self.active_sessions = {}
        self.users = {}  # In-memory user store (should be replaced with DB)
        self.role_permissions = self._initialize_role_permissions()
        
        # Initialize default admin user
        self._create_default_admin()
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize role-based permissions"""
        return {
            UserRole.ADMIN: {
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.DELETE_DATA,
                Permission.UPLOAD_DATA, Permission.TRAIN_MODEL, Permission.DEPLOY_MODEL,
                Permission.PREDICT_MODEL, Permission.DELETE_MODEL, Permission.VIEW_ANALYTICS,
                Permission.EXPORT_ANALYTICS, Permission.MANAGE_DRIFT, Permission.MANAGE_USERS,
                Permission.VIEW_LOGS, Permission.MANAGE_SYSTEM
            },
            UserRole.DATA_SCIENTIST: {
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.UPLOAD_DATA,
                Permission.TRAIN_MODEL, Permission.DEPLOY_MODEL, Permission.PREDICT_MODEL,
                Permission.VIEW_ANALYTICS, Permission.EXPORT_ANALYTICS, Permission.MANAGE_DRIFT
            },
            UserRole.ANALYST: {
                Permission.READ_DATA, Permission.VIEW_ANALYTICS, Permission.EXPORT_ANALYTICS,
                Permission.PREDICT_MODEL, Permission.MANAGE_DRIFT
            },
            UserRole.VIEWER: {
                Permission.READ_DATA, Permission.VIEW_ANALYTICS, Permission.PREDICT_MODEL
            },
            UserRole.GUEST: {
                Permission.PREDICT_MODEL
            }
        }
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@guardsafe.ai",
            password_hash=self._hash_password(admin_password),
            role=UserRole.ADMIN,
            permissions=self.role_permissions[UserRole.ADMIN],
            created_at=datetime.now(timezone.utc)
        )
        self.users["admin"] = admin_user
        logger.info("✅ [AUTH] Default admin user created")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_jwt_token(self, user_id: str, session_id: str) -> str:
        """Generate JWT token"""
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            "iat": datetime.now(timezone.utc)
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def _verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str = None, user_agent: str = None) -> Dict[str, Any]:
        """Authenticate user and create session"""
        try:
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                logger.warning(f"🚨 [AUTH] Login attempt with unknown username: {username}")
                raise ValueError("Invalid credentials")
            
            # Check if user is locked
            if user.locked_until and user.locked_until > datetime.now(timezone.utc):
                logger.warning(f"🚨 [AUTH] Login attempt for locked user: {username}")
                raise ValueError("Account is temporarily locked")
            
            # Verify password
            password_hash = self._hash_password(password)
            if user.password_hash != password_hash:
                user.failed_login_attempts += 1
                
                # Lock account after 5 failed attempts
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                    logger.warning(f"🚨 [AUTH] Account locked for user: {username}")
                
                logger.warning(f"🚨 [AUTH] Failed login attempt for user: {username}")
                raise ValueError("Invalid credentials")
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now(timezone.utc)
            
            # Create session
            session_id = secrets.token_urlsafe(32)
            token = self._generate_jwt_token(user.user_id, session_id)
            
            session = Session(
                session_id=session_id,
                user_id=user.user_id,
                token=token,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=datetime.now(timezone.utc)
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"✅ [AUTH] User {username} authenticated successfully")
            
            return {
                "token": token,
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions],
                "session_id": session_id,
                "expires_at": session.expires_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ [AUTH] Authentication failed: {e}")
            raise
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user info"""
        try:
            payload = self._verify_jwt_token(token)
            session_id = payload["session_id"]
            
            # Check if session exists and is valid
            if session_id not in self.active_sessions:
                raise ValueError("Session not found")
            
            session = self.active_sessions[session_id]
            if session.expires_at < datetime.now(timezone.utc):
                del self.active_sessions[session_id]
                raise ValueError("Session expired")
            
            # Get user info
            user = self.users.get(session.user_id)
            if not user or not user.is_active:
                raise ValueError("User not found or inactive")
            
            return {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions],
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"❌ [AUTH] Token verification failed: {e}")
            raise
    
    async def logout_user(self, token: str):
        """Logout user and invalidate session"""
        try:
            payload = self._verify_jwt_token(token)
            session_id = payload["session_id"]
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"✅ [AUTH] User logged out: {payload['user_id']}")
            
        except Exception as e:
            logger.error(f"❌ [AUTH] Logout failed: {e}")
    
    def check_permission(self, user_permissions: Set[Permission], required_permission: Permission) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
    
    async def create_user(self, username: str, email: str, password: str, 
                         role: UserRole, created_by: str) -> str:
        """Create new user (admin only)"""
        try:
            # Check if creator has permission
            creator = self.users.get(created_by)
            if not creator or not self.check_permission(creator.permissions, Permission.MANAGE_USERS):
                raise ValueError("Insufficient permissions to create user")
            
            # Check if user already exists
            for user in self.users.values():
                if user.username == username or user.email == email:
                    raise ValueError("User already exists")
            
            # Create new user
            user_id = secrets.token_urlsafe(16)
            new_user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=self._hash_password(password),
                role=role,
                permissions=self.role_permissions[role],
                created_at=datetime.now(timezone.utc)
            )
            
            self.users[user_id] = new_user
            
            logger.info(f"✅ [AUTH] New user created: {username} ({role.value})")
            return user_id
            
        except Exception as e:
            logger.error(f"❌ [AUTH] Failed to create user: {e}")
            raise
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
    
    async def list_users(self, requester_id: str) -> List[Dict[str, Any]]:
        """List all users (admin only)"""
        try:
            requester = self.users.get(requester_id)
            if not requester or not self.check_permission(requester.permissions, Permission.MANAGE_USERS):
                raise ValueError("Insufficient permissions")
            
            users = []
            for user in self.users.values():
                users.append({
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login": user.last_login.isoformat() if user.last_login else None
                })
            
            return users
            
        except Exception as e:
            logger.error(f"❌ [AUTH] Failed to list users: {e}")
            raise

# Global authentication service
auth_service = AuthenticationService()
