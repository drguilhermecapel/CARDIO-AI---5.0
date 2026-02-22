from fastapi import Request, HTTPException
from functools import wraps
from .auth import AuthManager, TokenData

class RBACMiddleware:
    """
    Role-Based Access Control Decorator/Dependency.
    """
    def __init__(self, required_roles: list):
        self.required_roles = required_roles

    async def __call__(self, request: Request):
        # Extract token from header manually if needed, 
        # but usually used with Depends(AuthManager.get_current_user)
        # This class is best used as a dependency factory.
        pass

def require_role(roles: list):
    """
    FastAPI Dependency for RBAC.
    Usage: @app.get("/", dependencies=[Depends(require_role(["admin"]))])
    """
    async def dependency(current_user_and_token = fastapi.Depends(AuthManager.get_current_user)):
        user, token_data = current_user_and_token
        AuthManager.check_permissions(token_data, roles)
        return user
    return dependency

# Helper for route decorators if preferred
import fastapi

def RoleChecker(allowed_roles: list):
    async def _role_checker(user_data = fastapi.Depends(AuthManager.get_current_user)):
        user, token_data = user_data
        if token_data.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Operation not permitted")
        if not token_data.mfa_verified:
             raise HTTPException(status_code=403, detail="MFA required")
        return user
    return _role_checker
