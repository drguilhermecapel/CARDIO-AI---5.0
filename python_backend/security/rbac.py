from fastapi import HTTPException, Depends, status
from typing import List
from security.auth import AuthManager, User, TokenData

class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user_data: tuple = Depends(AuthManager.get_current_user)):
        user, token_data = user_data
        
        if token_data.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail=f"Operation not permitted for role: {token_data.role}"
            )
            
        if not token_data.mfa_verified:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="MFA verification required"
            )
            
        return user
