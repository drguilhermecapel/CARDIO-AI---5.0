import pytest
from security.encryption import EncryptionManager
from security.auth import AuthManager
from security.rbac import RoleChecker
from fastapi import HTTPException

def test_encryption():
    manager = EncryptionManager()
    text = "Sensitive Data"
    enc = manager.encrypt(text)
    dec = manager.decrypt(enc)
    assert text == dec
    assert text != enc

def test_auth_hashing():
    pwd = "secret_password"
    hashed = AuthManager.get_password_hash(pwd)
    assert AuthManager.verify_password(pwd, hashed)
    assert not AuthManager.verify_password("wrong", hashed)

def test_rbac_logic():
    # Mock Token Data
    from security.auth import TokenData, User
    
    # Allowed
    checker = RoleChecker(["admin"])
    user = User(username="admin", role="admin", disabled=False)
    token = TokenData(username="admin", role="admin", mfa_verified=True)
    
    assert checker((user, token)) == user
    
    # Denied (Role)
    checker_denied = RoleChecker(["cardiologist"])
    try:
        checker_denied((user, token))
        assert False, "Should raise 403"
    except HTTPException as e:
        assert e.status_code == 403
        
    # Denied (MFA)
    token_no_mfa = TokenData(username="admin", role="admin", mfa_verified=False)
    try:
        checker((user, token_no_mfa))
        assert False, "Should raise 403 for missing MFA"
    except HTTPException as e:
        assert e.status_code == 403
