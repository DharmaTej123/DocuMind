"""
Auth endpoints — FR1 (register/login, JWT session).
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User
from app.core.security import hash_password, verify_password, create_access_token
from app.schemas import RegisterRequest, RegisterResponse, LoginRequest, TokenResponse

router = APIRouter()


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "An account with this email already exists")

    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return RegisterResponse(id=user.id)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")

    token = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=token)
