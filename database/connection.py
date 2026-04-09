# database/connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models.db_models import Base
import os

class DatabaseManager:
    def __init__(self, db_url: str = None):
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "sqlite:///./clinical_data.db")
        
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def init_db(self):
        """Создает все таблицы в базе данных"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self) -> Session:
        """Возвращает новую сессию для работы с БД"""
        return self.SessionLocal()
