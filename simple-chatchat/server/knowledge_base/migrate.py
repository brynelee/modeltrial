
from server.db.base import Base, engine

def create_tables():
    Base.metadata.create_all(bind=engine)

def reset_tables():
    Base.metadata.drop_all(bind=engine)
    create_tables()