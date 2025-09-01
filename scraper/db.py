import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

from sqlalchemy import (Column, Integer, String, Text, Float, ForeignKey, DateTime)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class Source(Base):
    __tablename__ = 'sources'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    base_url = Column(String(1024), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    products = relationship('Product', back_populates='source')


class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('sources.id'), nullable=False)
    source = relationship('Source', back_populates='products')
    source_product_id = Column(String(512), nullable=True, index=True)
    title = Column(String(1024))
    description = Column(Text)
    price = Column(Float)
    currency = Column(String(10))
    image_urls = Column(JSONB)
    page_url = Column(String(2048))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    reviews = relationship('Review', back_populates='product')
    # embeddings will be added in a separate migration if needed


class Review(Base):
    __tablename__ = 'reviews'
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    product = relationship('Product', back_populates='reviews')
    review_text = Column(Text)
    rating = Column(Float, nullable=True)
    author = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def get_engine(echo: bool = False):
    if not DATABASE_URL:
        raise RuntimeError('DATABASE_URL not set in environment')
    engine = create_engine(DATABASE_URL, echo=echo)
    return engine


def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()
