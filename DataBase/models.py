from sqlalchemy import Column, Integer, String, Float
from database import Base

class House(Base):
    __tablename__ = "houses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    district_name = Column(String(255), nullable=False)
    longitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    building_area = Column(Float, nullable=False)
    listing_price = Column(Float, nullable=False)
    year_built = Column(Integer, nullable=False)
    house_ownership = Column(String(255), nullable=False)
