import pandas as pd
from sqlalchemy.orm import sessionmaker
from database import engine, Base
from models import House  # 确保 House 类已定义

# 创建所有表（如果不存在）
Base.metadata.create_all(bind=engine)


#将房产数据从csv加载到数据库

# 创建数据库会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

# 读取 CSV 文件
csv_file = "Data/cd_pro.csv"  # CSV 文件路径
df = pd.read_csv(csv_file)

# 遍历 DataFrame，将每行数据插入数据库
for _, row in df.iterrows():
    house = House(
        avg_transaction=row["avg_transaction"],
        transaction_cycle=row["transaction_cycle"],
        longitude=row["longitude"],
        latitude=row["latitude"],
        building_area=row["building_area"],
        total_floors=row["total_floors"],
        listing_price=row["listing_price"],
        year_built=row["year_built"],
        price_adjustment=row["price_adjustment"],
        viewing_count=row["viewing_count"],
        followers_count=row["followers_count"],
        visitors_count=row["visitors_count"],
        ladders=row["ladders"],
        houses=row["houses"],
        kitchens=row["kitchens"],
        bedrooms=row["bedrooms"],
        living_rooms=row["living_rooms"],
        bathrooms=row["bathrooms"],
        listing_year=row["listing_year"],
        listing_month=row["listing_month"],
        listing_day=row["listing_day"],
        district_name=row["district_name"],  
        business_area=row["business_area"],
        community_name=row["community_name"],
        apartment_type=row["apartment_type"],
        building_type=row["building_type"],
        house_orientation=row["house_orientation"],
        renovation_status=row["renovation_status"],
        building_structure=row["building_structure"],
        heating_method=row["heating_method"],
        elevator_available=row["elevator_available"],
        transaction_rights=row["transaction_rights"],
        house_usage=row["house_usage"],
        house_age=row["house_age"],
        house_ownership=row["house_ownership"]
    )
    session.add(house)

# 提交事务
session.commit()

# 关闭数据库会话
session.close()
