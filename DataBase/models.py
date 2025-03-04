from sqlalchemy import Column, Integer, Float, Boolean
from database import Base

class House(Base):
    __tablename__ = "houses"  

    id = Column(Integer, primary_key=True, autoincrement=True)  # 主键，自增ID
    
    avg_transaction = Column(Float, nullable=False)  # 平均交易额
    transaction_cycle = Column(Float, nullable=False)  # 交易周期
    longitude = Column(Float, nullable=False)  # 经度
    latitude = Column(Float, nullable=False)  # 纬度
    building_area = Column(Float, nullable=False)  # 建筑面积
    total_floors = Column(Integer, nullable=False)  # 总楼层数
    listing_price = Column(Float, nullable=False)  # 挂牌价格
    year_built = Column(Integer, nullable=False)  # 建造年份
    price_adjustment = Column(Float, nullable=False)  # 价格调整
    viewing_count = Column(Integer, nullable=False)  # 查看次数
    followers_count = Column(Integer, nullable=False)  # 关注人数
    visitors_count = Column(Integer, nullable=False)  # 访问人数
    ladders = Column(Integer, nullable=False)  # 楼梯数量
    houses = Column(Integer, nullable=False)  # 住宅数量
    kitchens = Column(Integer, nullable=False)  # 厨房数量
    bedrooms = Column(Integer, nullable=False)  # 卧室数量
    living_rooms = Column(Integer, nullable=False)  # 客厅数量
    bathrooms = Column(Integer, nullable=False)  # 卫生间数量
    listing_year = Column(Integer, nullable=False)  # 挂牌年份
    listing_month = Column(Integer, nullable=False)  # 挂牌月份
    listing_day = Column(Integer, nullable=False)  # 挂牌日期
    nei_price = Column(Float, nullable=False)  # 邻里价格

    district_name = Column(Integer, nullable=False)  # 区域名称
    business_area = Column(Integer, nullable=False)  # 商圈
    community_name = Column(Integer, nullable=False)  # 社区名称
    apartment_type = Column(Integer, nullable=False)  # 公寓类型
    building_type = Column(Integer, nullable=False)  # 建筑类型
    house_orientation = Column(Integer, nullable=False)  # 房屋朝向
    renovation_status = Column(Integer, nullable=False)  # 装修状态
    building_structure = Column(Integer, nullable=False)  # 建筑结构
    heating_method = Column(Integer, nullable=False)  # 供暖方式
    elevator_available = Column(Boolean, nullable=False)  # 是否有电梯
    transaction_rights = Column(Integer, nullable=False)  # 交易权利
    house_usage = Column(Integer, nullable=False)  # 房屋用途
    house_age = Column(Integer, nullable=False)  # 房屋年龄
    house_ownership = Column(Integer, nullable=False)  # 房屋产权
