from fastapi import FastAPI
from pydantic import BaseModel
import json
import pandas as pd
from TabularCNP.cDeepFM import CDeepFM, CDeepFM_predict
from TabularCNP.dataloader import cWDLDataset
from typing import Optional
from DataBase.database import engine
from sqlalchemy.orm import Session

app = FastAPI()


class HouseInfo(BaseModel):
    # id: int  # 主键，自增ID
    # avg_transaction: float  # 平均交易额
    # transaction_cycle: float  # 交易周期
    
    #Real Estate Profile
    total_floors: int  # 总楼层数
    #listing_price: float  # 挂牌价格
    year_built: Optional[int] = None  # 建造年份
    building_area: float  # 建筑面积
    # price_adjustment: float  # 价格调整
    # viewing_count: int  # 查看次数
    # followers_count: int  # 关注人数
    # visitors_count: int  # 访问人数
    ladders: int  # 楼梯数量
    houses: int  # 住宅数量
    kitchens: int  # 厨房数量
    bedrooms: int  # 卧室数量
    living_rooms: int  # 客厅数量
    bathrooms: int  # 卫生间数量

    #Geographical
    longitude: Optional[float] = None  # 经度
    latitude: Optional[float] = None  # 纬度
    district_name: Optional[int] = None  # 区域名称
    business_area: Optional[int] = None  # 商圈
    community_name: Optional[int] = None  # 社区名称

    #Building Attributes
    apartment_type: int  # 公寓类型
    building_type: Optional[int] = None  # 建筑类型
    house_orientation: int  # 房屋朝向
    renovation_status: int  # 装修状态
    building_structure: Optional[int] = None  # 建筑结构
    heating_method: Optional[int] = None  # 供暖方式
    elevator_available: bool  # 是否有电梯

    #Transaction and Usage
    transaction_rights: Optional[int] = None  # 交易权利
    house_usage: Optional[int] = None  # 房屋用途
    house_age: Optional[int] = None  # 房屋年龄
    house_ownership: Optional[int] = None  # 房屋产权

    #Listing Information
    listing_year: Optional[int] = None  # 挂牌年份
    listing_month: Optional[int] = None  # 挂牌月份
    listing_day: Optional[int] = None  # 挂牌日期
    #nei_price: Optional[float] = None  # 邻里价格





@app.post("/valuation/")
def predict_valuation(house: HouseInfo):
    # #类别变量字典
    # with open('TabularCNP/sparse_feature_encodings.json', 'r',
    #             encoding='utf-8') as f:
    #     query_dict = json.load(f)
    # print(query_dict)

    """从数据库读取 `houses` 表的所有数据，并转换为 Pandas DataFrame"""
    with Session(engine) as session:  # 使用 SQLAlchemy session
        query = "SELECT * FROM houses"
        all_data = pd.read_sql(query, con=session.bind)  # 直接转换为 DataFrame

    

    #测试数据样本
    houseInfo = HouseInfo(
        longitude=104.103673,
        latitude=30.60889,
        building_area=54,
        total_floors=11,
        #listing_price=74,
        year_built=2010,
        # price_adjustment=1,
        # viewing_count=7,
        # followers_count=31,
        # visitors_count=2409,
        ladders=1,
        houses=4,
        kitchens=1,
        bedrooms=1,
        living_rooms=1,
        bathrooms=1,
        listing_year=2021,
        listing_month=1,
        listing_day=1,
        district_name=2,
        business_area=59,
        community_name=1968,
        apartment_type=1,
        building_type=1,
        house_orientation=1,
        renovation_status=3,
        building_structure=2,
        heating_method=0,
        elevator_available=0,
        transaction_rights=0,
        house_usage=0,
        house_age=1,
        house_ownership=1,
    )

    houseInfo_df = pd.DataFrame([houseInfo.model_dump()])
    print(houseInfo_df)
    #预测
    prediction = CDeepFM_predict(houseInfo_df, all_data)

    # 假设计算了成交均价和成交周期
    transaction_avg_price = prediction #预测结果
    transaction_cycle = 30#目前模型是单输出模型，暂不预测成交周期


    return {
        "estimated_price": transaction_avg_price,
    }
