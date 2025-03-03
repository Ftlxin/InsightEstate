from fastapi import FastAPI
from pydantic import BaseModel
import json
import pandas as pd
from TabularCNP.cDeepFM import CDeepFM, CDeepFM_predict
from TabularCNP.dataloader import cWDLDataset


app = FastAPI()


class HouseInfo(BaseModel):
    avg_transaction: float  # 平均交易额
    transaction_cycle: float  # 交易周期
    longitude: float  # 经度
    latitude: float  # 纬度
    building_area: float  # 建筑面积
    total_floors: float  # 总楼层数
    listing_price: float  # 挂牌价格
    year_built: float  # 建造年份
    price_adjustment: float  # 价格调整
    viewing_count: float  # 查看次数
    followers_count: float  # 关注人数
    visitors_count: float  # 访问人数
    ladders: float  # 楼梯数量
    houses: float  # 住宅数量
    kitchens: float  # 厨房数量
    bedrooms: float  # 卧室数量
    living_rooms: float  # 客厅数量
    bathrooms: float  # 卫生间数量
    listing_year: float  # 挂牌年份
    listing_month: float  # 挂牌月份
    listing_day: float  # 挂牌日期

    # 下面这些字段可能更适合字符串类型
    district_name: str  # 区域名称
    business_area: str  # 商圈
    community_name: str  # 社区名称
    apartment_type: str  # 公寓类型
    building_type: str  # 建筑类型
    house_orientation: str  # 房屋朝向
    renovation_status: str  # 装修状态
    building_structure: str  # 建筑结构
    heating_method: str  # 供暖方式
    elevator_available: bool  # 是否有电梯
    transaction_rights: str  # 交易权利
    house_usage: str  # 房屋用途
    house_age: float  # 房屋年龄
    house_ownership: str  # 房屋产权


@app.post("/valuation")
def predict_valuation(house: HouseInfo):
    #类别变量字典
    with open('TabularCNP/sparse_feature_encodings.json', 'r',
                encoding='utf-8') as f:
        query_dict = json.load(f)
    print(query_dict)

    #整体数据集
    all = CD.objects.all().values()
    all_data = pd.DataFrame(all)

    #测试数据样本
    houseInfo = HouseInfo(
        longitude=104.103673,
        latitude=30.60889,
        building_area=54,
        total_floors=11,
        listing_price=74,
        year_built=2010,
        price_adjustment=1,
        viewing_count=7,
        followers_count=31,
        visitors_count=2409,
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

    houseInfo_dict = houseInfo.__dict__
    houseInfo_df = pd.DataFrame([houseInfo_dict])
    print(houseInfo_df)
    #预测
    prediction = CDeepFM_predict(houseInfo_df, all_data)

    # 假设计算了成交均价和成交周期
    transaction_avg_price = prediction #预测结果
    transaction_cycle = 30#目前模型是单输出模型，暂不预测成交周期


    return {
        "location": house.location,
        "estimated_price": estimated_price,
        "confidence": confidence,
        "recommendation": "该地区房价稳定，建议持有"
    }
