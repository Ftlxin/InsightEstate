from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# MySQL 连接配置
DATABASE_URL = "mysql+pymysql://username:password@localhost:3306/dbname"

# 创建数据库引擎
engine = create_engine(DATABASE_URL, echo=True)

# 创建会话（Session）
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建 ORM 基类
Base = declarative_base()
