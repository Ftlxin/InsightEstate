from database import engine, Base

# 创建所有表（如果不存在）
Base.metadata.create_all(bind=engine)
