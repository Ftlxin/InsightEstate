from fastapi import FastAPI
from Routers.ValuationReport import router as valuation_router


app = FastAPI()

# 绑定路由
app.include_router(valuation_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "欢迎来到 FastAPI 项目"}
