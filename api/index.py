from fastapi import FastAPI
from mangum import Mangum  # Needed for running FastAPI on Vercel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI on Vercel"}

# Handler to make FastAPI work with serverless environments (like Vercel)
handler = Mangum(app)
