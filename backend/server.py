from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, date

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class Category(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    weight_percentage: float
    color: str = "#3b82f6"
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CategoryCreate(BaseModel):
    name: str
    weight_percentage: float
    color: str = "#3b82f6"

class Habit(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category_id: str
    target_time: Optional[str] = None
    description: Optional[str] = None
    is_completed: bool = False
    streak_count: int = 0
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HabitCreate(BaseModel):
    name: str
    category_id: str
    target_time: Optional[str] = None
    description: Optional[str] = None

class HabitUpdate(BaseModel):
    is_completed: bool
    miss_reason: Optional[str] = None

class DailyRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: str
    habits_completed: List[str] = []
    daily_notes: Optional[str] = None
    category_percentages: dict = {}
    overall_percentage: float = 0.0
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DailyRecordCreate(BaseModel):
    date: str
    daily_notes: Optional[str] = None

class Settings(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    notification_enabled: bool = True
    dark_mode: bool = False
    reminder_times: dict = {}

# Helper functions
def prepare_for_mongo(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
    return data

def parse_from_mongo(item):
    if isinstance(item.get('created_date'), str):
        item['created_date'] = datetime.fromisoformat(item['created_date'])
    return item

# Category Routes
@api_router.post("/categories", response_model=Category)
async def create_category(category_data: CategoryCreate):
    category = Category(**category_data.dict())
    category_dict = prepare_for_mongo(category.dict())
    await db.categories.insert_one(category_dict)
    return category

@api_router.get("/categories", response_model=List[Category])
async def get_categories():
    categories = await db.categories.find().to_list(100)
    return [Category(**parse_from_mongo(cat)) for cat in categories]

@api_router.get("/categories/{category_id}", response_model=Category)
async def get_category(category_id: str):
    category = await db.categories.find_one({"id": category_id})
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return Category(**parse_from_mongo(category))

@api_router.put("/categories/{category_id}", response_model=Category)
async def update_category(category_id: str, category_data: CategoryCreate):
    updated_data = prepare_for_mongo(category_data.dict())
    result = await db.categories.update_one(
        {"id": category_id}, 
        {"$set": updated_data}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    
    category = await db.categories.find_one({"id": category_id})
    return Category(**parse_from_mongo(category))

@api_router.delete("/categories/{category_id}")
async def delete_category(category_id: str):
    # Delete associated habits first
    await db.habits.delete_many({"category_id": category_id})
    result = await db.categories.delete_one({"id": category_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"message": "Category deleted successfully"}

# Habit Routes
@api_router.post("/habits", response_model=Habit)
async def create_habit(habit_data: HabitCreate):
    habit = Habit(**habit_data.dict())
    habit_dict = prepare_for_mongo(habit.dict())
    await db.habits.insert_one(habit_dict)
    return habit

@api_router.get("/habits", response_model=List[Habit])
async def get_habits():
    habits = await db.habits.find().to_list(1000)
    return [Habit(**parse_from_mongo(habit)) for habit in habits]

@api_router.get("/habits/category/{category_id}", response_model=List[Habit])
async def get_habits_by_category(category_id: str):
    habits = await db.habits.find({"category_id": category_id}).to_list(1000)
    return [Habit(**parse_from_mongo(habit)) for habit in habits]

@api_router.put("/habits/{habit_id}", response_model=Habit)
async def update_habit_completion(habit_id: str, habit_update: HabitUpdate):
    # Update streak if completing habit
    update_data = {"is_completed": habit_update.is_completed}
    
    if habit_update.is_completed:
        # Increment streak
        await db.habits.update_one(
            {"id": habit_id}, 
            {"$inc": {"streak_count": 1}}
        )
    elif habit_update.miss_reason:
        # Reset streak and store miss reason
        update_data["streak_count"] = 0
        update_data["miss_reason"] = habit_update.miss_reason
    
    result = await db.habits.update_one(
        {"id": habit_id}, 
        {"$set": update_data}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Habit not found")
    
    habit = await db.habits.find_one({"id": habit_id})
    return Habit(**parse_from_mongo(habit))

@api_router.delete("/habits/{habit_id}")
async def delete_habit(habit_id: str):
    result = await db.habits.delete_one({"id": habit_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Habit not found")
    return {"message": "Habit deleted successfully"}

# Daily Record Routes
@api_router.get("/daily-record/{date}")
async def get_daily_record(date: str):
    record = await db.daily_records.find_one({"date": date})
    if not record:
        # Create new record for the date
        new_record = DailyRecord(date=date)
        record_dict = prepare_for_mongo(new_record.dict())
        await db.daily_records.insert_one(record_dict)
        return new_record
    return DailyRecord(**parse_from_mongo(record))

@api_router.put("/daily-record/{date}")
async def update_daily_record(date: str, record_data: DailyRecordCreate):
    # Calculate productivity based on completed habits
    habits = await db.habits.find().to_list(1000)
    categories = await db.categories.find().to_list(100)
    
    # Get completed habits for today
    completed_habits = [h for h in habits if h.get('is_completed', False)]
    completed_habit_ids = [h['id'] for h in completed_habits]
    
    # Calculate category percentages
    category_percentages = {}
    overall_percentage = 0.0
    
    for category in categories:
        cat_habits = [h for h in habits if h['category_id'] == category['id']]
        completed_cat_habits = [h for h in completed_habits if h['category_id'] == category['id']]
        
        if len(cat_habits) > 0:
            cat_percentage = (len(completed_cat_habits) / len(cat_habits)) * 100
            category_percentages[category['id']] = cat_percentage
            overall_percentage += cat_percentage * (category['weight_percentage'] / 100)
    
    update_data = {
        "habits_completed": completed_habit_ids,
        "category_percentages": category_percentages,
        "overall_percentage": round(overall_percentage, 1),
        "daily_notes": record_data.daily_notes
    }
    
    result = await db.daily_records.update_one(
        {"date": date}, 
        {"$set": update_data}, 
        upsert=True
    )
    
    # Return updated record
    record = await db.daily_records.find_one({"date": date})
    return DailyRecord(**parse_from_mongo(record))

# Dashboard Route
@api_router.get("/dashboard/{date}")
async def get_dashboard_data(date: str):
    # Get today's habits
    habits_raw = await db.habits.find().to_list(1000)
    categories_raw = await db.categories.find().to_list(100)
    daily_record = await db.daily_records.find_one({"date": date})
    
    # Parse data from MongoDB
    habits = [Habit(**parse_from_mongo(h)).dict() for h in habits_raw]
    categories = [Category(**parse_from_mongo(c)).dict() for c in categories_raw]
    
    if not daily_record:
        daily_record = DailyRecord(date=date).dict()
    else:
        daily_record = DailyRecord(**parse_from_mongo(daily_record)).dict()
    
    # Group habits by category
    habits_by_category = {}
    for category in categories:
        cat_habits = [h for h in habits if h['category_id'] == category['id']]
        habits_by_category[category['id']] = {
            "category": category,
            "habits": cat_habits
        }
    
    return {
        "date": date,
        "habits_by_category": habits_by_category,
        "daily_record": daily_record,
        "summary": {
            "total_habits": len(habits),
            "completed_habits": len([h for h in habits if h.get('is_completed', False)]),
            "overall_percentage": daily_record.get('overall_percentage', 0.0)
        }
    }

# Settings Routes
@api_router.get("/settings")
async def get_settings():
    settings = await db.settings.find_one()
    if not settings:
        # Create default settings
        default_settings = Settings()
        settings_dict = prepare_for_mongo(default_settings.dict())
        await db.settings.insert_one(settings_dict)
        return default_settings
    return Settings(**parse_from_mongo(settings))

@api_router.put("/settings")
async def update_settings(settings_data: Settings):
    settings_dict = prepare_for_mongo(settings_data.dict())
    result = await db.settings.update_one(
        {}, 
        {"$set": settings_dict}, 
        upsert=True
    )
    return settings_data

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()