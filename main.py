from fastapi import FastAPI, HTTPException, Path, Request, Query, Body
from databases import Database
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import sqlalchemy
from passlib.hash import bcrypt
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func , desc
from enum import Enum
from asyncpg.exceptions import UniqueViolationError
from transformers import pipeline
from textblob import TextBlob
from typing import Dict, Set, List, Optional

DATABASE_URL = "postgresql+asyncpg://postgres:affan@localhost:5432/mindcare"
database = Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()
announcements: List[Dict] = []
seen: Dict[int, Set[int]] = {}  # {user_id: set([announcement_ids])}

class UserRole(str, Enum):
    user = "user"
    counselor = "counselor"
    admin = "admin"

class SessionStatus(str, Enum):
    pending = "pending"
    completed = "completed"
    cancelled = "cancelled"
    confirmed = "confirmed"

class ReportIn(BaseModel):
    user_id: int
    user_name: str
    type: str
    description: str
    date: str
    status: Optional[str] = "pending"

class ReportOut(BaseModel):
    id: int
    user_id: int
    user_name: str
    type: str
    description: str
    date: str
    status: str

users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(length=100)),
    sqlalchemy.Column("email", sqlalchemy.String(length=100), unique=True, index=True),
    sqlalchemy.Column("password", sqlalchemy.String(length=255)),
    sqlalchemy.Column("role", sqlalchemy.String(length=20), index=True),
    sqlalchemy.Column("gender", sqlalchemy.String(length=20), nullable=True),
    sqlalchemy.Column("phone_number", sqlalchemy.String(length=20), nullable=True),
    sqlalchemy.Column("is_active", sqlalchemy.Boolean, default=True),
)

sessions = sqlalchemy.Table(
    "sessions",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, index=True),
    sqlalchemy.Column("counselor_id", sqlalchemy.Integer, index=True),
    sqlalchemy.Column("scheduled_at", sqlalchemy.DateTime),
    sqlalchemy.Column("status", sqlalchemy.String(length=20), index=True)
)

mood_logs = sqlalchemy.Table(
    "mood_logs",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, index=True),
    sqlalchemy.Column("mood", sqlalchemy.String(length=20)),
    sqlalchemy.Column("note", sqlalchemy.Text),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now(), index=True)
)

reminders = sqlalchemy.Table(
    "reminders",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, index=True),
    sqlalchemy.Column("date", sqlalchemy.String(length=12)),
    sqlalchemy.Column("time", sqlalchemy.String(length=8)),
    sqlalchemy.Column("text", sqlalchemy.String(length=120)),
)

reports = sqlalchemy.Table(
    "reports",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("user_name", sqlalchemy.String(length=100)),
    sqlalchemy.Column("type", sqlalchemy.String(length=30)),  # e.g. "Spam", "Abuse", etc.
    sqlalchemy.Column("description", sqlalchemy.String(length=300)),
    sqlalchemy.Column("date", sqlalchemy.String(length=12)),  # Or Date if you want (string for simplicity)
    sqlalchemy.Column("status", sqlalchemy.String(length=15), default="pending"),  # "pending", "resolved", "ignored"
)

announcements = sqlalchemy.Table(
    "announcements",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("title", sqlalchemy.String(length=200)),
    sqlalchemy.Column("message", sqlalchemy.Text),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now()),
    sqlalchemy.Column("sender_id", sqlalchemy.Integer, nullable=False),
)


engine = sqlalchemy.create_engine(str(DATABASE_URL).replace("+asyncpg", ""))
metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="frontend")

class UserIn(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.user
    gender: Optional[str] = None
    phone_number: Optional[str] = None

class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: UserRole
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    is_active: Optional[bool] = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class SessionIn(BaseModel):
    user_id: int
    counselor_id: int
    scheduled_at: datetime
    status: SessionStatus = SessionStatus.pending

class SessionOut(BaseModel):
    id: int
    user_id: int
    counselor_id: int
    scheduled_at: datetime
    status: SessionStatus

class MoodLogIn(BaseModel):
    user_id: int
    mood: str
    note: Optional[str] = None
    date: Optional[datetime] = None  # Optional: allow custom date

class MoodLogOut(BaseModel):
    id: int
    user_id: int
    mood: str
    note: Optional[str]
    created_at: datetime

class ReminderIn(BaseModel):
    user_id: int
    date: str
    time: str
    text: str

class ReminderOut(BaseModel):
    id: int
    user_id: int
    date: str
    time: str
    text: str

class SessionBooking(BaseModel):
    user_id: int
    counselor_id: int
    scheduled_at: datetime  # ISO datetime string
    status: SessionStatus = SessionStatus.pending
class AnnouncementIn(BaseModel):
    title: str
    message: str
    sender_id: Optional[int] = None  # Optional, remove if not needed

class AnnouncementOut(BaseModel):
    id: int
    title: str
    message: str
    created_at: datetime
    sender_id: Optional[int] = None

class AnnouncementCreate(BaseModel):
    title: Optional[str] = "Announcement"
    message: str
    sender_id: int


class SeenRequest(BaseModel):
    user_id: int
    announcement_id: int
    
def row_to_userout(row):
    d = dict(row)
    if d.get("phone_number") is not None:
        d["phone_number"] = str(d["phone_number"])
    return d

@app.get("/", tags=["Root"])
def root():
    return {"message": "🚀 FastAPI is running!"}

@app.post("/users/", response_model=UserOut, tags=["Users"])
async def create_user(user: UserIn):
    hashed_password = bcrypt.hash(user.password)
    query = users.insert().values(
        name=user.name,
        email=user.email,
        password=hashed_password,
        role=user.role.value,
        gender=user.gender,
        phone_number=user.phone_number
    )
    try:
        user_id = await database.execute(query)
        return {**user.dict(exclude={"password"}), "id": user_id}
    except UniqueViolationError:
        raise HTTPException(status_code=400, detail="Email already registered.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User creation failed: {str(e)}")

@app.get("/moods/summary/", tags=["moods"])
async def moods_summary(user_id: Optional[int] = 1):
    """
    Returns { "recent": [list of logs] } for the given user,
    matching what your frontend expects.
    """
    query = (
        select(mood_logs)
        .where(mood_logs.c.user_id == user_id)
        .order_by(desc(mood_logs.c.created_at))
        .limit(20)
    )
    logs = await database.fetch_all(query)
    # Convert Row objects to dicts
    logs_dicts = [dict(log) for log in logs]
    return {"recent": logs_dicts}


@app.post("/login/", tags=["Users"])
async def login(user: UserLogin):
    query = users.select().where(users.c.email == user.email)
    result = await database.fetch_one(query)
    if result is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not bcrypt.verify(user.password, result["password"]):
        raise HTTPException(status_code=401, detail="Incorrect password")
    return {
        "message": "Login successful",
        "user": {
            "id": result["id"],
            "name": result["name"],
            "email": result["email"],
            "role": result["role"]
        }
    }

@app.get("/users/", response_model=List[UserOut], tags=["Users"])
async def get_users():
    query = users.select()
    rows = await database.fetch_all(query)
    return [row_to_userout(row) for row in rows]

@app.get("/users/{user_id}", response_model=UserOut, tags=["Users"])
async def get_user(user_id: int = Path(..., gt=0)):
    query = users.select().where(users.c.id == user_id)
    row = await database.fetch_one(query)
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return row_to_userout(row)

@app.get("/counselors/", response_model=List[UserOut], tags=["Users"])
async def get_counselors():
    query = users.select().where(users.c.role == UserRole.counselor.value)
    rows = await database.fetch_all(query)
    return [row_to_userout(row) for row in rows]

@app.patch("/users/{user_id}/status")
async def update_user_status(user_id: int = Path(...), is_active: bool = Body(...)):
    query = users.update().where(users.c.id == user_id).values(is_active=is_active)
    result = await database.execute(query)
    if result:
        return {"id": user_id, "is_active": is_active}
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.post("/sessions/book/", response_model=SessionOut, tags=["Sessions"])
async def book_session(session: SessionBooking):
    query = sessions.insert().values(
        user_id=session.user_id,
        counselor_id=session.counselor_id,
        scheduled_at=session.scheduled_at,
        status=session.status.value
    )
    try:
        session_id = await database.execute(query)
        return {**session.dict(), "id": session_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to book session: {str(e)}")

@app.get("/sessions/", response_model=List[SessionOut], tags=["Sessions"])
async def get_sessions(limit: int = Query(100, le=100), offset: int = Query(0)):
    query = sessions.select().order_by(sessions.c.scheduled_at.desc()).limit(limit).offset(offset)
    return await database.fetch_all(query)

@app.get("/sessions/user/{user_id}", response_model=List[SessionOut], tags=["Sessions"])
async def get_sessions_by_user(user_id: int = Path(..., gt=0)):
    query = sessions.select().where(sessions.c.user_id == user_id).order_by(sessions.c.scheduled_at.desc())
    return await database.fetch_all(query)

@app.get("/sessions/counselor/{counselor_id}", response_model=List[SessionOut], tags=["Sessions"])
async def get_sessions_for_counselor(counselor_id: int = Path(..., gt=0)):
    query = sessions.select().where(sessions.c.counselor_id == counselor_id).order_by(sessions.c.scheduled_at.desc())
    return await database.fetch_all(query)

@app.post("/moods/", response_model=MoodLogOut, tags=["Moods"])
async def log_mood(mood: MoodLogIn):
    created = mood.date if mood.date else datetime.utcnow()
    query = mood_logs.insert().values(
        user_id=mood.user_id,
        mood=mood.mood,
        note=mood.note,
        created_at=created
    )
    mood_id = await database.execute(query)
    return {
        "id": mood_id,
        "user_id": mood.user_id,
        "mood": mood.mood,
        "note": mood.note,
        "created_at": created
    }

@app.get("/moods/", response_model=List[MoodLogOut], tags=["Moods"])
async def get_mood_logs(limit: int = Query(100, le=100), offset: int = Query(0)):
    query = mood_logs.select().order_by(mood_logs.c.created_at.desc()).limit(limit).offset(offset)
    return await database.fetch_all(query)

@app.get("/moods/user/{user_id}", response_model=List[MoodLogOut], tags=["Moods"])
async def get_mood_logs_by_user(user_id: int = Path(..., gt=0)):
    query = mood_logs.select().where(mood_logs.c.user_id == user_id).order_by(mood_logs.c.created_at.desc())
    return await database.fetch_all(query)

@app.delete("/moods/", tags=["Moods"])
async def clear_mood_logs():
    query = mood_logs.delete()
    await database.execute(query)
    return {"message": "All moods deleted."}

@app.get("/all-users", response_class=HTMLResponse, tags=["Admin"])
async def all_users_page(request: Request):
    return templates.TemplateResponse("all-users.html", {"request": request})

@app.get("/all-sessions", response_class=HTMLResponse, tags=["Admin"])
async def all_sessions_page(request: Request):
    return templates.TemplateResponse("all-sessions.html", {"request": request})

@app.get("/analytics", response_class=HTMLResponse, tags=["Admin"])
async def analytics_page(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/admin-dashboard-stats/", tags=["Admin"])
async def admin_dashboard_stats():
    users_count_query = select(func.count(users.c.id))
    users_count = await database.fetch_val(users_count_query)

    counselors_count_query = select(func.count(users.c.id)).where(users.c.role == UserRole.counselor.value)
    counselors_count = await database.fetch_val(counselors_count_query)

    sessions_count_query = select(func.count(sessions.c.id))
    sessions_count = await database.fetch_val(sessions_count_query)

    reports_count = 5

    return {
        "users": users_count or 0,
        "counselors": counselors_count or 0,
        "sessions": sessions_count or 0,
        "reports": reports_count
    }

@app.post("/reminders/", response_model=ReminderOut, tags=["Reminders"])
async def create_reminder(rem: ReminderIn):
    query = reminders.insert().values(
        user_id=rem.user_id,
        date=rem.date,
        time=rem.time,
        text=rem.text
    )
    rid = await database.execute(query)
    return {**rem.dict(), "id": rid}

@app.get("/reminders/", response_model=List[ReminderOut], tags=["Reminders"])
async def get_reminders(user_id: int = Query(...)):
    query = reminders.select().where(reminders.c.user_id == user_id).order_by(reminders.c.date, reminders.c.time)
    return await database.fetch_all(query)

@app.delete("/reminders/{reminder_id}/", tags=["Reminders"])
async def delete_reminder(reminder_id: int = Path(..., gt=0)):
    query = reminders.delete().where(reminders.c.id == reminder_id)
    result = await database.execute(query)
    if result:
        return {"message": "Reminder deleted"}
    else:
        raise HTTPException(status_code=404, detail="Reminder not found")

@app.get("/reports/", response_model=List[ReportOut])
async def get_reports():
    query = reports.select().order_by(reports.c.id.desc())
    rows = await database.fetch_all(query)
    return [dict(row) for row in rows]

@app.post("/reports/", response_model=ReportOut)
async def create_report(report: ReportIn):
    query = reports.insert().values(
        user_id=report.user_id,
        user_name=report.user_name,
        type=report.type,
        description=report.description,
        date=report.date,
        status=report.status or "pending"
    )
    new_id = await database.execute(query)
    return {**report.dict(), "id": new_id}

@app.patch("/reports/{report_id}/status")
async def update_report_status(report_id: int, status: str = Body(...)):
    query = reports.update().where(reports.c.id == report_id).values(status=status)
    result = await database.execute(query)
    if result:
        return {"id": report_id, "status": status}
    else:
        raise HTTPException(status_code=404, detail="Report not found")

@app.delete("/reports/{report_id}/")
async def delete_report(report_id: int):
    query = reports.delete().where(reports.c.id == report_id)
    result = await database.execute(query)
    if result:
        return {"message": "Report deleted"}
    else:
        raise HTTPException(status_code=404, detail="Report not found")
    
@app.post("/announcements/", response_model=AnnouncementOut, tags=["Announcements"])
async def create_announcement(announcement: AnnouncementIn):
    try:
        admin_query = users.select().where(users.c.id == announcement.sender_id)
        admin = await database.fetch_one(admin_query)
        if not admin or admin["role"] != UserRole.admin.value:
            raise HTTPException(status_code=403, detail="Only admin can send announcements.")

        query = announcements.insert().values(
            title=announcement.title,
            message=announcement.message,
            sender_id=announcement.sender_id
        )
        ann_id = await database.execute(query)
        row = await database.fetch_one(announcements.select().where(announcements.c.id == ann_id))
        return AnnouncementOut(**dict(row))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

@app.get("/announcements/", response_model=List[AnnouncementOut], tags=["Announcements"])
async def get_announcements(limit: int = Query(10, le=50), offset: int = Query(0)):
    """Get recent announcements for users and counselors"""
    query = announcements.select().order_by(desc(announcements.c.created_at)).limit(limit).offset(offset)
    rows = await database.fetch_all(query)
    return [AnnouncementOut(**dict(row)) for row in rows]

@app.get("/announcements/latest/", response_model=List[AnnouncementOut], tags=["Announcements"])
async def get_latest_announcements(limit: int = Query(3, le=10)):
    """Get latest announcements for notifications"""
    query = announcements.select().order_by(desc(announcements.c.created_at)).limit(limit)
    rows = await database.fetch_all(query)
    return [AnnouncementOut(**dict(row)) for row in rows]

@app.delete("/announcements/{announcement_id}/", tags=["Announcements"])
async def delete_announcement(announcement_id: int = Path(..., gt=0), sender_id: int = Query(...)):
    try:
        admin_query = users.select().where(users.c.id == sender_id)
        admin = await database.fetch_one(admin_query)
        if not admin or admin["role"] != UserRole.admin.value:
            raise HTTPException(status_code=403, detail="Only admin can delete announcements.")

        query = announcements.delete().where(announcements.c.id == announcement_id)
        result = await database.execute(query)
        if result:
            return {"message": "Announcement deleted"}
        else:
            raise HTTPException(status_code=404, detail="Announcement not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
    
    
    
    # --- Summarizer API begins here ---

class Entry(BaseModel):
    text: str

try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print("⚠️ Could not load summarization model:", e)
    summarizer = None

def get_mood(score):
    if score > 0.3:
        return "Positive"
    elif score < -0.3:
        return "Negative"
    return "Neutral"
@app.post("/summarize/")
async def summarize_entry(entry: Entry):
    text = entry.text.strip()

    # Basic fallback if model not available
    if summarizer is None or not text:
        summary = text if text else "No input provided."
    else:
        try:
            # Ensure minimum length to trigger summarization
            if len(text.split()) < 15:
                text += " " + text  # repeat text to allow summarization
            summary = summarizer(
                text, max_length=60, min_length=15, do_sample=False
            )[0]['summary_text']
        except Exception as e:
            summary = f"(Failed to summarize: {e})"

    # Sentiment Analysis
    blob = TextBlob(text)
    sentiment_score = round(blob.sentiment.polarity, 2)
    mood = get_mood(sentiment_score)

    return {
        "summary": summary.strip(),
        "mood": mood,
        "sentiment_score": sentiment_score
    }
# ... (everything above remains unchanged)

# --- Update user PATCH endpoint for full profile editing (including emoji profile_pic) ---
class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    is_active: Optional[bool] = None
    profile_pic: Optional[str] = None  # Accept any emoji or unicode string

@app.patch("/users/{user_id}", response_model=UserOut, tags=["Users"])
async def update_user(user_id: int, user_update: UserUpdate):
    query = users.update().where(users.c.id == user_id).values(**user_update.dict(exclude_unset=True))
    result = await database.execute(query)
    if result:
        row = await database.fetch_one(users.select().where(users.c.id == user_id))
        return row_to_userout(row)
    else:
        raise HTTPException(status_code=404, detail="User not found")

# --- Mood logs: Only users can log moods. Filter out unwanted moods when displaying ---
@app.post("/moods/", response_model=MoodLogOut, tags=["Moods"])
async def log_mood(mood: MoodLogIn):
    user_row = await database.fetch_one(users.select().where(users.c.id == mood.user_id))
    if not user_row or user_row["role"] != UserRole.user.value:
        raise HTTPException(status_code=403, detail="Only users can log mood.")
    created = mood.date if mood.date else datetime.utcnow()
    query = mood_logs.insert().values(
        user_id=mood.user_id,
        mood=mood.mood,
        note=mood.note,
        created_at=created
    )
    mood_id = await database.execute(query)
    return {
        "id": mood_id,
        "user_id": mood.user_id,
        "mood": mood.mood,
        "note": mood.note,
        "created_at": created
    }

@app.get("/moods/user/{user_id}", response_model=List[MoodLogOut], tags=["Moods"])
async def get_mood_logs_by_user(user_id: int = Path(..., gt=0)):
    query = mood_logs.select().where(mood_logs.c.user_id == user_id).order_by(mood_logs.c.created_at.desc())
    results = await database.fetch_all(query)
    # Only include meaningful moods (ignore Custom 4 etc)
    return [dict(row) for row in results if not row["mood"].startswith("Custom")]

@app.get("/moods/", response_model=List[MoodLogOut], tags=["Moods"])
async def get_mood_logs(limit: int = Query(100, le=100), offset: int = Query(0)):
    query = mood_logs.select().order_by(mood_logs.c.created_at.desc()).limit(limit).offset(offset)
    results = await database.fetch_all(query)
    return [dict(row) for row in results if not row["mood"].startswith("Custom")]

@app.get("/moods/summary/", tags=["Moods"])
async def moods_summary(user_id: Optional[int] = 1):
    query = (
        select(mood_logs)
        .where(mood_logs.c.user_id == user_id)
        .order_by(desc(mood_logs.c.created_at))
        .limit(20)
    )
    logs = await database.fetch_all(query)
    logs_dicts = [dict(log) for log in logs if not log["mood"].startswith("Custom")]
    return {"recent": logs_dicts}

# --- Remove mood log feature for counselors in frontend ---
# (No backend changes needed, just filter by role above!)

# --- Reminders: Add recurrence and category ---
class ReminderIn(BaseModel):
    user_id: int
    date: str
    time: str
    text: str
    recurrence: Optional[str] = None  # daily, weekly, monthly, etc.
    category: Optional[str] = None    # e.g. Health, Work, Personal

class ReminderOut(BaseModel):
    id: int
    user_id: int
    date: str
    time: str
    text: str
    recurrence: Optional[str] = None
    category: Optional[str] = None

@app.post("/reminders/", response_model=ReminderOut, tags=["Reminders"])
async def create_reminder(rem: ReminderIn):
    query = reminders.insert().values(
        user_id=rem.user_id,
        date=rem.date,
        time=rem.time,
        text=rem.text,
        recurrence=rem.recurrence,
        category=rem.category
    )
    rid = await database.execute(query)
    return {**rem.dict(), "id": rid}
@app.get("/announcements/", response_model=List[AnnouncementOut], tags=["Announcements"])
async def get_announcements(limit: int = Query(10, le=50), offset: int = Query(0)):
    """Get recent announcements for users and counselors"""
    query = announcements.select().order_by(desc(announcements.c.created_at)).limit(limit).offset(offset)
    rows = await database.fetch_all(query)
    return [AnnouncementOut(**dict(row)) for row in rows]

@app.get("/announcements/latest/", response_model=List[AnnouncementOut], tags=["Announcements"])
async def get_latest_announcements(limit: int = Query(3, le=10)):
    """Get latest announcements for notifications"""
    query = announcements.select().order_by(desc(announcements.c.created_at)).limit(limit)
    rows = await database.fetch_all(query)
    return [AnnouncementOut(**dict(row)) for row in rows]
@app.get("/reminders/", response_model=List[ReminderOut], tags=["Reminders"])
async def get_reminders(user_id: int = Query(...)):
    query = reminders.select().where(reminders.c.user_id == user_id).order_by(reminders.c.date, reminders.c.time)
    rows = await database.fetch_all(query)
    return [dict(row) for row in rows]

# --- Announcements, Reports, Sessions, etc remain unchanged ---
# (Keep your existing endpoints for these unless you want further changes)

# ----------- MODELS -----------



# ----------- ROUTES -----------



