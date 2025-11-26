# bot.py (version 0.3) — SQLite storage, weighted average (recent marks have higher weight)
import os
import json
import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

# ---------------- CONFIG ----------------
# Укажи свой токен
BOT_TOKEN = "8598055235:AAEcMaVgBkiKYokFXxDd2_govw4ytGp8Rn4"

# Файлы
SCHEDULE_FILE = "schedule.json"   # файл с расписанием (см. ранее)
STATE_FILE = "state.json"         # для хранения последних сообщений с кнопками
DB_FILE = "transport.db"          # sqlite база для отметок

# Сессии нажатий: user_id -> (pressed_ts, expiry_ts)
PRESSED_SESSIONS: Dict[int, Tuple[int, int]] = {}
SESSION_TTL = 180  # seconds

# aiogram bot init (3.7+)
bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ---------------- Utilities: schedule & state ----------------
def load_schedule() -> List[Dict]:
    if not os.path.exists(SCHEDULE_FILE):
        raise RuntimeError(f"Schedule file '{SCHEDULE_FILE}' not found. Create it (see docs).")
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    stops = data.get("stops", [])
    # add order index if not present
    for idx, s in enumerate(stops):
        s.setdefault("order", idx)
    return stops

def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_state(obj: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------------- DB layer ----------------
def init_db():
    need = not os.path.exists(DB_FILE)
    conn = sqlite3.connect(DB_FILE, isolation_level=None)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stop_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            user_id INTEGER
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_stop ON events(stop_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
    conn.commit()
    conn.close()
    return need

def add_event(stop_id: int, ts: int, user_id: Optional[int] = None):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO events (stop_id, timestamp, user_id) VALUES (?, ?, ?)", (stop_id, int(ts), user_id))
    conn.commit()
    conn.close()

def get_today_events() -> List[Tuple[int,int]]:
    """Return list of (stop_id, timestamp) for today's events (UTC)"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    now = datetime.utcnow()
    start_of_day = datetime(now.year, now.month, now.day, 0, 0, 0)
    start_ts = int(start_of_day.timestamp())
    cur.execute("SELECT stop_id, timestamp FROM events WHERE timestamp >= ?", (start_ts,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_events_by_stop_today() -> Dict[int, List[int]]:
    rows = get_today_events()
    d: Dict[int, List[int]] = {}
    for sid, ts in rows:
        d.setdefault(int(sid), []).append(int(ts))
    return d

# ---------------- UI helpers: main menu & button lifecycle ----------------
def main_menu_builder() -> InlineKeyboardBuilder:
    kb = InlineKeyboardBuilder()
    kb.button(text="📍 Где автобус?", callback_data="where")
    kb.button(text="🚌 Отметить прибытие", callback_data="press")
    kb.adjust(1)
    return kb

async def remove_buttons_async(chat_id: int, message_id: int):
    try:
        await bot.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=None)
    except Exception:
        # ignore if message removed or other error
        pass
    
    async def register_buttons_message_async(chat_id: int, message_id: int):state = load_state()
    last = state.get("last_buttons", [])
    last.append({"chat_id": chat_id, "message_id": message_id})
    # keep only last two; remove reply_markup for older
    while len(last) > 2:
        old = last.pop(0)
        await remove_buttons_async(old["chat_id"], old["message_id"])
    state["last_buttons"] = last
    save_state(state)

# ---------------- Schedule helpers ----------------
SCHEDULE = load_schedule()  # list of dicts with id, name, time ("HH:MM"), optional order

def parse_sched_dt(timestr: str, ref_date: datetime) -> datetime:
    h,m = map(int, timestr.split(":"))
    return datetime(year=ref_date.year, month=ref_date.month, day=ref_date.day, hour=h, minute=m)

def schedule_map_dt() -> Dict[int, datetime]:
    today = datetime.utcnow().date()
    sm = {}
    for s in SCHEDULE:
        sm[s["id"]] = parse_sched_dt(s["time"], today)
    return sm

# ---------------- Weighted average logic (variant B) ----------------
def compute_weighted_offset_seconds() -> Tuple[float, int]:
    """
    Compute global offset (seconds) to apply to schedule.
    Weighted by recency: for events sorted by timestamp ascending,
    weight = 1..n (last event has highest weight).
    Returns (offset_seconds, total_reports)
    Each event's contribution: (event_ts - schedule_ts_of_its_stop) * weight
    """
    events_by_stop = get_events_by_stop_today()  # {stop_id: [ts,...]}
    # flatten events to list of (ts, stop_id)
    flat: List[Tuple[int,int]] = []
    for sid, timestamps in events_by_stop.items():
        for ts in timestamps:
            flat.append((ts, sid))
    if not flat:
        return 0.0, 0
    # sort by ts ascending
    flat.sort(key=lambda x: x[0])
    total = 0.0
    weight_sum = 0.0
    # compute schedule dt map
    sched_map = schedule_map_dt()
    n = len(flat)
    # assign weights 1..n (older smaller, recent larger)
    for idx, (ts, sid) in enumerate(flat, start=1):
        weight = idx  # simple linear weights; last gets n
        event_dt = datetime.utcfromtimestamp(ts)
        sched_dt = sched_map.get(sid)
        if sched_dt is None:
            # if schedule missing, skip
            continue
        delta = (event_dt - sched_dt).total_seconds()
        total += delta * weight
        weight_sum += weight
    if weight_sum == 0:
        return 0.0, len(flat)
    offset = total / weight_sum
    return offset, len(flat)

# ---------------- Confidence metric ----------------
def compute_confidence(report_count: int, latest_event_age_minutes: Optional[float]) -> int:
    # base by number of reports
    if report_count == 0:
        return 30
    pct = min(95, 40 + report_count * 5)  # e.g., 10 reports -> 90
    if latest_event_age_minutes is not None:
        if latest_event_age_minutes > 60:
            pct = int(pct * 0.6)
        elif latest_event_age_minutes > 30:
            pct = int(pct * 0.8)
    return int(pct)

# ---------------- Build ETA and prepare window (5 stops, key 3rd) ----------------
def build_eta_and_window() -> Tuple[List[Dict], int, str]:
    """
    Returns:
      window_list: list of dicts for chosen 5 stops: {id, name, eta_dt, eta_str, is_key}
      confidence_pct
      status_text ("опаздывает", "спешит", "в графике")
    """
    offset_seconds, total_reports = compute_weighted_offset_seconds()
    # compute latest event time for age
    events = get_today_events()
    latest_ts = max([ts for (_,ts) in events]) if events else None
    latest_age_min = None
    if latest_ts:
        latest_age_min = (int(datetime.utcnow().timestamp()) - latest_ts) / 60.0

    conf = compute_confidence(total_reports, latest_age_min)

    # schedule map
    sched_map = schedule_map_dt()
    # build eta map: schedule + offset
    eta_map_dt: Dict[int, datetime] = {}
    for s in SCHEDULE:
        sid = s["id"]
        sched_dt = sched_map[sid]
        eta_dt = sched_dt + timedelta(seconds=offset_seconds)
        eta_map_dt[sid] = eta_dt

    # determine key stop (closest ETA to now)
    ref_ts = datetime.utcnow()
    diffs = [(sid, abs((eta_map_dt[sid] - ref_ts).total_seconds())) for sid in eta_map_dt.keys()]
    diffs_sorted = sorted(diffs, key=lambda x: x[1])
    key_sid = diffs_sorted[0][0]

    # find key index in schedule ordering
    ids_ordered = [s["id"] for s in SCHEDULE]
    idx_key = ids_ordered.index(key_sid)
    start = max(0, idx_key - 2)
    end = start + 5
    if end > len(ids_ordered):
        end = len(ids_ordered)
        start = max(0, end - 5)
    chosen_ids = ids_ordered[start:end]
    window = []
    for i, sid in enumerate(chosen_ids):
        s = next(x for x in SCHEDULE if x["id"] == sid)
        eta_dt = eta_map_dt[sid]
        window.append({
            "id": sid,
            "name": s["name"],
            "eta_dt": eta_dt,
            "eta_str": eta_dt.strftime("%H:%M"),
            "is_key": (sid == key_sid)
        })

    # status_text based on average offset (minutes)
    avg_offset_min = offset_seconds / 60.0
    if avg_offset_min > 1.5:
        status = f"автобус опаздывает на {int(round(avg_offset_min))} мин."
    elif avg_offset_min < -1.5:
        status = f"автобус спешит на {abs(int(round(avg_offset_min)))} мин."
    else:
        status = "автобус идёт по расписанию"

    return window, conf, status

# ---------------- Handlers ----------------
@dp.message(Command("start"))
async def cmd_start(message: Message):
    kb = main_menu_builder()
    msg = await message.answer("Привет! Бот транспорта (v0.3). Выберите действие:", reply_markup=kb.as_markup())
    await register_buttons_message_async(msg.chat.id, msg.message_id)

@dp.callback_query(F.data == "where")
async def cb_where(callback: CallbackQuery):
    window, conf, status = build_eta_and_window()
    lines = ["<b>Расчетное время:</b>\n"]
    for i, it in enumerate(window):
        text = f"{it['name']} — {it['eta_str']}"
        if it["is_key"]:
            text = f"➡️ <b>{it['name']} — {it['eta_str']}</b>"
        lines.append(text)
    lines.append("")
    lines.append(f"Точность прогноза: {conf}%")
    lines.append(f"Ситуация: {status}")

    kb = main_menu_builder()
    msg = await callback.message.answer("\n".join(lines), reply_markup=kb.as_markup())
    await register_buttons_message_async(msg.chat.id, msg.message_id)
    await callback.answer()

@dp.callback_query(F.data == "press")
async def cb_press(callback: CallbackQuery):
    # store pressed time now (UTC)
    now_ts = int(datetime.utcnow().timestamp())
    uid = callback.from_user.id
    expiry = now_ts + SESSION_TTL
    PRESSED_SESSIONS[uid] = (now_ts, expiry)
    # show stops keyboard
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=s["name"], callback_data=f"stop_{s['id']}")] for s in SCHEDULE])
    msg = await callback.message.answer(f"Отметка времени зафиксирована: <b>{datetime.utcfromtimestamp(now_ts).strftime('%H:%M:%S')}</b>\nВыберите остановку:", reply_markup=kb)
    await register_buttons_message_async(msg.chat.id, msg.message_id)
    await callback.answer()

@dp.callback_query(F.data.startswith("stop_"))
async def cb_stop(callback: CallbackQuery):
    uid = callback.from_user.id
    session = PRESSED_SESSIONS.get(uid)
    if not session:
        kb = main_menu_builder()
        msg = await callback.message.answer("Сессия истекла или отсутствует. Нажмите «🚌 Отметить прибытие» и повторите.", reply_markup=kb.as_markup())
        await register_buttons_message_async(msg.chat.id, msg.message_id)
        await callback.answer()
        return
    pressed_ts, expiry = session
    if int(datetime.utcnow().timestamp()) > expiry:
        PRESSED_SESSIONS.pop(uid, None)
        kb = main_menu_builder()
        msg = await callback.message.answer("Сессия устарела. Пожалуйста, нажмите «🚌 Отметить прибытие» ещё раз.", reply_markup=kb.as_markup())
        await register_buttons_message_async(msg.chat.id, msg.message_id)
        await callback.answer()
        return

    # use pressed_ts as the event time
    stop_id = int(callback.data.split("_",1)[1])
    add_event(stop_id, pressed_ts, uid)
    # cleanup session
    PRESSED_SESSIONS.pop(uid, None)

    # human time in local-ish format (UTC shown)
    human_time = datetime.utcfromtimestamp(pressed_ts).strftime("%H:%M:%S")
    # compute deviation vs schedule for that stop
    sched_map = schedule_map_dt()
    sched_dt = sched_map.get(stop_id)
    delta_min = int(round((pressed_ts - int(sched_dt.timestamp()))/60.0)) if sched_dt else 0

    # Save convenience last state in state.json for UI (optional)
    st = load_state()
    st["last_stop"] = stop_id
    st["last_time"] = pressed_ts
    save_state(st)

    # Friendly message (like v0.1)
    stop_name = next(s["name"] for s in SCHEDULE if s["id"] == stop_id)
    text = f"Спасибо! Автобус отмечен на остановке <b>{stop_name}</b> в <b>{human_time}</b>.\nОтклонение от расписания: <b>{delta_min:+} мин.</b>"

    kb = main_menu_builder()
    msg = await callback.message.answer(text, reply_markup=kb.as_markup())
    await register_buttons_message_async(msg.chat.id, msg.message_id)
    await callback.answer()

# ---------------- Startup ----------------
if __name__ == "__main__":
    init_db()
    # ensure state file exists
    if not os.path.exists(STATE_FILE):
        save_state({})
    print("Transport bot v0.3 started.")
    asyncio.run(dp.start_polling(bot))