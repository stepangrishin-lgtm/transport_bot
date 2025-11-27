import os
import json
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

BOT_TOKEN = "8598055235:AAEcMaVgBkiKYokFXxDd2_govw4ytGp8Rn4"  # <<< ВСТАВЬ СВОЙ ТОКЕН

SCHEDULE_FILE = "schedule.json"
DB_FILE = "transport.db"

SESSION_TTL = 180  # 3 минуты
MAX_DELTA_MIN = 60
MIN_SEGMENT_MIN = 1
EMA_ALPHA = 0.5

# Фоновая проверка смены суток: каждые 3 часа
MIDNIGHT_CHECK_INTERVAL = 10800   # 3 часа = 10800 секунд

LAST_BUTTON_MESSAGES: List[Tuple[int, int]] = []
PRESSED_SESSIONS: Dict[int, Tuple[int, int, str]] = {}

# Бот
bot = Bot(
    BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()


# -------------------------------------------------------------------
# TIME HELPERS
# -------------------------------------------------------------------

def now_local() -> datetime:
    return datetime.now()


def today_str() -> str:
    return now_local().strftime("%Y-%m-%d")


def minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def now_minute_of_day() -> int:
    return minute_of_day(now_local())


def human_time_from_minute(m: int) -> str:
    return f"{m//60:02d}:{m%60:02d}"


# -------------------------------------------------------------------
# SCHEDULE
# -------------------------------------------------------------------

def load_schedule():
    if not os.path.exists(SCHEDULE_FILE):
        raise RuntimeError("schedule.json not found")

    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    stops = data["stops"]
    for s in stops:
        s["id"] = int(s["id"])
        hh, mm = map(int, s["time"].split(":"))
        s["minute"] = hh * 60 + mm
    return stops


SCHEDULE = load_schedule()


# -------------------------------------------------------------------
# DATABASE
# -------------------------------------------------------------------

def init_db():
    fresh = not os.path.exists(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT NOT NULL,
            stop_id INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            user_id INTEGER
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_day ON events(day)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_day_stop ON events(day, stop_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_day_minute ON events(day, minute)")

    conn.commit()
    conn.close()
    return fresh


def add_event(day: str, stop_id: int, minute: int, user_id: Optional[int]):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (day, stop_id, minute, user_id) VALUES (?, ?, ?, ?)",
        (day, stop_id, minute, user_id),
    )
    conn.commit()
    conn.close()


def get_today_events() -> List[Tuple[int, int]]:
    day = today_str()
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT stop_id, minute FROM events WHERE day = ? ORDER BY minute ASC",
        (day,),
    )
    rows = cur.fetchall()
    conn.close()
    return [(int(sid), int(m)) for sid, m in rows]


def get_events_by_stop_today() -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for sid, m in get_today_events():
        out.setdefault(sid, []).append(m)
    return out


# -------------------------------------------------------------------
# UI HELPERS
# -------------------------------------------------------------------

def main_menu():
    kb = InlineKeyboardBuilder()
    kb.button(text="📍 Где автобус?", callback_data="where")
    kb.button(text="🚌 Отметить прибытие", callback_data="press")
    kb.adjust(1)
    return kb.as_markup()


async def register_buttons_message(chat_id: int, message_id: int):
    global LAST_BUTTON_MESSAGES
    LAST_BUTTON_MESSAGES.append((chat_id, message_id))

    while len(LAST_BUTTON_MESSAGES) > 2:
        old_chat, old_msg = LAST_BUTTON_MESSAGES.pop(0)
        try:
            await bot.edit_message_reply_markup(
                chat_id=old_chat,
                message_id=old_msg,
                reply_markup=None
            )
        except:
            pass


async def answer_with_menu(message: Message, text: str):
    msg = await message.answer(text, reply_markup=main_menu())
    await register_buttons_message(msg.chat.id, msg.message_id)


async def callback_answer_with_menu(callback: CallbackQuery, text: str):
    msg = await callback.message.answer(text, reply_markup=main_menu())
    await register_buttons_message(msg.chat.id, msg.message_id)


# -------------------------------------------------------------------
# COMPUTATION CORE (0.5 LOGIC)
# -------------------------------------------------------------------

def compute_clean_means_by_stop():
    """
    Фильтрация выбросов + среднее по каждой остановке.
    """
    events = get_events_by_stop_today()
    plan = {s["id"]: s["minute"] for s in SCHEDULE}
    means = {}
    total_used = 0
    latest_minute = None
    latest_stop = None

    for sid, mins in events.items():
        if sid not in plan:
            continue
        pm = plan[sid]
        # фильтр по MAX_DELTA_MIN
        filtered = [m for m in mins if abs(m - pm) <= MAX_DELTA_MIN]
        if not filtered:
            continue

        avg = sum(filtered) / len(filtered)
        means[sid] = avg
        total_used += len(filtered)

        for m in filtered:
            if latest_minute is None or m > latest_minute:
                latest_minute = m
                latest_stop = sid

    return means, total_used, latest_minute, latest_stop


def build_eta_with_segments_and_ema():
    """
    Сегментная модель движения + EMA сглаживание.
    """
    means, total_used, latest_minute, latest_stop = compute_clean_means_by_stop()
    plan = {s["id"]: s["minute"] for s in SCHEDULE}
    ids = [s["id"] for s in SCHEDULE]

    if not means:
        return {sid: float(plan[sid]) for sid in ids}, 40, "нет данных — автобус по расписанию", None, None

    # Confidence
    conf = min(95, 40 + total_used * 5)
    if latest_minute is not None:
        age = now_minute_of_day() - latest_minute
        if age > 60:
            conf = int(conf * 0.6)
        elif age > 30:
            conf = int(conf * 0.8)

    # raw ETA
    eta_raw = {sid: means[sid] for sid in means}

    # расписание сегментов
    seg_plan = {}
    for a, b in zip(ids[:-1], ids[1:]):
        d = plan[b] - plan[a]
        seg_plan[(a, b)] = max(MIN_SEGMENT_MIN, d)

    # фактические сегменты
    seg_fact = {}
    for a, b in zip(ids[:-1], ids[1:]):
        if a in means and b in means:
            diff = means[b] - means[a]
            if diff >= MIN_SEGMENT_MIN:
                seg_fact[(a, b)] = int(round(diff))

    # propagate ETA
    changed = True
    while changed:
        changed = False
        for a, b in zip(ids[:-1], ids[1:]):
            if a in eta_raw and b not in eta_raw:
                seg = seg_fact.get((a, b), seg_plan[(a, b)])
                eta_raw[b] = eta_raw[a] + seg
                changed = True
            if b in eta_raw and a not in eta_raw:
                seg = seg_fact.get((a, b), seg_plan[(a, b)])
                eta_raw[a] = eta_raw[b] - seg
                changed = True

    # fallback
    for sid in ids:
        if sid not in eta_raw:
            eta_raw[sid] = float(plan[sid])

    # EMA сглаживание
    offsets = {sid: eta_raw[sid] - plan[sid] for sid in ids}
    ema_offsets = {}
    ema = None
    for sid in ids:
        if ema is None:
            ema = offsets[sid]
        else:
            ema = EMA_ALPHA * offsets[sid] + (1 - EMA_ALPHA) * ema
        ema_offsets[sid] = ema

    eta_final = {sid: plan[sid] + ema_offsets[sid] for sid in ids}

    avg_off = sum(offsets.values()) / len(offsets)
    if avg_off > 1.5:
        status = f"автобус опаздывает на {int(round(avg_off))} мин."
    elif avg_off < -1.5:
        status = f"автобус спешит на {abs(int(round(avg_off)))} мин."
    else:
        status = "автобус идёт по расписанию"

    return eta_final, conf, status, latest_minute, latest_stop


def build_eta_window():
    eta_map, conf, status, latest_minute, latest_stop = build_eta_with_segments_and_ema()
    ids = [s["id"] for s in SCHEDULE]
    now_m = now_minute_of_day()

    diffs = [(sid, abs(eta_map[sid] - now_m)) for sid in ids]
    diffs.sort(key=lambda x: x[1])
    key_sid = diffs[0][0]

    key_index = ids.index(key_sid)
    start = max(0, key_index - 2)
    end = min(len(ids), start + 5)
    if end - start < 5:
        start = max(0, end - 5)

    chosen = ids[start:end]

    window = []
    for sid in chosen:
        stop = next(s for s in SCHEDULE if s["id"] == sid)
        eta_minute = eta_map[sid]
        window.append({
            "id": sid,
            "name": stop["name"],
            "eta_str": human_time_from_minute(int(round(eta_minute))),
            "is_key": sid == key_sid
        })

    return window, conf, status, latest_minute, latest_stop


# -------------------------------------------------------------------
# AUTORESET AT MIDNIGHT (CHECK EVERY 3 HOURS)
# -------------------------------------------------------------------

LAST_RESET_DAY = today_str()

async def auto_reset_daily():
    global LAST_RESET_DAY
    while True:
        now_day = today_str()

        if now_day != LAST_RESET_DAY:
            # Смена суток → очищаем таблицу
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("DELETE FROM events")
            conn.commit()
            conn.close()

            LAST_RESET_DAY = now_day
            print(f"[AUTO RESET] Database cleared at midnight → {now_day}")

        # Проверка каждые 3 часа
        await asyncio.sleep(MIDNIGHT_CHECK_INTERVAL)


# -------------------------------------------------------------------
# HANDLERS
# -------------------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await answer_with_menu(message, "Привет! Бот транспорта (v0.5.2).\nВыберите действие:")


@dp.callback_query(F.data == "where")
async def on_where(callback: CallbackQuery):
    window, conf, status, latest_minute, latest_stop = build_eta_window()

    lines = []

    if latest_minute is not None:
        stop_name = next(s["name"] for s in SCHEDULE if s["id"] == latest_stop)
        lines.append(f"📍 Последняя отметка: <b>{stop_name}</b> — <b>{human_time_from_minute(latest_minute)}</b>\n")
    else:
        lines.append("Нет отметок за сегодня.\n")

    lines.append("<b>Расчетное время:</b>\n")

    for w in window:
        if w["is_key"]:
            lines.append(f"➡️ <b>{w['name']} — {w['eta_str']}</b>")
        else:
            lines.append(f"{w['name']} — {w['eta_str']}")

    lines.append("")
    lines.append(f"Точность прогноза: {conf}%")
    lines.append(f"Ситуация: {status}")

    await callback_answer_with_menu(callback, "\n".join(lines))
    await callback.answer()


@dp.callback_query(F.data == "press")
async def on_press(callback: CallbackQuery):
    now_m = now_minute_of_day()
    day = today_str()
    expiry_m = now_m + (SESSION_TTL // 60) + 1
    PRESSED_SESSIONS[callback.from_user.id] = (now_m, expiry_m, day)

    # TOP-5 остановок по времени
    diffs = [(s["id"], abs(s["minute"] - now_m)) for s in SCHEDULE]
    diffs.sort(key=lambda x: x[1])
    top_ids = [sid for sid, _ in diffs[:5]]

    kb = InlineKeyboardBuilder()
    for s in SCHEDULE:
        if s["id"] in top_ids:
            kb.button(text=s["name"], callback_data=f"stop_{s['id']}")
    kb.button(text="Показать все остановки", callback_data="all_stops")
    kb.adjust(1)

    msg = await callback.message.answer("Выберите остановку:", reply_markup=kb.as_markup())
    await register_buttons_message(msg.chat.id, msg.message_id)
    await callback.answer()


@dp.callback_query(F.data == "all_stops")
async def on_all_stops(callback: CallbackQuery):
    kb = InlineKeyboardBuilder()
    for s in SCHEDULE:
        kb.button(text=s["name"], callback_data=f"stop_{s['id']}")
    kb.adjust(1)

    msg = await callback.message.answer("Полный список остановок:", reply_markup=kb.as_markup())
    await register_buttons_message(msg.chat.id, msg.message_id)
    await callback.answer()


@dp.callback_query(F.data.startswith("stop_"))
async def on_stop(callback: CallbackQuery):
    user_id = callback.from_user.id
    session = PRESSED_SESSIONS.get(user_id)

    if not session:
        await callback_answer_with_menu(callback, "Сессия истекла. Повторите отметку.")
        await callback.answer()
        return

    pressed_m, expiry_m, day = session
    now_m = now_minute_of_day()

    if now_m > expiry_m:
        PRESSED_SESSIONS.pop(user_id, None)
        await callback_answer_with_menu(callback, "Сессия устарела. Повторите отметку.")
        await callback.answer()
        return

    stop_id = int(callback.data.split("_")[1])
    PRESSED_SESSIONS.pop(user_id, None)

    add_event(day, stop_id, pressed_m, user_id)

    plan_min = next(s["minute"] for s in SCHEDULE if s["id"] == stop_id)
    delta = pressed_m - plan_min

    stop_name = next(s["name"] for s in SCHEDULE if s["id"] == stop_id)
    human = human_time_from_minute(pressed_m)

    text = (
        f"Спасибо! Автобус отмечен на остановке <b>{stop_name}</b> "
        f"в <b>{human}</b>.\n"
        f"Отклонение от расписания: <b>{delta:+} мин.</b>"
    )

    await callback_answer_with_menu(callback, text)
    await callback.answer()


# -------------------------------------------------------------------
# START BOT
# -------------------------------------------------------------------

async def main():
    init_db()

    # запускаем фоновый авто-сброс
    asyncio.create_task(auto_reset_daily())

    print("Transport bot 0.5.2 started.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
