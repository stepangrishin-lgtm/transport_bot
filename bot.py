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

BOT_TOKEN = "8598055235:AAEcMaVgBkiKYokFXxDd2_govw4ytGp8Rn4"  # <-- вставь сюда токен

SCHEDULE_FILE = "schedule.json"
DB_FILE = "transport.db"

SESSION_TTL = 180  # 3 минуты — срок действия отметки

# Память: последние два сообщения с кнопками
LAST_BUTTON_MESSAGES: List[Tuple[int, int]] = []

# Память: user_id → (pressed_minute, expiry_minute)
PRESSED_SESSIONS: Dict[int, Tuple[int, int]] = {}


bot = Bot(
    BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()


# -------------------------------------------------------------------
# TIME HELPERS
# -------------------------------------------------------------------

def now_minute_of_day() -> int:
    """
    Возвращает текущее локальное московское время как минуту с начала суток.
    """
    t = datetime.now()
    return t.hour * 60 + t.minute


def human_time_from_minute(m: int) -> str:
    """504 → '08:24'"""
    h = m // 60
    mi = m % 60
    return f"{h:02d}:{mi:02d}"


# -------------------------------------------------------------------
# LOAD SCHEDULE
# -------------------------------------------------------------------

def load_schedule():
    if not os.path.exists(SCHEDULE_FILE):
        raise RuntimeError(f"File '{SCHEDULE_FILE}' not found")
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    stops = data["stops"]
    # превращаем time → минуты
    for s in stops:
        hh, mm = map(int, s["time"].split(":"))
        s["minute"] = hh * 60 + mm
    return stops


SCHEDULE: List[Dict] = load_schedule()


# -------------------------------------------------------------------
# DATABASE
# -------------------------------------------------------------------

def init_db():
    is_new = not os.path.exists(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stop_id INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            user_id INTEGER
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_stop ON events(stop_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_minute ON events(minute)")
    conn.commit()
    conn.close()
    return is_new


def add_event(stop_id: int, minute: int, user_id: Optional[int]):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (stop_id, minute, user_id) VALUES (?, ?, ?)",
        (stop_id, minute, user_id),
    )
    conn.commit()
    conn.close()


def get_today_events() -> List[Tuple[int, int]]:
    """
    Возвращает список (stop_id, minute_of_day) сегодняшних событий.
    Поскольку храним только минуты — достаточно фильтровать >= 0.
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT stop_id, minute FROM events ORDER BY minute ASC"
    )
    rows = cur.fetchall()
    conn.close()
    return [(int(sid), int(m)) for sid, m in rows]


def get_events_by_stop_today() -> Dict[int, List[int]]:
    events = get_today_events()
    d: Dict[int, List[int]] = {}
    for sid, m in events:
        d.setdefault(sid, []).append(m)
    return d


# -------------------------------------------------------------------
# UI HELPERS
# -------------------------------------------------------------------

def main_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📍 Где автобус?", callback_data="where")
    kb.button(text="🚌 Отметить прибытие", callback_data="press")
    kb.adjust(1)
    return kb.as_markup()


async def register_buttons_message(chat_id: int, message_id: int):
    """
    Оставляем клавиатуры только у последних двух сообщений.
    """
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
# WEIGHTED OFFSET (variant B)
# -------------------------------------------------------------------

def compute_weighted_offset_minutes() -> Tuple[float, int]:
    """
    Взвешенное среднее смещение времени (fact - plan).
    Последние отметки имеют больший вес.

    Возвращает:
      offset_minutes (float),
      total_reports
    """
    events_by_stop = get_events_by_stop_today()
    flat: List[Tuple[int, int]] = []  # (minute, stop_id)

    for sid, minutes in events_by_stop.items():
        for m in minutes:
            flat.append((m, sid))

    if not flat:
        return 0.0, 0

    flat.sort(key=lambda x: x[0])

    total = 0.0
    weight_sum = 0.0
    n = len(flat)

    # stop_id -> плановая минута
    plan = {s["id"]: s["minute"] for s in SCHEDULE}

    for idx, (fact_minute, sid) in enumerate(flat, start=1):
        weight = idx
        plan_minute = plan.get(sid)
        if plan_minute is None:
            continue
        delta = fact_minute - plan_minute
        total += delta * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0, n

    return total / weight_sum, n


# -------------------------------------------------------------------
# ETA WINDOW (5 STOPS)
# -------------------------------------------------------------------

def build_eta_window() -> Tuple[List[Dict], int, str]:
    offset, count = compute_weighted_offset_minutes()

    # ETA = план + offset
    eta_map: Dict[int, float] = {}
    for s in SCHEDULE:
        eta_map[s["id"]] = s["minute"] + offset

    # Находим ключевую остановку (ETA ближе всего к текущему времени)
    now_m = now_minute_of_day()
    diffs = [(sid, abs(eta_map[sid] - now_m)) for sid in eta_map]
    diffs.sort(key=lambda x: x[1])

    key_sid = diffs[0][0]

    ids_ordered = [s["id"] for s in SCHEDULE]
    key_index = ids_ordered.index(key_sid)

    start = max(0, key_index - 2)
    end = start + 5
    if end > len(ids_ordered):
        end = len(ids_ordered)
        start = max(0, end - 5)

    chosen_ids = ids_ordered[start:end]

    window = []
    for sid in chosen_ids:
        stop = next(s for s in SCHEDULE if s["id"] == sid)
        eta_min = eta_map[sid]
        window.append({
            "id": sid,
            "name": stop["name"],
            "eta_min": eta_min,
            "eta_str": human_time_from_minute(int(round(eta_min))),
            "is_key": sid == key_sid
        })

    # статус
    avg_offset_min = offset
    if avg_offset_min > 1.5:
        status = f"автобус опаздывает на {int(round(avg_offset_min))} мин."
    elif avg_offset_min < -1.5:
        status = f"автобус спешит на {abs(int(round(avg_offset_min)))} мин."
    else:
        status = "автобус идёт по расписанию"

    # точность (очень простая)
    conf = min(95, 40 + count * 5)

    return window, conf, status


# -------------------------------------------------------------------
# HANDLERS
# -------------------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await answer_with_menu(message, "Привет! Бот транспорта (v0.4). Выберите действие:")


@dp.callback_query(F.data == "where")
async def on_where(callback: CallbackQuery):
    window, conf, status = build_eta_window()
    lines = ["<b>Расчетное время:</b>\n"]

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
    """
    Нажата кнопка «Отметить прибытие».
    Фиксируем время нажатия в минутах.
    """
    now_m = now_minute_of_day()
    expiry_m = now_m + (SESSION_TTL // 60) + 1

    PRESSED_SESSIONS[callback.from_user.id] = (now_m, expiry_m)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=s["name"], callback_data=f"stop_{s['id']}")]
            for s in SCHEDULE
        ]
    )

    human = human_time_from_minute(now_m)
    msg = await callback.message.answer(
        f"Отметка времени зафиксирована: <b>{human}</b>\nВыберите остановку:",
        reply_markup=kb,
    )
    await register_buttons_message(msg.chat.id, msg.message_id)
    await callback.answer()


@dp.callback_query(F.data.startswith("stop_"))
async def on_stop(callback: CallbackQuery):
    user_id = callback.from_user.id
    session = PRESSED_SESSIONS.get(user_id)

    if not session:
        await callback_answer_with_menu(
            callback,
            "Сессия истекла. Нажмите «🚌 Отметить прибытие» ещё раз."
        )
        await callback.answer()
        return

    pressed_m, expiry_m = session
    now_m = now_minute_of_day()
    if now_m > expiry_m:
        PRESSED_SESSIONS.pop(user_id, None)
        await callback_answer_with_menu(
            callback,
            "Сессия устарела. Нажмите «🚌 Отметить прибытие» ещё раз."
        )
        await callback.answer()
        return

    stop_id = int(callback.data.split("_")[1])

    # Сохраняем событие
    add_event(stop_id, pressed_m, user_id)
    PRESSED_SESSIONS.pop(user_id, None)

    # отклонение от расписания
    plan_min = next(s["minute"] for s in SCHEDULE if s["id"] == stop_id)
    delta = pressed_m - plan_min

    stop_name = next(s["name"] for s in SCHEDULE if s["id"] == stop_id)
    human = human_time_from_minute(pressed_m)

    text = (
        f"Спасибо! Автобус отмечен на остановке <b>{stop_name}</b> "
        f"в <b>{human}</b>.\nОтклонение от расписания: <b>{delta:+} мин.</b>"
    )

    await callback_answer_with_menu(callback, text)
    await callback.answer()


# -------------------------------------------------------------------
# START BOT
# -------------------------------------------------------------------

async def main():
    init_db()
    print("Transport bot 0.4 started.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
