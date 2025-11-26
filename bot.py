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

# ------------- CONFIG -------------
BOT_TOKEN = "8598055235:AAEcMaVgBkiKYokFXxDd2_govw4ytGp8Rn4"  # <-- сюда вставь свой токен

SCHEDULE_FILE = "schedule.json"
DB_FILE = "transport.db"

SESSION_TTL = 180  # секунды, сколько живёт "сессия нажатия"

# В памяти: последние два сообщения с кнопками
LAST_BUTTON_MESSAGES: List[Tuple[int, int]] = []

# В памяти: user_id -> (pressed_ts, expiry_ts)
PRESSED_SESSIONS: Dict[int, Tuple[int, int]] = {}

bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


# ------------- Время и расписание -------------
def now_local() -> datetime:
    # считаем, что системное время Windows уже в Москве
    return datetime.now()


def load_schedule() -> List[Dict]:
    if not os.path.exists(SCHEDULE_FILE):
        raise RuntimeError(f"Файл расписания '{SCHEDULE_FILE}' не найден.")
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    stops = data.get("stops", [])
    for s in stops:
        s["id"] = int(s["id"])
    return stops


SCHEDULE: List[Dict] = load_schedule()


def schedule_map_dt() -> Dict[int, datetime]:
    """stop_id -> плановое время (сегодня, локальное)."""
    today = now_local().date()
    result: Dict[int, datetime] = {}
    for s in SCHEDULE:
        h, m = map(int, s["time"].split(":"))
        dt = datetime(
            year=today.year,
            month=today.month,
            day=today.day,
            hour=h,
            minute=m,
        )
        result[s["id"]] = dt
    return result


# ------------- БД -------------
def init_db():
    need_new = not os.path.exists(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stop_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            user_id INTEGER
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_stop ON events(stop_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
    conn.commit()
    conn.close()
    return need_new


def add_event(stop_id: int, ts: int, user_id: Optional[int] = None):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (stop_id, timestamp, user_id) VALUES (?, ?, ?)",
        (int(stop_id), int(ts), user_id),
    )
    conn.commit()
    conn.close()


def get_today_events() -> List[Tuple[int, int]]:
    """Список (stop_id, timestamp) за текущий календарный день (по локальному времени)."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    now = now_local()
    start_of_day = datetime(
        year=now.year,
        month=now.month,
        day=now.day,
        hour=0,
        minute=0,
        second=0,
    )
    start_ts = int(start_of_day.timestamp())
    cur.execute(
        "SELECT stop_id, timestamp FROM events WHERE timestamp >= ? ORDER BY timestamp ASC",
        (start_ts,),
    )
    rows = cur.fetchall()
    conn.close()
    return [(int(sid), int(ts)) for sid, ts in rows]


def get_events_by_stop_today() -> Dict[int, List[int]]:
    events = get_today_events()
    d: Dict[int, List[int]] = {}
    for sid, ts in events:
        d.setdefault(sid, []).append(ts)
    return d


# ------------- UI: главное меню + очистка клавиатур -------------
def main_menu_markup() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📍 Где автобус?", callback_data="where")
    kb.button(text="🚌 Отметить прибытие", callback_data="press")
    kb.adjust(1)
    return kb.as_markup()


async def register_buttons_message(chat_id: int, message_id: int):
    """
    Оставляем клавиатуру только у двух последних сообщений.
    """
    global LAST_BUTTON_MESSAGES
    LAST_BUTTON_MESSAGES.append((chat_id, message_id))

    while len(LAST_BUTTON_MESSAGES) > 2:
        old_chat_id, old_msg_id = LAST_BUTTON_MESSAGES.pop(0)
        try:
            await bot.edit_message_reply_markup(
                chat_id=old_chat_id,
                message_id=old_msg_id,
                reply_markup=None,
            )
        except Exception:
            pass


async def send_with_main_menu_from_message(message: Message, text: str):
    msg = await message.answer(text, reply_markup=main_menu_markup())
    await register_buttons_message(msg.chat.id, msg.message_id)


async def send_with_main_menu_from_callback(callback: CallbackQuery, text: str):
    msg = await callback.message.answer(text, reply_markup=main_menu_markup())
    await register_buttons_message(msg.chat.id, msg.message_id)


# ------------- Взвешенное среднее отклонение (вариант B) -------------
def compute_weighted_offset_seconds() -> Tuple[float, int, Optional[float]]:
    """
    Глобальное смещение (секунды) между фактами и расписанием.
    Новые отметки весомее старых.

    Возвращает:
      offset_seconds,
      total_reports,
      latest_event_age_minutes
    """
    events_by_stop = get_events_by_stop_today()
    flat: List[Tuple[int, int]] = []
    for sid, ts_list in events_by_stop.items():
        for ts in ts_list:
            flat.append((int(ts), int(sid)))

    if not flat:
        return 0.0, 0, None

    flat.sort(key=lambda x: x[0])
    sched_map = schedule_map_dt()

    total_weighted_delta = 0.0
    weight_sum = 0.0

    for idx, (ts, sid) in enumerate(flat, start=1):
        weight = idx
        event_dt = datetime.fromtimestamp(ts)
        sched_dt = sched_map.get(sid)
        if sched_dt is None:
            continue
        delta_sec = (event_dt - sched_dt).total_seconds()
        total_weighted_delta += delta_sec * weight
        weight_sum += weight

    if weight_sum == 0:
        offset_sec = 0.0
    else:
        offset_sec = total_weighted_delta / weight_sum

    latest_ts = flat[-1][0]
    latest_age_min = (now_local().timestamp() - latest_ts) / 60.0

    return offset_sec, len(flat), latest_age_min


def compute_confidence(report_count: int, latest_event_age_minutes: Optional[float]) -> int:
    if report_count == 0:
        return 40
    pct = min(95, 40 + report_count * 5)
    if latest_event_age_minutes is not None:
        if latest_event_age_minutes > 60:
            pct = int(pct * 0.6)
        elif latest_event_age_minutes > 30:
            pct = int(pct * 0.8)
    return int(pct)


# ------------- Построение ETA и окна из 5 остановок -------------
def build_eta_window() -> Tuple[List[Dict], int, str]:
    """
    Строим ETA (по расписанию + глобальное смещение),
    выбираем окно из 5 остановок, где ключевая ближе всего к текущему времени.
    """
    offset_sec, report_count, latest_age_min = compute_weighted_offset_seconds()
    confidence = compute_confidence(report_count, latest_age_min)

    sched_map = schedule_map_dt()
    eta_map: Dict[int, datetime] = {}
    for s in SCHEDULE:
        sid = s["id"]
        sched_dt = sched_map[sid]
        eta_map[sid] = sched_dt + timedelta(seconds=offset_sec)

    now = now_local()
    diffs = [(sid, abs((eta - now).total_seconds())) for sid, eta in eta_map.items()]
    diffs.sort(key=lambda x: x[1])
    key_sid = diffs[0][0] if diffs else SCHEDULE[0]["id"]

    ids_ordered = [s["id"] for s in SCHEDULE]
    key_index = ids_ordered.index(key_sid)
    start = max(0, key_index - 2)
    end = start + 5
    if end > len(ids_ordered):
        end = len(ids_ordered)
        start = max(0, end - 5)

    chosen_ids = ids_ordered[start:end]

    window: List[Dict] = []
    for sid in chosen_ids:
        stop = next(s for s in SCHEDULE if s["id"] == sid)
        eta_dt = eta_map[sid]
        window.append(
            {
                "id": sid,"name": stop["name"],
                "eta_dt": eta_dt,
                "eta_str": eta_dt.strftime("%H:%M"),
                "is_key": sid == key_sid,
            }
        )

    avg_offset_min = offset_sec / 60.0
    if avg_offset_min > 1.5:
        status = f"автобус опаздывает на {int(round(avg_offset_min))} мин."
    elif avg_offset_min < -1.5:
        status = f"автобус спешит на {abs(int(round(avg_offset_min)))} мин."
    else:
        status = "автобус идёт по расписанию"

    return window, confidence, status


# ------------- Хэндлеры -------------
@dp.message(Command("start"))
async def on_start(message: Message):
    await send_with_main_menu_from_message(
        message,
        "Привет! Бот транспорта (v0.3.2). Выберите действие:",
    )


@dp.callback_query(F.data == "where")
async def on_where(callback: CallbackQuery):
    window, confidence, status = build_eta_window()

    lines = ["<b>Расчетное время:</b>\n"]
    for item in window:
        if item["is_key"]:
            lines.append(f"➡️ <b>{item['name']} — {item['eta_str']}</b>")
        else:
            lines.append(f"{item['name']} — {item['eta_str']}")

    lines.append("")
    lines.append(f"Точность прогноза: {confidence}%")
    lines.append(f"Ситуация: {status}")

    text = "\n".join(lines)
    await send_with_main_menu_from_callback(callback, text)
    await callback.answer()


@dp.callback_query(F.data == "press")
async def on_press(callback: CallbackQuery):
    """
    Нажата кнопка «Отметить прибытие».
    Фиксируем время нажатия, затем просим выбрать остановку.
    """
    user_id = callback.from_user.id
    now_ts = int(now_local().timestamp())
    expiry_ts = now_ts + SESSION_TTL
    PRESSED_SESSIONS[user_id] = (now_ts, expiry_ts)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=s["name"], callback_data=f"stop_{s['id']}")]
            for s in SCHEDULE
        ]
    )

    human_time = datetime.fromtimestamp(now_ts).strftime("%H:%M:%S")
    msg = await callback.message.answer(
        f"Отметка времени зафиксирована: <b>{human_time}</b>.\nВыберите остановку:",
        reply_markup=kb,
    )
    await register_buttons_message(msg.chat.id, msg.message_id)
    await callback.answer()


@dp.callback_query(F.data.startswith("stop_"))
async def on_stop(callback: CallbackQuery):
    user_id = callback.from_user.id
    session = PRESSED_SESSIONS.get(user_id)
    if not session:
        await send_with_main_menu_from_callback(
            callback,
            "Сессия истекла или отсутствует. Нажмите «🚌 Отметить прибытие» и повторите.",
        )
        await callback.answer()
        return

    pressed_ts, expiry_ts = session
    now_ts = int(now_local().timestamp())
    if now_ts > expiry_ts:
        PRESSED_SESSIONS.pop(user_id, None)
        await send_with_main_menu_from_callback(
            callback,
            "Сессия устарела. Пожалуйста, нажмите «🚌 Отметить прибытие» ещё раз.",
        )
        await callback.answer()
        return

    stop_id = int(callback.data.split("_", 1)[1])

    add_event(stop_id, pressed_ts, user_id)
    PRESSED_SESSIONS.pop(user_id, None)

    sched_map = schedule_map_dt()
    sched_dt = sched_map.get(stop_id)
    if sched_dt is not None:
        delta_min = int(round((pressed_ts - int(sched_dt.timestamp())) / 60.0))
    else:
        delta_min = 0

    stop_name = next(s["name"] for s in SCHEDULE if s["id"] == stop_id)
    human_time = datetime.fromtimestamp(pressed_ts).strftime("%H:%M:%S")

    text = (
        f"Спасибо! Автобус отмечен на остановке <b>{stop_name}</b> "
        f"в <b>{human_time}</b>.\n"
        f"Отклонение от расписания: <b>{delta_min:+} мин.</b>"
    )

    await send_with_main_menu_from_callback(callback, text)
    await callback.answer()


# ------------- Запуск -------------
async def main():
    init_db()
    print("Transport bot v0.3.2 started (local time).")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())