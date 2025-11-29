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
)
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

BOT_TOKEN = "8598055235:AAEcMaVgBkiKYokFXxDd2_govw4ytGp8Rn4"  # <<< ВСТАВЬ СВОЙ ТОКЕН
ADMIN_ID = 331165172  # твой Telegram ID

ROUTES_FILE = "routes.json"
DB_FILE = "transport.db"

SESSION_TTL = 180        # 3 минуты — время жизни сессии отметки
MAX_DELTA_MIN = 60       # максимум допустимого отклонения для «нормальной» отметки
MIN_SEGMENT_MIN = 1      # минимальное время между соседними остановками (минуты)
EMA_ALPHA = 0.5          # коэффициент сглаживания EMA
MIDNIGHT_CHECK_INTERVAL = 10800  # 3 часа, сек

# user_id -> (pressed_m, expiry_m, day, route_id)
PRESSED_SESSIONS: Dict[int, Tuple[int, int, str, str]] = {}

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
    return f"{m // 60:02d}:{m % 60:02d}"


# -------------------------------------------------------------------
# ROUTES / SCHEDULES
# -------------------------------------------------------------------

def load_routes():
    if not os.path.exists(ROUTES_FILE):
        raise RuntimeError("routes.json not found")

    with open(ROUTES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    routes_dict: Dict[str, Dict] = {}
    for r in data.get("routes", []):
        rid = r["id"]
        name = r["name"]
        stops = r["stops"]
        # добавляем поле minute в каждую остановку
        for s in stops:
            t = s["time"]
            hh, mm = map(int, t.split(":"))
            s["minute"] = hh * 60 + mm
            s["id"] = int(s["id"])
        routes_dict[rid] = {
            "id": rid,
            "name": name,
            "stops": stops,
        }
    return routes_dict


ROUTES: Dict[str, Dict] = load_routes()  # route_id -> {id, name, stops}


def get_route(route_id: str) -> Optional[Dict]:
    return ROUTES.get(route_id)


def list_routes_ordered() -> List[Dict]:
    # упорядочим по id M1, M2, ... если возможно
    def sort_key(r):
        rid = r["id"]
        if rid.startswith("M") and rid[1:].isdigit():
            return int(rid[1:])
        return 9999
    return sorted(ROUTES.values(), key=sort_key)


# -------------------------------------------------------------------
# DATABASE
# -------------------------------------------------------------------

def init_db():
    fresh = not os.path.exists(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # события
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT NOT NULL,
            route_id TEXT NOT NULL,
            stop_id INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            user_id INTEGER
        )
    """)

    # настройки пользователей (выбор маршрута)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            route_id TEXT NOT NULL
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_day_route ON events(day, route_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_day_route_stop ON events(day, route_id, stop_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_route_minute ON events(route_id, minute)")

    conn.commit()
    conn.close()
    return fresh


def add_event(day: str, route_id: str, stop_id: int, minute: int, user_id: Optional[int]):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (day, route_id, stop_id, minute, user_id) VALUES (?, ?, ?, ?, ?)",
        (day, route_id, stop_id, minute, user_id),
    )
    conn.commit()
    conn.close()


def get_today_events(route_id: str) -> List[Tuple[int, int]]:
    """Вернёт (stop_id, minute) за сегодня по конкретному маршруту."""
    day = today_str()
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT stop_id, minute FROM events WHERE day = ? AND route_id = ? ORDER BY minute ASC",
        (day, route_id),
    )
    rows = cur.fetchall()
    conn.close()
    return [(int(sid), int(m)) for sid, m in rows]


def get_events_by_stop_today(route_id: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for sid, m in get_today_events(route_id):
        out.setdefault(sid, []).append(m)
    return out


# -------------------------------------------------------------------
# USER SETTINGS (ROUTE CHOICE)
# -------------------------------------------------------------------

def get_user_route_id(user_id: int) -> Optional[str]:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT route_id FROM user_settings WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0]
    return None


def set_user_route_id(user_id: int, route_id: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_settings (user_id, route_id) VALUES (?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET route_id = excluded.route_id",
        (user_id, route_id),
    )
    conn.commit()
    conn.close()


def is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID


# -------------------------------------------------------------------
# UI HELPERS
# -------------------------------------------------------------------

def main_menu():
    kb = InlineKeyboardBuilder()
    kb.button(text="📍 Где автобус?", callback_data="where")
    kb.button(text="🚌 Отметить прибытие", callback_data="press")
    kb.adjust(1)
    return kb.as_markup()


def routes_keyboard():
    kb = InlineKeyboardBuilder()
    for r in list_routes_ordered():
        kb.button(text=r["name"], callback_data=f"route_{r['id']}")
    kb.adjust(1)
    return kb.as_markup()


async def answer_with_menu(message: Message, text: str):
    await message.answer(text, reply_markup=main_menu())


async def callback_answer_with_menu(callback: CallbackQuery, text: str):
    await callback.message.answer(text, reply_markup=main_menu())


async def ask_route_select_message(message: Message):
    await message.answer(
        "🚍 Выберите ваш маршрут:",
        reply_markup=routes_keyboard()
    )


async def ask_route_select_callback(callback: CallbackQuery):
    await callback.message.answer(
        "🚍 Выберите ваш маршрут:",
        reply_markup=routes_keyboard()
    )


# -------------------------------------------------------------------
# CORE COMPUTATION (PER ROUTE)
# -------------------------------------------------------------------

def compute_clean_means_by_stop(route_id: str):
    """
    Фильтрация выбросов и среднее время прибытия по каждой остановке
    для конкретного маршрута.
    """
    route = get_route(route_id)
    if not route:
        return {}, 0, None, None

    schedule = route["stops"]
    plan = {s["id"]: s["minute"] for s in schedule}

    events = get_events_by_stop_today(route_id)
    means: Dict[int, float] = {}
    total_used = 0
    latest_minute = None
    latest_stop = None

    for sid, mins in events.items():
        if sid not in plan:
            continue
        pm = plan[sid]
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


def build_eta_with_segments_and_ema(route_id: str):
    """
    Сегментная модель + EMA для одного маршрута.
    """
    route = get_route(route_id)
    if not route:
        return {}, 0, "маршрут не найден", None, None, 0.0

    schedule = route["stops"]
    plan = {s["id"]: s["minute"] for s in schedule}
    ids = [s["id"] for s in schedule]

    means, total_used, latest_minute, latest_stop = compute_clean_means_by_stop(route_id)

    if not means:
        eta_map = {sid: float(plan[sid]) for sid in ids}
        return eta_map, 40, "нет данных — автобус по расписанию", None, None, 0.0

    conf = min(95, 40 + total_used * 5)
    if latest_minute is not None:
        age = now_minute_of_day() - latest_minute
        if age > 60:
            conf = int(conf * 0.6)
        elif age > 30:
            conf = int(conf * 0.8)

    eta_raw: Dict[int, float] = {sid: means[sid] for sid in means}

    # сегменты по расписанию
    seg_plan: Dict[Tuple[int, int], int] = {}
    for a, b in zip(ids[:-1], ids[1:]):
        d = plan[b] - plan[a]
        seg_plan[(a, b)] = max(MIN_SEGMENT_MIN, d)

    # сегменты по фактам
    seg_fact: Dict[Tuple[int, int], int] = {}
    for a, b in zip(ids[:-1], ids[1:]):
        if a in means and b in means:
            diff = means[b] - means[a]
            if diff >= MIN_SEGMENT_MIN:
                seg_fact[(a, b)] = int(round(diff))

    # распространение ETA вперёд/назад
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

    for sid in ids:
        if sid not in eta_raw:
            eta_raw[sid] = float(plan[sid])

    # смещения и EMA
    offsets = {sid: eta_raw[sid] - plan[sid] for sid in ids}
    ema_offsets: Dict[int, float] = {}
    ema: Optional[float] = None

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

    return eta_final, conf, status, latest_minute, latest_stop, avg_off


def build_eta_window(route_id: str):
    """
    Окно из 5 остановок вокруг ключевой для конкретного маршрута.
    """
    route = get_route(route_id)
    if not route:
        return [], 0, "маршрут не найден", None, None, 0.0

    schedule = route["stops"]
    eta_map, conf, status, latest_minute, latest_stop, avg_off = build_eta_with_segments_and_ema(route_id)

    ids = [s["id"] for s in schedule]
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
        stop = next(s for s in schedule if s["id"] == sid)
        eta_minute = eta_map[sid]
        window.append({
            "id": sid,
            "name": stop["name"],
            "eta_str": human_time_from_minute(int(round(eta_minute))),
            "is_key": sid == key_sid,
        })

    return window, conf, status, latest_minute, latest_stop, avg_off


# -------------------------------------------------------------------
# AUTO RESET AT MIDNIGHT
# -------------------------------------------------------------------

LAST_RESET_DAY = today_str()

async def auto_reset_daily():
    global LAST_RESET_DAY
    while True:
        now_day = today_str()
        if now_day != LAST_RESET_DAY:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("DELETE FROM events")
            conn.commit()
            conn.close()

            LAST_RESET_DAY = now_day
            print(f"[AUTO RESET] Database cleared at midnight → {now_day}")

        await asyncio.sleep(MIDNIGHT_CHECK_INTERVAL)


# -------------------------------------------------------------------
# HANDLERS
# -------------------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: Message):
    user_id = message.from_user.id
    route_id = get_user_route_id(user_id)

    if route_id and get_route(route_id):
        route_name = get_route(route_id)["name"]
        await answer_with_menu(
            message,
            f"🚍 Ваш маршрут: <b>{route_name}</b>\nВыберите действие:"
        )
    else:
        await ask_route_select_message(message)


@dp.message(Command("change_route"))
async def cmd_change_route(message: Message):
    await ask_route_select_message(message)


@dp.message(Command("stats_today"))
async def cmd_stats_today(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("⛔ У вас нет доступа к этой команде.")
        return

    route_id = get_user_route_id(message.from_user.id)
    if not route_id or not get_route(route_id):
        await message.answer("Сначала выберите маршрут командой /change_route.")
        return

    events = get_today_events(route_id)
    if not events:
        await answer_with_menu(
            message,
            f"📊 Статистика за сегодня ({get_route(route_id)['name']}):\nОтметок за сегодня нет."
        )
        return

    schedule = get_route(route_id)["stops"]
    plan = {s["id"]: s["minute"] for s in schedule}

    offsets: List[int] = []
    for sid, m in events:
        if sid in plan:
            offsets.append(m - plan[sid])

    if not offsets:
        await answer_with_menu(
            message,
            "📊 Статистика за сегодня:\nДанные есть, но не удалось сопоставить с расписанием."
        )
        return

    total = len(offsets)
    unique_stops = len(set(sid for sid, _ in events))
    avg_off = sum(offsets) / total
    min_off = min(offsets)
    max_off = max(offsets)

    last_sid, last_minute = events[-1]
    last_stop_name = next(s["name"] for s in schedule if s["id"] == last_sid)
    last_time = human_time_from_minute(last_minute)

    lines = [
        f"📊 Статистика за сегодня ({get_route(route_id)['name']}):",
        f"• Отметок: {len(events)}",
        f"• Уникальных остановок: {unique_stops}",
        f"• Среднее отклонение: {avg_off:+.1f} мин",
        f"• Минимальное отклонение: {min_off:+d} мин",
        f"• Максимальное отклонение: {max_off:+d} мин",
        f"• Последняя отметка: {last_stop_name} ({last_time})",
    ]

    await answer_with_menu(message, "\n".join(lines))


@dp.message(Command("reset_now"))
async def cmd_reset_now(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("⛔ У вас нет доступа к этой команде.")
        return

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM events")
    conn.commit()
    conn.close()

    await answer_with_menu(message, "🗑 Данные по отметкам очищены (все маршруты, все дни).")


@dp.callback_query(F.data.startswith("route_"))
async def on_route_select(callback: CallbackQuery):
    user_id = callback.from_user.id
    route_id = callback.data.split("_", 1)[1]

    if not get_route(route_id):
        await callback.answer("Маршрут не найден.", show_alert=True)
        return

    set_user_route_id(user_id, route_id)
    route_name = get_route(route_id)["name"]
    await callback.message.answer(
        f"Маршрут выбран: <b>{route_name}</b>.\nТеперь вы можете пользоваться кнопками ниже.",
        reply_markup=main_menu()
    )
    await callback.answer()


@dp.callback_query(F.data == "where")
async def on_where(callback: CallbackQuery):
    user_id = callback.from_user.id
    route_id = get_user_route_id(user_id)

    if not route_id or not get_route(route_id):
        await ask_route_select_callback(callback)
        await callback.answer()
        return

    route = get_route(route_id)
    window, conf, status, latest_minute, latest_stop, avg_off = build_eta_window(route_id)

    lines: List[str] = []
    lines.append(f"Маршрут: <b>{route['name']}</b>\n")

    if latest_minute is not None and latest_stop is not None:
        schedule = route["stops"]
        stop_name = next(s["name"] for s in schedule if s["id"] == latest_stop)
        lines.append(
            f"📍 Последняя отметка: <b>{stop_name}</b> — <b>{human_time_from_minute(latest_minute)}</b>\n"
        )
    else:
        lines.append("Нет отметок за сегодня.\n")

    lines.append("<b>Расчетное время:</b>\n")

    for w in window:
        if w["is_key"]:
            lines.append(f"➡️ <b>{w['name']} — {w['eta_str']}</b>")
        else:
            lines.append(f"{w['name']} — {w['eta_str']}")

    if avg_off > 1.5:
        emoji = "🟥"
    elif avg_off < -1.5:
        emoji = "🟨"
    else:
        emoji = "🟩"

    lines.append("")
    lines.append(f"Точность прогноза: {conf}%")
    lines.append(f"Ситуация: {emoji} {status}")

    await callback_answer_with_menu(callback, "\n".join(lines))
    await callback.answer()


@dp.callback_query(F.data == "press")
async def on_press(callback: CallbackQuery):
    """
    «Отметить прибытие» — фиксируем время и предлагаем TOP-5 остановок маршрута.
    """
    user_id = callback.from_user.id
    route_id = get_user_route_id(user_id)

    if not route_id or not get_route(route_id):
        await ask_route_select_callback(callback)
        await callback.answer()
        return

    route = get_route(route_id)
    schedule = route["stops"]

    now_m = now_minute_of_day()
    day = today_str()
    expiry_m = now_m + (SESSION_TTL // 60) + 1

    PRESSED_SESSIONS[user_id] = (now_m, expiry_m, day, route_id)

    # TOP-5 ближайших остановок по времени расписания
    diffs = [(s["id"], abs(s["minute"] - now_m)) for s in schedule]
    diffs.sort(key=lambda x: x[1])
    top_ids = [sid for sid, _ in diffs[:5]]

    kb = InlineKeyboardBuilder()
    for s in schedule:
        if s["id"] in top_ids:
            kb.button(text=s["name"], callback_data=f"stop_{s['id']}")
    kb.button(text="Показать все остановки", callback_data="all_stops")
    kb.adjust(1)

    await callback.message.answer("Выберите остановку:", reply_markup=kb.as_markup())
    await callback.answer()


@dp.callback_query(F.data == "all_stops")
async def on_all_stops(callback: CallbackQuery):
    user_id = callback.from_user.id
    session = PRESSED_SESSIONS.get(user_id)

    if not session:
        await callback_answer_with_menu(callback, "Сессия истекла. Нажмите «🚌 Отметить прибытие» ещё раз.")
        await callback.answer()
        return

    _, _, _, route_id = session
    route = get_route(route_id)
    if not route:
        await callback_answer_with_menu(callback, "Маршрут не найден. Попробуйте снова.")
        await callback.answer()
        return

    schedule = route["stops"]

    kb = InlineKeyboardBuilder()
    for s in schedule:
        kb.button(text=s["name"], callback_data=f"stop_{s['id']}")
    kb.adjust(1)

    await callback.message.answer("Полный список остановок:", reply_markup=kb.as_markup())
    await callback.answer()


@dp.callback_query(F.data.startswith("stop_"))
async def on_stop(callback: CallbackQuery):
    user_id = callback.from_user.id
    session = PRESSED_SESSIONS.get(user_id)

    if not session:
        await callback_answer_with_menu(callback, "Сессия истекла. Повторите отметку.")
        await callback.answer()
        return

    pressed_m, expiry_m, day, route_id = session
    now_m = now_minute_of_day()

    if now_m > expiry_m:
        PRESSED_SESSIONS.pop(user_id, None)
        await callback_answer_with_menu(callback, "Сессия устарела. Повторите отметку.")
        await callback.answer()
        return

    route = get_route(route_id)
    if not route:
        PRESSED_SESSIONS.pop(user_id, None)
        await callback_answer_with_menu(callback, "Маршрут не найден. Повторите отметку.")
        await callback.answer()
        return

    schedule = route["stops"]

    stop_id = int(callback.data.split("_")[1])
    PRESSED_SESSIONS.pop(user_id, None)

    plan_min = next(s["minute"] for s in schedule if s["id"] == stop_id)
    delta = pressed_m - plan_min

    stop_name = next(s["name"] for s in schedule if s["id"] == stop_id)
    human = human_time_from_minute(pressed_m)

    # сохраняем событие
    add_event(day, route_id, stop_id, pressed_m, user_id)

    text = (
        f"Спасибо! Автобус отмечен на маршруте <b>{route['name']}</b>\n"
        f"Остановка: <b>{stop_name}</b>\n"
        f"Время: <b>{human}</b>\n"
        f"Отклонение от расписания: <b>{delta:+} мин.</b>"
    )

    await callback_answer_with_menu(callback, text)
    await callback.answer()


# -------------------------------------------------------------------
# START BOT
# -------------------------------------------------------------------

async def main():
    init_db()
    asyncio.create_task(auto_reset_daily())
    print("Transport bot 1.0 (multi-route) started.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
