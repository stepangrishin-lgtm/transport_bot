import os
import json
import sqlite3
import asyncio
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

BOT_TOKEN = "8598055235:AAEcMaVgBkiKYokFXxDd2_govw4ytGp8Rn4"  # <<< ВСТАВЬ СВОЙ ТОКЕН
ADMIN_ID = 331165172                     # твой Telegram ID

ROUTES_FILE = "routes.json"
DB_FILE = "transport.db"

SESSION_TTL = 180        # 3 минуты — время жизни сессии отметки
MAX_DELTA_MIN = 60       # макс. отклонение для нормальной отметки
MIN_SEGMENT_MIN = 1      # минимальное время сегмента (мин)
EMA_ALPHA = 0.5          # коэффициент сглаживания EMA
MIDNIGHT_CHECK_INTERVAL = 10800  # 3 часа, сек
SEGMENT_UPDATE_INTERVAL = 300    # 5 минут, сек

# --- Уведомления о задержках ---
DELAY_CHECK_INTERVAL = 180       # каждые 3 минуты проверяем задержки
DELAY_THRESHOLD_MIN = 12         # сильная задержка, мин
MIN_EVENTS_FOR_NOTIF = 3         # минимум отметок за день
MIN_NOTIF_INTERVAL_MIN = 15      # не чаще одного уведомления раз в 15 минут
DELAY_INCREASE_MIN = 5           # задержка должна увеличиться хотя бы на 5 минут

# id чата для уведомлений (супергруппа с темами)
GROUP_CHAT_ID = -1002877243877

# соответствие маршрутов и тем (topics)
ROUTE_TOPICS: Dict[str, int] = {
    "M1": 63,
    "M2": 64,
    "M3": 66,
    "M4": 7,
    "M5": 5,
    "M6": 2,
    "M7": 3,
    "M8": 6,
}

# user_id -> (pressed_m, expiry_m, day, route_id)
PRESSED_SESSIONS: Dict[int, Tuple[int, int, str, str]] = {}

# Жёстко задаём московский часовой пояс (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))

bot = Bot(
    BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()

# -------------------------------------------------------------------
# TIME HELPERS
# -------------------------------------------------------------------

def now_local() -> datetime:
    return datetime.now(MOSCOW_TZ)


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


ROUTES: Dict[str, Dict] = load_routes()


def get_route(route_id: str) -> Optional[Dict]:
    return ROUTES.get(route_id)


def list_routes_ordered() -> List[Dict]:
    def sort_key(r):
        rid = r["id"]
        if rid.startswith("M") and rid[1:].isdigit():
            return int(rid[1:])
        return 9999
    return sorted(ROUTES.values(), key=sort_key)

# -------------------------------------------------------------------
# DATABASE
# -------------------------------------------------------------------

def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


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

    # настройки пользователей
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            route_id TEXT NOT NULL
        )
    """)

    # миграция segment_stats: сбрасываем старую схему (из 1.1) и создаём новую
    if _table_exists(cur, "segment_stats"):
        cur.execute("PRAGMA table_info(segment_stats)")
        cols = [row[1] for row in cur.fetchall()]
        if "abnormal_count" not in cols or "critical" not in cols:
            cur.execute("DROP TABLE segment_stats")
            if _table_exists(cur, "segment_updates"):
                cur.execute("DROP TABLE segment_updates")

    # историческая статистика сегментов
    cur.execute("""
        CREATE TABLE IF NOT EXISTS segment_stats (
            route_id TEXT NOT NULL,
            from_stop_id INTEGER NOT NULL,
            to_stop_id INTEGER NOT NULL,
            avg_min REAL NOT NULL,
            n INTEGER NOT NULL,
            abnormal_count INTEGER NOT NULL DEFAULT 0,
            critical INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(route_id, from_stop_id, to_stop_id)
        )
    """)

    # отметки об обновлении сегментов
    cur.execute("""
        CREATE TABLE IF NOT EXISTS segment_updates (
            route_id TEXT NOT NULL,
            day TEXT NOT NULL,
            PRIMARY KEY(route_id, day)
        )
    """)

    # таблица для уведомлений о задержках
    cur.execute("""
        CREATE TABLE IF NOT EXISTS delay_notifications (
            route_id TEXT PRIMARY KEY,
            day TEXT,
            last_minute INTEGER,
            last_delay REAL,
            first_sent INTEGER NOT NULL DEFAULT 0
        )
    """)

    # индексы
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
# USER SETTINGS
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
# CORE COMPUTATION
# -------------------------------------------------------------------

def compute_clean_means_by_stop(route_id: str):
    """
    Среднее время прибытия по каждой остановке (за сегодня) с фильтрацией выбросов.
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


def load_segment_stats_for_route(route_id: str) -> Tuple[Dict[Tuple[int, int], float],
                                                         Dict[Tuple[int, int], int]]:
    """
    Загружает исторические средние и флаг критичности по сегментам для маршрута.
    Возвращает два словаря:
      avg[(a, b)] = avg_min
      crit[(a, b)] = 0/1
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT from_stop_id, to_stop_id, avg_min, critical "
        "FROM segment_stats WHERE route_id = ?",
        (route_id,)
    )
    rows = cur.fetchall()
    conn.close()
    avg: Dict[Tuple[int, int], float] = {}
    crit: Dict[Tuple[int, int], int] = {}
    for from_id, to_id, avg_min, critical in rows:
        key = (int(from_id), int(to_id))
        avg[key] = float(avg_min)
        crit[key] = int(critical)
    return avg, crit


def build_eta_with_segments_and_ema(route_id: str):
    """
    Сегментная модель + история + критичные сегменты + EMA для одного маршрута.
    """
    route = get_route(route_id)
    if not route:
        return {}, 0, "маршрут не найден", None, None, 0.0

    schedule = route["stops"]
    plan = {s["id"]: s["minute"] for s in schedule}
    ids = [s["id"] for s in schedule]

    means, total_used, latest_minute, latest_stop = compute_clean_means_by_stop(route_id)

    # исторические сегменты и критичность
    hist_avg, hist_crit = load_segment_stats_for_route(route_id)

    # нет сегодняшних данных — работаем только по плану + истории
    if not means:
        eta_map: Dict[int, float] = {}
        eta_map[ids[0]] = float(plan[ids[0]])

        for a, b in zip(ids[:-1], ids[1:]):
            plan_seg = max(MIN_SEGMENT_MIN, plan[b] - plan[a])
            hist_seg = hist_avg.get((a, b))
            if hist_seg is not None:
                base_seg = (plan_seg + hist_seg) / 2.0
                # если сегмент уже критичный — усиливаем влияние истории
                if hist_crit.get((a, b), 0) == 1:
                    base_seg = (plan_seg + hist_seg * 1.5) / 2.0
            else:
                base_seg = plan_seg
            eta_map[b] = eta_map[a] + base_seg

        return eta_map, 40, "нет данных — автобус по расписанию и истории", None, None, 0.0

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

    # базовые сегменты = расписание + история (+ критичность)
    seg_base: Dict[Tuple[int, int], float] = {}
    for a, b in zip(ids[:-1], ids[1:]):
        plan_seg = seg_plan[(a, b)]
        hist_seg = hist_avg.get((a, b))
        if hist_seg is not None:
            base_seg = (plan_seg + hist_seg) / 2.0
            if hist_crit.get((a, b), 0) == 1:
                # критичный сегмент → усиливаем влияние истории
                base_seg = (plan_seg + hist_seg * 1.5) / 2.0
            seg_base[(a, b)] = base_seg
        else:
            seg_base[(a, b)] = float(plan_seg)

    # сегменты по фактам сегодня (если есть две соседние остановки с данными)
    seg_fact: Dict[Tuple[int, int], int] = {}
    for a, b in zip(ids[:-1], ids[1:]):
        if a in means and b in means:
            diff = means[b] - means[a]
            if diff >= MIN_SEGMENT_MIN:
                seg_fact[(a, b)] = int(round(diff))

    # распространение ETA вперёд/назад по маршруту
    changed = True
    while changed:
        changed = False
        for a, b in zip(ids[:-1], ids[1:]):
            if a in eta_raw and b not in eta_raw:
                seg = seg_fact.get((a, b), seg_base[(a, b)])
                eta_raw[b] = eta_raw[a] + seg
                changed = True
            if b in eta_raw and a not in eta_raw:
                seg = seg_fact.get((a, b), seg_base[(a, b)])
                eta_raw[a] = eta_raw[b] - seg
                changed = True

    for sid in ids:
        if sid not in eta_raw:
            if sid == ids[0]:
                eta_raw[sid] = float(plan[sid])
            else:
                idx = ids.index(sid)
                a = ids[idx - 1]
                b = sid
                seg = seg_base[(a, b)]
                eta_raw[sid] = eta_raw[a] + seg

    # offsets + EMA
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
    Новая логика окна из 5 остановок:
    
    1) Если есть свежая отметка (<= 12 мин):
        - ключевая = ближайшая "будущая" остановка по прогнозу
          среди тех, что идут ВПЕРЁД после последней отметки.
          
    2) Если отметка устарела (> 12 мин) или отметок нет:
        - ключевая = ближайшая будущая по ETA (ETA >= now).
        - если таких нет — последняя остановка.
        
    Окно = ключевая ± 2 остановки по маршруту.
    """

    route = get_route(route_id)
    if not route:
        return [], 0, "маршрут не найден", None, None, 0.0

    schedule = route["stops"]
    ids = [s["id"] for s in schedule]
    id_to_index = {sid: i for i, sid in enumerate(ids)}

    eta_map, conf, status, latest_minute, latest_stop, avg_off = build_eta_with_segments_and_ema(route_id)
    now_m = now_minute_of_day()

    # --------------------------------------------------------------
    # 1. Если НЕТ отметок за сегодня → используем чистый ETA выбор
    # --------------------------------------------------------------
    if latest_stop is None:
        # Берём ближайшую будущую ETA
        future = [(sid, eta_map[sid]) for sid in ids if eta_map[sid] >= now_m]
        if future:
            key_sid, _ = min(future, key=lambda x: x[1])
        else:
            key_sid = ids[-1]  # автобус должен быть у конца маршрута
            
        key_index = id_to_index[key_sid]

    else:
        # ----------------------------------------------------------
        # 2. Есть отметка → проверяем свежая ли она
        # ----------------------------------------------------------
        age = now_m - latest_minute

        if age <= 12:
            # ------------------------------------------------------
            # 2A. СВЕЖАЯ отметка → выбираем будущую остановку
            # ------------------------------------------------------
            last_idx = id_to_index[latest_stop]

            # Кандидаты вперёд: остановки начиная с последней отмеченной
            forward_ids = ids[last_idx:]

            # Ищем среди них ближайшую будущую ETA (>= now)
            future = [(sid, eta_map[sid]) for sid in forward_ids if eta_map[sid] >= now_m]

            if future:
                key_sid, _ = min(future, key=lambda x: x[1])
            else:
                key_sid = forward_ids[-1]   # автобус уже должен быть в конце этого участка

            key_index = id_to_index[key_sid]

        else:
            # ------------------------------------------------------
            # 2B. СТАРАЯ отметка → fallback к ETA-позиционированию
            # ------------------------------------------------------
            future = [(sid, eta_map[sid]) for sid in ids if eta_map[sid] >= now_m]
            if future:
                key_sid, _ = min(future, key=lambda x: x[1])
            else:
                key_sid = ids[-1]

            key_index = id_to_index[key_sid]

    # --------------------------------------------------------------
    # 3. Формируем окно из 5 остановок вокруг ключевой
    # --------------------------------------------------------------
    start = max(0, key_index - 2)
    end = min(len(ids), start + 5)
    if end - start < 5:
        start = max(0, end - 5)

    chosen = ids[start:end]

    window = []
    for sid in chosen:
        stop = next(s for s in schedule if s["id"] == sid)
        eta_str = human_time_from_minute(int(round(eta_map[sid])))
        window.append({
            "id": sid,
            "name": stop["name"],
            "eta_str": eta_str,
            "is_key": sid == key_sid
        })

    return window, conf, status, latest_minute, latest_stop, avg_off


# -------------------------------------------------------------------
# ROUTE STATE & SEGMENT STATS UPDATE
# -------------------------------------------------------------------

def get_route_state(route_id: str) -> str:
    """
    Состояние рейса: NOT_STARTED, IN_PROGRESS, FINISHED
    """
    route = get_route(route_id)
    if not route:
        return "UNKNOWN"

    schedule = route["stops"]
    first_min = schedule[0]["minute"]
    last_min = schedule[-1]["minute"]

    events = get_today_events(route_id)
    now_m = now_minute_of_day()

    if not events:
        if now_m < first_min - 15:
            return "NOT_STARTED"
        if now_m > last_min + 60:
            return "FINISHED"
        return "IN_PROGRESS"

    last_sid, last_mark_minute = events[-1]
    last_three_ids = [s["id"] for s in schedule[-3:]]
    age = now_m - last_mark_minute

    if (last_sid in last_three_ids) and (age >= 60):
        return "FINISHED"

    if now_m < first_min - 15:
        return "NOT_STARTED"

    return "IN_PROGRESS"


def is_segment_updated_today(route_id: str) -> bool:
    day = today_str()
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM segment_updates WHERE route_id = ? AND day = ?",
        (route_id, day)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def mark_segment_updated_today(route_id: str):
    day = today_str()
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO segment_updates (route_id, day) VALUES (?, ?)",
        (route_id, day)
    )
    conn.commit()
    conn.close()


def update_segment_stats_for_route(route_id: str):
    """
    Обновляет историческую статистику сегментов для маршрута
    на основе сегодняшних очищенных средних по остановкам.
    Учитывает "критичные сегменты" с накоплением по рабочим дням.
    """
    if is_segment_updated_today(route_id):
        return

    route = get_route(route_id)
    if not route:
        return

    schedule = route["stops"]
    ids = [s["id"] for s in schedule]
    plan = {s["id"]: s["minute"] for s in schedule}

    means, total_used, latest_minute, latest_stop = compute_clean_means_by_stop(route_id)
    if not means:
        # нет данных за сегодняшний рейс — просто помечаем и выходим
        mark_segment_updated_today(route_id)
        return

    # сегменты "сегодня" (там, где есть данные на A и B)
    today_segments: Dict[Tuple[int, int], float] = {}
    for a, b in zip(ids[:-1], ids[1:]):
        if a in means and b in means:
            seg = means[b] - means[a]
            if seg >= MIN_SEGMENT_MIN:
                today_segments[(a, b)] = float(seg)

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    for a, b in zip(ids[:-1], ids[1:]):
        key = (a, b)
        if key not in today_segments:
            continue

        today_val = today_segments[key]
        plan_seg = max(MIN_SEGMENT_MIN, plan[b] - plan[a])

        # критерий "аномальный сегмент" за сегодняшний рабочий день:
        is_abnormal_today = (today_val > plan_seg * 1.5) and (today_val > plan_seg + 2)

        # читаем старые значения
        cur.execute(
            "SELECT avg_min, n, abnormal_count, critical "
            "FROM segment_stats WHERE route_id = ? AND from_stop_id = ? AND to_stop_id = ?",
            (route_id, a, b)
        )
        row = cur.fetchone()

        if row is None:
            avg_old = None
            n_old = 0
            streak = 0
            critical = 0
        else:
            avg_old, n_old, streak, critical = row
            avg_old = float(avg_old)
            n_old = int(n_old)
            streak = int(streak)
            critical = int(critical)

        # обновление avg_min и n (как и раньше, с "памятью" до 7 дней)
        if n_old == 0:
            avg_new = today_val
            n_new = 1
        else:
            if n_old < 7:
                avg_new = (avg_old * n_old + today_val) / (n_old + 1)
                n_new = n_old + 1
            else:
                avg_new = avg_old * 0.8 + today_val * 0.2
                n_new = 7

        # streak > 0  → подряд аномальные дни
        # streak < 0  → подряд нормальные дни после критичности
        # streak == 0 → нейтральное состояние
        if is_abnormal_today:
            if streak >= 0:
                streak_new = streak + 1
            else:
                streak_new = 1
        else:
            if critical == 1:
                if streak <= 0:
                    streak_new = streak - 1
                else:
                    streak_new = -1
            else:
                streak_new = 0

        critical_new = critical
        # сегмент становится критичным после 3 аномальных рабочих дней подряд
        if streak_new >= 3:
            critical_new = 1
        # сегмент перестаёт быть критичным после 3 нормальных рабочих дней подряд
        if streak_new <= -3:
            critical_new = 0
            streak_new = 0  # сбрасываем счётчик

        if row is None:
            cur.execute(
                "INSERT INTO segment_stats "
                "(route_id, from_stop_id, to_stop_id, avg_min, n, abnormal_count, critical) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (route_id, a, b, avg_new, n_new, streak_new, critical_new)
            )
        else:
            cur.execute(
                "UPDATE segment_stats "
                "SET avg_min = ?, n = ?, abnormal_count = ?, critical = ? "
                "WHERE route_id = ? AND from_stop_id = ? AND to_stop_id = ?",
                (avg_new, n_new, streak_new, critical_new, route_id, a, b)
            )

    conn.commit()
    conn.close()

    mark_segment_updated_today(route_id)


async def auto_segment_update_loop():
    """
    Периодически проверяет, завершён ли рейс по каждому маршруту,
    и если да — обновляет статистику сегментов (раз в день),
    с учётом "критичных" сегментов.
    """
    while True:
        try:
            for route_id in ROUTES.keys():
                if is_segment_updated_today(route_id):
                    continue

                route = get_route(route_id)
                if not route:
                    continue

                events = get_today_events(route_id)
                if not events:
                    continue

                state = get_route_state(route_id)
                if state == "FINISHED":
                    update_segment_stats_for_route(route_id)
        except Exception as e:
            print(f"[SEGMENT UPDATE ERROR] {e}")

        await asyncio.sleep(SEGMENT_UPDATE_INTERVAL)

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
            cur.execute("DELETE FROM segment_updates")
            conn.commit()
            conn.close()

            LAST_RESET_DAY = now_day
            print(f"[AUTO RESET] Database cleared at midnight → {now_day}")

        await asyncio.sleep(MIDNIGHT_CHECK_INTERVAL)

# -------------------------------------------------------------------
# DELAY NOTIFICATIONS
# -------------------------------------------------------------------

DELAY_MESSAGES = [
    "⚠ На маршруте {route_name} образовалась задержка ~{delay} минут.\n"
    "Последняя отметка: {stop_name} ({stop_time}).\n"
    "Учитывайте это при планировании времени.",

    "⚠ Автобус {route_name} движется медленнее обычного.\n"
    "Отклонение от графика — около {delay} минут.\n"
    "Последняя отметка: {stop_name}.",

    "⚠ Небольшая сложность на маршруте {route_name}.\n"
    "Автобус задерживается примерно на {delay} минут.\n"
    "Последняя отметка: {stop_name} ({stop_time}).",

    "⚠ Задержка на маршруте {route_name}: ~{delay} минут.\n"
    "Фиксация была на {stop_name}.",

    "⚠ Автобус {route_name} застрял в пути — задержка около {delay} минут.\n"
    "Последняя отметка: {stop_name}.",

    "⚠ Дорожная обстановка повлияла на маршрут {route_name}.\n"
    "Задержка: примерно {delay} минут.\n"
    "Последняя отметка: {stop_name} ({stop_time}).",

    "⚠ Автобус {route_name} сегодня идёт с заметным опозданием (~{delay} минут).\n"
    "Последняя зафиксированная остановка: {stop_name}.",

    "⚠ На маршруте {route_name} временная задержка (около {delay} минут).\n"
    "Учтите это при выходе на остановку.",

    "⚠ Автобус {route_name} немного выбился из графика.\n"
    "Отклонение: {delay} минут.\n"
    "Отметка: {stop_name} ({stop_time}).",

    "⚠ На маршруте {route_name} возможна задержка.\n"
    "Текущее отклонение — примерно {delay} минут.\n"
    "Последняя отметка: {stop_name}.",

    "⚠ На пути автобуса {route_name} затруднённое движение.\n"
    "Задержка: ~{delay} минут.\n"
    "Последняя остановка: {stop_name}.",

    "⚠ Автобус {route_name} задерживается примерно на {delay} минут.\n"
    "Информация по последней остановке: {stop_name} ({stop_time}).",
]

POSITIVE_MESSAGES = [
    "Я слежу за маршрутом и обновлю информацию, как только появятся новые отметки 🙂",
    "Спасибо за отметки — благодаря вам я могу точнее рассчитать прогноз!",
    "Если появятся новые данные, я сразу подкорректирую время прибытия.",
    "Держу маршрут под контролем. Сообщу, если ситуация изменится.",
    "Я обновлю прогноз, как только увижу следующую отметку.",
    "Благодарю всех, кто отмечает остановки — это делает информацию точнее 🙂",
]


async def auto_delay_notifications_loop():
    """
    Цикл, который каждые DELAY_CHECK_INTERVAL секунд проверяет маршруты
    и при сильной задержке (>=12 мин, минимум 3 отметки, рост >=5 мин,
    интервал >=15 минут) отправляет уведомления в нужные темы.
    """
    while True:
        try:
            day = today_str()
            now_m = now_minute_of_day()

            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()

            for route_id, route in ROUTES.items():
                # 1) Маршрут должен быть известен и иметь тему
                if route_id not in ROUTE_TOPICS:
                    continue

                # 2) Должно быть хотя бы MIN_EVENTS_FOR_NOTIF отметок за сегодня
                events = get_today_events(route_id)
                if len(events) < MIN_EVENTS_FOR_NOTIF:
                    continue

                # 3) Рейс должен быть в процессе
                state = get_route_state(route_id)
                if state != "IN_PROGRESS":
                    continue

                # 4) Считаем ETA и среднюю задержку
                eta_map, conf, status, latest_minute, latest_stop, avg_off = build_eta_with_segments_and_ema(route_id)

                # если чисто расписание без данных (avg_off ~ 0) — не тревожим
                if avg_off < DELAY_THRESHOLD_MIN:
                    continue

                # 5) Берём информацию о последней остановке
                schedule = route["stops"]
                stop_name = schedule[-1]["name"]
                stop_time_str = "—"

                if latest_stop is not None and latest_minute is not None:
                    try:
                        stop_obj = next(s for s in schedule if s["id"] == latest_stop)
                        stop_name = stop_obj["name"]
                        stop_time_str = human_time_from_minute(latest_minute)
                    except StopIteration:
                        # fallback: берём последнюю отметку из events
                        last_sid, last_min = events[-1]
                        try:
                            stop_obj = next(s for s in schedule if s["id"] == last_sid)
                            stop_name = stop_obj["name"]
                            stop_time_str = human_time_from_minute(last_min)
                        except StopIteration:
                            stop_name = schedule[-1]["name"]
                            stop_time_str = human_time_from_minute(last_min)
                else:
                    # нет latest_* из модели — берём последнюю реальную отметку
                    last_sid, last_min = events[-1]
                    try:
                        stop_obj = next(s for s in schedule if s["id"] == last_sid)
                        stop_name = stop_obj["name"]
                    except StopIteration:
                        stop_name = schedule[-1]["name"]
                    stop_time_str = human_time_from_minute(last_min)

                delay_now = float(avg_off)

                # 6) Читаем, было ли уже уведомление сегодня по этому маршруту
                cur.execute(
                    "SELECT day, last_minute, last_delay, first_sent FROM delay_notifications WHERE route_id = ?",
                    (route_id,)
                )
                row = cur.fetchone()

                if row:
                    row_day, last_minute, last_delay, first_sent = row
                    row_day = row_day or ""
                    last_minute = last_minute if last_minute is not None else None
                    last_delay = float(last_delay) if last_delay is not None else None
                    first_sent = int(first_sent) if first_sent is not None else 0
                else:
                    row_day = ""
                    last_minute = None
                    last_delay = None
                    first_sent = 0

                # если день не совпадает — считаем, что сегодня ещё не уведомляли
                if row_day != day:
                    last_minute = None
                    last_delay = None
                    first_sent = 0

                # 7) Ограничение по интервалу: не чаще, чем раз в 15 минут
                if last_minute is not None:
                    if now_m - last_minute < MIN_NOTIF_INTERVAL_MIN:
                        continue

                # 8) Задержка должна увеличиться хотя бы на +5 минут,
                #    либо уведомлений сегодня ещё не было
                if last_delay is not None and last_minute is not None:
                    if delay_now < last_delay + DELAY_INCREASE_MIN:
                        continue

                # 9) Сформировать сообщение
                delay_int = int(round(delay_now))
                base_text_template = random.choice(DELAY_MESSAGES)
                text = base_text_template.format(
                    route_name=route["name"],
                    delay=delay_int,
                    stop_name=stop_name,
                    stop_time=stop_time_str,
                )

                # 10) Добавить позитивную фразу, если это первое уведомление за день
                if first_sent == 0:
                    positive = random.choice(POSITIVE_MESSAGES)
                    text = f"{text}\n\n{positive}"
                    first_sent_new = 1
                else:
                    first_sent_new = first_sent

                # 11) Отправка в нужную тему группы
                topic_id = ROUTE_TOPICS[route_id]
                try:
                    await bot.send_message(
                        chat_id=GROUP_CHAT_ID,
                        message_thread_id=topic_id,
                        text=text
                    )
                    print(f"[DELAY NOTIFY] route={route_id}, delay={delay_int} min")
                except Exception as send_err:
                    print(f"[DELAY NOTIFY ERROR] {send_err}")

                # 12) Обновляем запись о последнем уведомлении
                cur.execute(
                    "INSERT INTO delay_notifications (route_id, day, last_minute, last_delay, first_sent) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(route_id) DO UPDATE SET "
                    "day = excluded.day, last_minute = excluded.last_minute, "
                    "last_delay = excluded.last_delay, first_sent = excluded.first_sent",
                    (route_id, day, now_m, delay_now, first_sent_new)
                )

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"[DELAY LOOP ERROR] {e}")

        await asyncio.sleep(DELAY_CHECK_INTERVAL)

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
    cur.execute("DELETE FROM segment_updates")
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
    state = get_route_state(route_id)

    if state == "FINISHED":
        events = get_today_events(route_id)
        if events:
            last_sid, last_minute = events[-1]
            schedule = route["stops"]
            stop_name = next(s["name"] for s in schedule if s["id"] == last_sid)
            last_time = human_time_from_minute(last_minute)
            text = (
                f"Маршрут: <b>{route['name']}</b>\n\n"
                f"🏁 Рейс завершён.\n"
                f"Последняя отметка: <b>{stop_name}</b> — <b>{last_time}</b>."
            )
        else:
            text = (
                f"Маршрут: <b>{route['name']}</b>\n\n"
                f"🏁 Рейс завершён на сегодня."
            )
        await callback_answer_with_menu(callback, text)
        await callback.answer()
        return

    window, conf, status, latest_minute, latest_stop, avg_off = build_eta_window(route_id)

    lines: List[str] = []
    lines.append(f"Маршрут: <b>{route['name']}</b>\n")

    if state == "NOT_STARTED":
        first_stop = route["stops"][0]
        lines.append(
            f"🚍 Рейс ещё не начался. Первая остановка: <b>{first_stop['name']}</b> "
            f"в <b>{human_time_from_minute(first_stop['minute'])}</b>.\n"
        )

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
    «Отметить прибытие» — фиксируем время и предлагаем TOP-5 остановок
    на основе ПРОГНОЗНОГО времени прибытия, а не расписания.
    """
    user_id = callback.from_user.id
    route_id = get_user_route_id(user_id)

    if not route_id or not get_route(route_id):
        await ask_route_select_callback(callback)
        await callback.answer()
        return

    route = get_route(route_id)
    schedule = route["stops"]
    ids = [s["id"] for s in schedule]

    now_m = now_minute_of_day()
    day = today_str()
    expiry_m = now_m + (SESSION_TTL // 60) + 1

    PRESSED_SESSIONS[user_id] = (now_m, expiry_m, day, route_id)

    eta_map, _, _, _, _, _ = build_eta_with_segments_and_ema(route_id)

    # TOP-5 ближайших остановок по ETA
    diffs = [(sid, abs(eta_map[sid] - now_m)) for sid in ids]
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
    asyncio.create_task(auto_segment_update_loop())
    asyncio.create_task(auto_delay_notifications_loop())
    print("Transport bot 1.3 (critical segments + delay notifications, UTC+3) started.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
