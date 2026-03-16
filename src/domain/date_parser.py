from __future__ import annotations

import re
from calendar import monthrange
from datetime import date
from typing import List, Optional, Tuple


_DATE_TEXT_RE = re.compile(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})")
_DATE_TEXT_COMPACT_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
_DATE_TEXT_CN_RE = re.compile(
    r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})(?:日|号)"
)
_MONTH_PHASE_RE = re.compile(r"(?:(20\d{2})年)?(\d{1,2})月(上旬|中旬|下旬)")
_DATE_RANGE_CN_PARTIAL_RE = re.compile(
    r"(?:从)?(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})(?:日|号)\s*(?:到|至|-|~|～)\s*(\d{1,2})(?:日|号)"
)
_RECENT_DAYS_RE = re.compile(r"(?:最近|近)(\d{1,2})天")
_FUTURE_DAYS_RE = re.compile(r"(?:未来)(\d{1,2})天")
_RELATIVE_DAYS_FORWARD_RE = re.compile(r"(\d{1,2})天后")
_RELATIVE_DAYS_BACKWARD_RE = re.compile(r"(\d{1,2})天前")


def _extract_month_phase_range(
    text: str, *, today: Optional[date] = None
) -> Optional[Tuple[date, date]]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    anchor = today or date.today()
    match = _MONTH_PHASE_RE.search(prompt)
    if not match:
        return None
    year = int(match.group(1)) if match.group(1) else anchor.year
    month = int(match.group(2))
    phase = match.group(3)
    try:
        month_end_day = monthrange(year, month)[1]
        if phase == "上旬":
            return date(year, month, 1), date(year, month, min(10, month_end_day))
        if phase == "中旬":
            return date(year, month, 11), date(year, month, min(20, month_end_day))
        return date(year, month, 21), date(year, month, month_end_day)
    except ValueError:
        return None


def _extract_partial_cn_date_range(
    text: str, *, today: Optional[date] = None
) -> Optional[Tuple[date, date]]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    anchor = today or date.today()
    match = _DATE_RANGE_CN_PARTIAL_RE.search(prompt)
    if not match:
        return None
    year = int(match.group(1)) if match.group(1) else anchor.year
    month = int(match.group(2))
    start_day = int(match.group(3))
    end_day = int(match.group(4))
    try:
        start = date(year, month, start_day)
        end = date(year, month, end_day)
    except ValueError:
        return None
    if end < start:
        return None
    return start, end


def extract_explicit_dates(text: str, *, today: Optional[date] = None) -> List[date]:
    prompt = str(text or "").strip()
    if not prompt:
        return []
    anchor = today or date.today()
    values: List[date] = []
    for match in _DATE_TEXT_RE.finditer(prompt):
        try:
            values.append(
                date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            )
        except ValueError:
            continue
    for match in _DATE_TEXT_COMPACT_RE.finditer(prompt):
        try:
            values.append(
                date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            )
        except ValueError:
            continue
    for match in _DATE_TEXT_CN_RE.finditer(prompt):
        year = int(match.group(1)) if match.group(1) else anchor.year
        try:
            values.append(date(year, int(match.group(2)), int(match.group(3))))
        except ValueError:
            continue
    return values


def extract_relative_date_range(
    text: str, *, today: Optional[date] = None
) -> Optional[Tuple[date, date]]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    anchor = today or date.today()
    if any(token in prompt for token in ("今天", "今日")):
        return anchor, anchor
    for offset, tokens in (
        (-3, ("大前天",)),
        (-2, ("前天",)),
        (-1, ("昨天", "昨日")),
    ):
        if any(token in prompt for token in tokens):
            target = anchor.fromordinal(anchor.toordinal() + offset)
            return target, target
    for offset, tokens in (
        (3, ("大后天",)),
        (2, ("后天",)),
        (1, ("明天",)),
    ):
        if any(token in prompt for token in tokens):
            target = anchor.fromordinal(anchor.toordinal() + offset)
            return target, target
    match = _RELATIVE_DAYS_BACKWARD_RE.search(prompt)
    if match:
        try:
            days = max(1, min(int(match.group(1)), 30))
            target = anchor.fromordinal(anchor.toordinal() - days)
            return target, target
        except ValueError:
            pass
    match = _RELATIVE_DAYS_FORWARD_RE.search(prompt)
    if match:
        try:
            days = max(1, min(int(match.group(1)), 30))
            target = anchor.fromordinal(anchor.toordinal() + days)
            return target, target
        except ValueError:
            pass
    recent_days: Optional[int] = None
    match = _RECENT_DAYS_RE.search(prompt)
    if match:
        try:
            recent_days = int(match.group(1))
        except ValueError:
            recent_days = None
    elif any(token in prompt for token in ("最近一周", "近一周", "最近7天", "近7天")):
        recent_days = 7
    elif any(token in prompt for token in ("最近半个月", "近半个月", "最近15天", "近15天")):
        recent_days = 15
    elif any(token in prompt for token in ("最近", "近期", "近来")):
        recent_days = 7
    if recent_days is not None:
        recent_days = max(1, min(recent_days, 30))
        start_date = anchor
        end_date = start_date.fromordinal(start_date.toordinal() + recent_days - 1)
        return start_date, end_date

    future_days: Optional[int] = None
    match = _FUTURE_DAYS_RE.search(prompt)
    if match:
        try:
            future_days = int(match.group(1))
        except ValueError:
            future_days = None
    elif any(token in prompt for token in ("未来一周", "未来7天")):
        future_days = 7
    elif any(token in prompt for token in ("未来三天", "未来3天")):
        future_days = 3
    if future_days is not None:
        future_days = max(1, min(future_days, 30))
        start_date = anchor
        end_date = start_date.fromordinal(start_date.toordinal() + future_days - 1)
        return start_date, end_date

    offset_weeks: Optional[int] = None
    if any(token in prompt for token in ("下下周", "下下星期", "下下个星期")):
        offset_weeks = 2
    elif any(token in prompt for token in ("下周", "下星期", "下个星期")):
        offset_weeks = 1
    elif any(token in prompt for token in ("本周", "这周", "本星期", "这星期")):
        offset_weeks = 0
    elif any(token in prompt for token in ("上周", "上星期", "上个星期")):
        offset_weeks = -1
    if offset_weeks is not None:
        week_start = anchor.fromordinal(
            anchor.toordinal() - anchor.weekday() + offset_weeks * 7
        )
        week_end = week_start.fromordinal(week_start.toordinal() + 6)
        return week_start, week_end
    if "周末" in prompt:
        week_start = anchor.fromordinal(anchor.toordinal() - anchor.weekday())
        weekend_start = week_start.fromordinal(week_start.toordinal() + 5)
        weekend_end = week_start.fromordinal(week_start.toordinal() + 6)
        return weekend_start, weekend_end

    if any(token in prompt for token in ("这个月", "本月", "这月")):
        month_start = anchor.replace(day=1)
        month_end = anchor.replace(day=monthrange(anchor.year, anchor.month)[1])
        if (month_end - month_start).days >= 30:
            month_end = month_start.fromordinal(month_start.toordinal() + 29)
        return month_start, month_end
    if any(token in prompt for token in ("下个月", "下月")):
        year = anchor.year + (1 if anchor.month == 12 else 0)
        month = 1 if anchor.month == 12 else anchor.month + 1
        month_start = date(year, month, 1)
        month_end = date(year, month, monthrange(year, month)[1])
        if "月初" in prompt:
            return month_start, date(year, month, min(10, month_end.day))
        if any(token in prompt for token in ("月底", "月末")):
            return date(year, month, max(1, month_end.day - 9)), month_end
        if (month_end - month_start).days >= 30:
            month_end = month_start.fromordinal(month_start.toordinal() + 29)
        return month_start, month_end
    if "月初" in prompt:
        month_start = anchor.replace(day=1)
        month_end = anchor.replace(
            day=min(10, monthrange(anchor.year, anchor.month)[1])
        )
        return month_start, month_end
    if any(token in prompt for token in ("月底", "月末")):
        month_end = anchor.replace(day=monthrange(anchor.year, anchor.month)[1])
        month_start = anchor.replace(day=max(1, month_end.day - 9))
        return month_start, month_end
    return None


def extract_date_range(text: str, *, today: Optional[date] = None) -> Optional[Tuple[date, date]]:
    explicit_dates = extract_explicit_dates(text, today=today)
    if len(explicit_dates) >= 2:
        return explicit_dates[0], explicit_dates[1]
    partial_cn_range = _extract_partial_cn_date_range(text, today=today)
    if partial_cn_range is not None:
        return partial_cn_range
    if len(explicit_dates) == 1:
        return explicit_dates[0], explicit_dates[0]
    month_phase_range = _extract_month_phase_range(text, today=today)
    if month_phase_range is not None:
        return month_phase_range
    return extract_relative_date_range(text, today=today)
