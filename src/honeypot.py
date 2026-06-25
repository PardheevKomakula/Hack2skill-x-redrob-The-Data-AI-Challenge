"""
Detects internally-inconsistent ("honeypot") profiles using only facts
present in the record — no external data needed. Returns a list of
reasons; an empty list means the profile passed all checks.

This is deliberately conservative: each check is something a careful
human recruiter would actually notice, matching the JD's own description
of honeypots ("8 years of experience claimed alongside skill durations
or company tenures that don't add up", "expert proficiency in skills
listed with ~0 months of use").
"""
from .features import parse_date


def detect_honeypot_flags(candidate: dict) -> list:
    flags = []
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0) or 0
    max_months_possible = years_exp * 12 + 6  # small buffer for rounding

    # 1. Career history date / duration consistency
    for job in candidate.get("career_history", []):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        duration = job.get("duration_months", 0) or 0
        is_current = job.get("is_current", False)

        if is_current and end is not None:
            flags.append("is_current=true but end_date is set")

        if start and end:
            computed_months = (end.year - start.year) * 12 + (end.month - start.month)
            if computed_months < 0:
                flags.append("career_history: end_date before start_date")
            elif abs(computed_months - duration) > 3:
                flags.append("career_history: duration_months inconsistent with dates")

        if duration > max_months_possible:
            flags.append("career_history: single role duration exceeds total years_of_experience")

    # 2. Skill duration / proficiency consistency
    for skill in candidate.get("skills", []):
        duration = skill.get("duration_months", 0) or 0
        proficiency = skill.get("proficiency", "")
        endorsements = skill.get("endorsements", 0) or 0

        if duration > max_months_possible:
            flags.append(f"skill '{skill.get('name')}': duration exceeds total experience")

        if proficiency == "expert" and duration < 6:
            flags.append(f"skill '{skill.get('name')}': expert proficiency with <6 months use")

        if proficiency in ("advanced", "expert") and endorsements == 0 and duration < 3:
            flags.append(f"skill '{skill.get('name')}': high proficiency, zero endorsements, near-zero duration")

    # 3. Seniority-vs-experience mismatch
    title = (profile.get("current_title") or "").lower()
    if years_exp < 1.5 and any(t in title for t in ("senior", "staff", "principal", "lead", "director", "head of")):
        flags.append("seniority title with <1.5 years total experience")

    # 4. Education year sanity
    for edu in candidate.get("education", []):
        start_y, end_y = edu.get("start_year"), edu.get("end_year")
        if start_y and end_y and end_y < start_y:
            flags.append("education: end_year before start_year")
        if start_y and end_y and (end_y - start_y) > 10:
            flags.append("education: implausible duration (>10 years for one degree)")

    # 5. redrob_signals internal sanity
    sig = candidate.get("redrob_signals", {})
    sd, lad = parse_date(sig.get("signup_date")), parse_date(sig.get("last_active_date"))
    if sd and lad and lad < sd:
        flags.append("redrob_signals: last_active_date before signup_date")

    # 6. years_of_experience vs actual career_history span/sum mismatch.
    # A candidate's structured years_of_experience field should roughly match
    # the sum of their career_history durations (allowing a generous buffer
    # for employment gaps, education, etc). A large gap — claiming far more
    # total experience than their own career history accounts for — is the
    # canonical "experience inflation" honeypot.
    total_history_months = sum((job.get("duration_months", 0) or 0) for job in candidate.get("career_history", []))
    claimed_months = years_exp * 12
    if claimed_months > 0 and total_history_months > 0:
        if claimed_months > total_history_months * 1.5 and (claimed_months - total_history_months) > 24:
            flags.append(
                f"years_of_experience ({years_exp} yrs) far exceeds career_history span "
                f"({total_history_months / 12:.1f} yrs accounted for)"
            )

    # 7. Identical career_history description reused twice within the SAME
    # candidate's own history (different jobs, different companies/eras
    # describing literally the same paragraph) — internally inconsistent
    # regardless of whether the duration math also breaks.
    descriptions = [job.get("description", "").strip() for job in candidate.get("career_history", []) if job.get("description")]
    seen_desc = set()
    for d in descriptions:
        if d in seen_desc:
            flags.append("career_history: identical description reused across two different roles for this candidate")
        seen_desc.add(d)

    return flags


def is_honeypot(candidate: dict, min_flags: int = 1) -> bool:
    return len(detect_honeypot_flags(candidate)) >= min_flags