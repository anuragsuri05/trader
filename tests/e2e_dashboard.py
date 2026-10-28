"""End-to-end test of the dashboard and every engine, against a RUNNING dashboard.

    python3 -m deltabot dashboard --port 8060      # ideally in a scratch copy of the project
    python3 tests/e2e_dashboard.py http://127.0.0.1:8060 [--with-ai]

Paper only. Resets the paper account of the dashboard it talks to, so don't point it at a session you care about.
--with-ai also runs one real decision from each AI trader (uses Claude plan limits + a little OpenAI credit).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else "http://127.0.0.1:8060").rstrip("/")
WITH_AI = "--with-ai" in sys.argv
RESULTS: list[tuple[str, bool, str]] = []


def call(path: str, body: dict | None = None, timeout: float = 120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get("Content-Type", "").startswith("application/json") else raw.decode()


def wait(cond, timeout: float, every: float = 3):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = call("/api/state")
        if cond(last):
            return last
        time.sleep(every)
    return last


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    return ok


def reset(**conf):
    base = {"start_inr": 5000, "target_inr": 5100, "inr_rate": 95, "instruments": "perpetual", "risk_per_trade": 0.01,
            "timeframe": "auto", "session_minutes": 30, "engine": "arena", "aggression": "normal"}
    base.update(conf)
    call("/api/bot/stop", {})
    call("/api/reset", base, timeout=180)


def run_engine(label: str, ready, timeout: float = 240, **conf):
    reset(**conf)
    call("/api/bot/start", {})
    s = wait(lambda s: s["bot"]["running"] and ready(s), timeout)
    ok = check(f"engine {label}: starts and works", s["bot"]["running"] and ready(s) and not s["bot"]["error"],
               f"error={s['bot']['error']!r}")
    call("/api/bot/stop", {}, timeout=90)
    s = call("/api/state")
    check(f"engine {label}: stops", not s["bot"]["running"])
    return s if ok else None


def main():
    t0 = time.time()
    # ---------------------------------------------------------------- basics
    html = call("/")
    check("page loads", "Deltabot" in html and "planBtn" in html)
    s = call("/api/state")
    check("state endpoint", all(k in s for k in ("equity_inr", "conf", "bot", "arena", "plan", "planning", "logs")))
    ch = call("/api/chart?symbol=BTCUSD")
    year = time.gmtime(ch["candles"][-1]["time"]).tm_year if ch.get("candles") else 0
    check("live chart", len(ch.get("candles", [])) > 100 and len(ch["ema_fast"]) > 50 and year >= 2025,
          f"{len(ch.get('candles', []))} candles, last year {year}")
    r = call("/api/connect", {"key": "bad", "secret": "bad", "region": "india"})
    check("Delta connect rejects a bad key", r.get("ok") is False and r.get("error"))
    r = call("/api/anthropic", {"key": "sk-ant-bad"})
    check("Anthropic key form rejects a bad key", r.get("ok") is False and r.get("error"))

    # ---------------------------------------------------------------- replay
    call("/api/replay", {"days": 30})
    end = time.time() + 240
    rp = {}
    while time.time() < end:
        rp = call("/api/replay")
        if rp["status"] in ("done", "error"):
            break
        time.sleep(3)
    res = rp.get("result") or {}
    check("replay runs", rp.get("status") == "done" and res.get("curve"), f"{len(res.get('trades', []))} trades")
    if res.get("trades"):
        sym = res["trades"][-1]["symbol"]
        rc = call(f"/api/chart?symbol={sym}&source=replay")
        check("replay chart with trade markers", len(rc["candles"]) > 100 and len(rc["markers"]) >= 2, sym)

    # ---------------------------------------------------------------- engines
    run_engine("single bot", lambda s: s["bot"]["last_scan"] > 0, engine="single")
    run_engine("arena normal", lambda s: s["arena"] and all(b["scan"].get("time") for b in s["arena"]["bots"]
                                                            if b["name"] != "claude"), engine="arena")
    run_engine("arena target mode", lambda s: s["arena"] and all(b["scan"].get("time") for b in s["arena"]["bots"]
                                                                 if b["name"] != "claude"),
               engine="arena", aggression="target")

    # session rules
    reset(engine="arena", aggression="target", target_inr=4000)            # below the start -> instantly reached
    call("/api/bot/start", {})
    s = wait(lambda s: (s["bot"]["halted"] or "").startswith("target:"), 120)
    check("target reached -> stops and closes everything", (s["bot"]["halted"] or "").startswith("target:")
          and not s["open"], s["bot"]["halted"][:60])
    call("/api/bot/stop", {})
    reset(engine="arena", aggression="target", session_minutes=1)
    call("/api/bot/start", {})
    s = wait(lambda s: (s["bot"]["halted"] or "").startswith("session:"), 180)
    check("session time up -> stops and closes everything", (s["bot"]["halted"] or "").startswith("session:")
          and not s["open"], s["bot"]["halted"][:60])
    call("/api/bot/stop", {})

    # ---------------------------------------------------------------- plan
    reset(engine="arena", aggression="target", target_inr=5500, session_minutes=60)
    call("/api/plan", {})
    s = wait(lambda s: s["planning"]["status"] in ("done", "error"), 300, 5)
    pl = s["plan"] or {}
    check("Plan & analyse", s["planning"]["status"] == "done" and pl.get("tested", 0) >= 100 and pl.get("risk_table"),
          f"tested {pl.get('tested')}, chance {pl.get('p_target', 0):.1%} at {pl.get('recommended_risk', 0):.0%} risk, "
          f"{len(pl.get('active', []))} strategies, {len(pl.get('backups', []))} backups")
    check("plan matches settings (not stale)", pl.get("stale") is False)
    r = call("/api/plan/use", {})
    check("Use this plan", r.get("ok") and call("/api/state")["conf"]["engine"] == "plan")
    call("/api/bot/start", {})
    s = wait(lambda s: s["arena"] and (s["arena"].get("monitor") or {}).get("checks", 0) >= 3
             and s["arena"]["monitor"].get("live_chance") is not None, 240)
    mon = (s["arena"] or {}).get("monitor") or {}
    check("plan engine: 5-second monitor + live chance", mon.get("checks", 0) >= 3 and mon.get("live_chance") is not None,
          f"checks {mon.get('checks')}, live chance {mon.get('live_chance')}, backups {mon.get('backups_left')}")
    call("/api/bot/stop", {}, timeout=90)
    call("/api/reset", {"engine": "plan", "target_inr": 6000, "start_inr": 5000, "session_minutes": 60})
    call("/api/bot/start", {})
    s = call("/api/state")
    check("stale plan blocks Start", not s["bot"]["running"] and "Plan & analyse" in (s["bot"]["error"] or ""),
          s["bot"]["error"][:60])

    # ---------------------------------------------------------------- all-weather portfolio
    reset(engine="allweather", start_inr=950000, target_inr=1000000, session_minutes=60)
    s = call("/api/state")
    bt = (s.get("allweather") or {}).get("backtest") or {}
    check("all-weather backtest loaded", bool(bt.get("portfolio")) and len(bt.get("curve", [])) > 100,
          f"{bt.get('period')} CAGR {bt.get('portfolio', {}).get('cagr', 0):.1%}")
    call("/api/bot/start", {})
    s = wait(lambda s: (s.get("allweather") or {}).get("last_rebalance"), 900 if WITH_AI else 240, 5)
    v = s.get("allweather") or {}
    check("all-weather: first rebalance builds trend book + carry", v.get("last_rebalance") and len(v.get("targets", {})) >= 5
          and len(v.get("carry", {})) >= 3, f"{len(v.get('targets', {}))} trend coins, carry on "
          f"{sum(1 for x in v.get('carry', {}).values() if x['on'])}, error={s['bot']['error']!r}")
    if WITH_AI:
        com = v.get("committee") or {}
        check("all-weather: AI risk committee decided", com.get("time") and com.get("risk_scale") is not None,
              f"scale {com.get('risk_scale')}, carry_ok {com.get('carry_ok')}, errors {com.get('errors')}")
    call("/api/bot/stop", {}, timeout=120)

    # ---------------------------------------------------------------- survival agent
    reset(engine="survival", start_inr=4750, target_inr=475000, session_minutes=0, rent_month=200)
    call("/api/bot/start", {})
    need = ("scout", "quant") if WITH_AI else ()
    s = wait(lambda s: (s.get("survival") or {}).get("desks") and all(
        (s["survival"]["desks"][k].get("calls") or 0) >= 1 for k in need)
        and s["survival"]["desks"]["trend"]["detail"] != "0 positions", 600 if WITH_AI else 180, 5)
    v = s.get("survival") or {}
    check("survival agent: alive, core desks built, rent ticking", v.get("alive") and v["desks"]["trend"]["detail"] != "0 positions"
          and v.get("rent", 0) > 0, f"balance ${v.get('equity', 0):.2f}, rent ${v.get('rent', 0):.4f}, {v['desks']['trend']['detail']}")
    if WITH_AI:
        for k in need:
            d = v["desks"][k]
            check(f"survival agent: {d['label']} desk decided without error", not d.get("error"), (d.get("read") or d.get("error") or "")[:90])
    call("/api/bot/stop", {}, timeout=120)

    # ---------------------------------------------------------------- AI duel
    if WITH_AI:
        s = run_engine("AI duel (Fable + ChatGPT decide once each)",
                       lambda s: s["arena"] and s["arena"].get("ai") and all(t["last"].get("time") for t in s["arena"]["ai"]),
                       timeout=300, engine="ai", aggression="target", target_inr=5500, session_minutes=60)
        if s:
            for t in s["arena"]["ai"]:
                check(f"AI {t['label']} decision without error", not t["last"].get("error"),
                      (t["last"].get("view") or t["last"].get("error") or "")[:90])

    # ---------------------------------------------------------------- summary
    call("/api/bot/stop", {})
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed in {time.time() - t0:.0f}s")
    for name, _, detail in failed:
        print(f"  FAILED: {name} {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
