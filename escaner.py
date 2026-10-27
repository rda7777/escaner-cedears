"""Escáner de CEDEARs: baja datos, calcula y escribe los JSON del tablero.

Uso: python3 escaner.py ratios.csv salida/
ratios.csv: columnas cedear,ticker,ratio[,nombre]  (ratio = CEDEARs por acción)
"""
import json, sys, time, math, os, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from analisis import (metricas, detectar_vcp, detectar_taza, senales_ema200, semanal,
                      puntajes, ema, rsi)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
S = requests.Session(); S.headers.update(UA)
CRUMB = None


def chart(sym, rng="5y", interval="1d"):
    for host in ("query1", "query2"):
        try:
            r = S.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                      params={"range": rng, "interval": interval, "includePrePost": "false", "events": "div,splits"}, timeout=20)
            if r.status_code != 200:
                continue
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
            df = pd.DataFrame({"o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"], "v": q["volume"]},
                              index=pd.to_datetime(res["timestamp"], unit="s").normalize())
            if adj:  # ajustar por splits/dividendos
                f = pd.Series(adj, index=df.index) / df.c
                for col in ("o", "h", "l"):
                    df[col] = df[col] * f
                df["c"] = adj
            df = df.dropna(subset=["c"])
            df = df[~df.index.duplicated(keep="last")]
            df["v"] = df.v.fillna(0)
            meta = res.get("meta", {})
            return df, meta
        except Exception:
            continue
    return None, None


def get_crumb():
    global CRUMB
    try:
        S.get("https://fc.yahoo.com", timeout=15)
    except Exception:
        pass
    try:
        r = S.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=15)
        if r.status_code == 200 and r.text and "<" not in r.text:
            CRUMB = r.text.strip()
    except Exception:
        pass
    return CRUMB


MODS = "summaryDetail,financialData,defaultKeyStatistics,calendarEvents,assetProfile,price"


def resumen(sym):
    p = {"modules": MODS}
    if CRUMB:
        p["crumb"] = CRUMB
    for host in ("query2", "query1"):
        try:
            r = S.get(f"https://{host}.finance.yahoo.com/v10/finance/quoteSummary/{sym}", params=p, timeout=20)
            if r.status_code == 200:
                return r.json()["quoteSummary"]["result"][0]
        except Exception:
            pass
    return {}


def raw(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    if isinstance(d, dict):
        d = d.get("raw")
    return d


def ccl_dolarapi():
    try:
        r = requests.get("https://dolarapi.com/v1/dolares/contadoconliqui", timeout=15)
        j = r.json()
        return float(j["venta"]), j.get("fechaActualizacion")
    except Exception:
        return None, None


def r2(x, n=2):
    if x is None:
        return None
    try:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x, n)
    except Exception:
        return x


def main(ratios_csv, out):
    os.makedirs(out, exist_ok=True)
    rat = pd.read_csv(ratios_csv)
    rat["ticker"] = rat.ticker.str.strip().str.upper()
    rat["cedear"] = rat.cedear.str.strip().str.upper()
    rat = rat.drop_duplicates("ticker")

    spy, _ = chart("SPY")
    get_crumb()

    def load(row):
        sym = row.ticker.replace(".", "-")
        df, meta = chart(sym)
        ba, _ = chart(f"{row.cedear}.BA", rng="3mo")
        info = resumen(sym)
        return row, sym, df, meta, ba, info

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(load, [r for r in rat.itertuples()]))

    # CCL implícito: mediana de (precio CEDEAR * ratio / precio subyacente)
    impl = []
    for row, sym, df, meta, ba, info in res:
        if df is None or ba is None or len(ba) < 5 or len(df) < 5:
            continue
        if ba.index[-1] != df.index[-1]:
            continue
        if (ba.c.iloc[-5:] * ba.v.iloc[-5:]).mean() < 5e6:  # poco líquidos fuera
            continue
        impl.append(ba.c.iloc[-1] * row.ratio / df.c.iloc[-1])
    ccl_impl = float(np.median(impl)) if impl else None
    # descartar precios BYMA incoherentes (ratio desactualizado o ticker distinto)
    if ccl_impl:
        res = [(row, sym, df, meta, (ba if (ba is not None and df is not None and len(ba) and len(df) and
                abs(ba.c.iloc[-1] * row.ratio / df.c.iloc[-1] / ccl_impl - 1) < 0.10) else None), info)
               for row, sym, df, meta, ba, info in res]
    ccl_api, ccl_api_at = ccl_dolarapi()
    CCL = ccl_impl or ccl_api

    uni, pats, sens, earn, hist = [], [], [], [], []
    hoy = pd.Timestamp.now('UTC').tz_localize(None).normalize()
    for row, sym, df, meta, ba, info in res:
        if df is None or len(df) < 260:
            continue
        qt = raw(info, "price", "quoteType") or (meta or {}).get("instrumentType")
        if qt and str(qt).upper() in ("ETF", "MUTUALFUND", "INDEX"):
            continue
        try:
            m = metricas(df, spy)
        except Exception as e:
            print("métricas", sym, e); continue
        w = semanal(df)
        setups = {}
        for tf, d in (("D", df), ("W", w)):
            for fn in (detectar_vcp, detectar_taza):
                try:
                    p = fn(d, tf)
                except Exception as e:
                    print("patrón", sym, tf, e); p = None
                if p:
                    p.update({"ticker": row.ticker, "cedear": row.cedear, "marco": "Diario" if tf == "D" else "Semanal"})
                    pats.append(p)
                    setups.setdefault(tf, p)
            try:
                spy_s = spy.c if tf == "D" else semanal(spy).c
                for s in senales_ema200(d, spy_s, tf):
                    s.update({"ticker": row.ticker, "cedear": row.cedear, "marco": "Diario" if tf == "D" else "Semanal",
                              "precio": r2(d.c.iloc[-1])})
                    sens.append(s)
            except Exception as e:
                print("ema", sym, tf, e)
        m["_setup"] = setups.get("D") or setups.get("W")

        # BYMA
        ba_ok = ba is not None and len(ba) >= 2
        m["cedear_ars"] = float(ba.c.iloc[-1]) if ba_ok else None
        m["cedear_var_1d"] = float((ba.c.iloc[-1] / ba.c.iloc[-2] - 1) * 100) if ba_ok else None
        m["cedear_vol_ars_20d"] = float((ba.c * ba.v).iloc[-20:].mean()) if ba_ok else None
        m["cedear_fecha"] = ba.index[-1].strftime("%Y-%m-%d") if ba_ok else None

        # fundamentals
        f = {
            "nombre": raw(info, "price", "longName") or raw(info, "price", "shortName") or getattr(row, "nombre", None) or row.ticker,
            "sector": raw(info, "assetProfile", "sector"), "industria": raw(info, "assetProfile", "industry"),
            "cap_mercado": raw(info, "summaryDetail", "marketCap") or raw(info, "price", "marketCap"),
            "per": raw(info, "summaryDetail", "trailingPE"), "per_fwd": raw(info, "summaryDetail", "forwardPE"),
            "peg": raw(info, "defaultKeyStatistics", "pegRatio"),
            "margen_neto": raw(info, "financialData", "profitMargins"), "roe": raw(info, "financialData", "returnOnEquity"),
            "crec_ingresos": raw(info, "financialData", "revenueGrowth"), "crec_ganancias": raw(info, "financialData", "earningsGrowth"),
            "deuda_patrimonio": raw(info, "financialData", "debtToEquity"),
            "precio_objetivo": raw(info, "financialData", "targetMeanPrice"),
            "recomendacion": raw(info, "financialData", "recommendationKey"),
            "analistas": raw(info, "financialData", "numberOfAnalystOpinions"),
            "div_yield": raw(info, "summaryDetail", "dividendYield"), "beta": raw(info, "summaryDetail", "beta"),
        }
        m.update(f)
        ed = (raw(info, "calendarEvents", "earnings") or {})
        eds = (info.get("calendarEvents", {}).get("earnings", {}) or {}).get("earningsDate") or []
        nxt = None
        for e in eds:
            t = pd.Timestamp(e.get("raw"), unit="s") if e.get("raw") else None
            if t is not None and t.normalize() >= hoy:
                nxt = t; break
        m["proximo_earnings"] = nxt.strftime("%Y-%m-%d") if nxt is not None else None
        if nxt is not None and (nxt.normalize() - hoy).days <= 45:
            ce = info.get("calendarEvents", {}).get("earnings", {})
            earn.append({
                "ticker": row.ticker, "cedear": row.cedear, "nombre": f["nombre"], "fecha": m["proximo_earnings"],
                "dias": int((nxt.normalize() - hoy).days),
                "estimada": bool((info.get("calendarEvents", {}).get("earnings", {}) or {}).get("isEarningsDateEstimate")),
                "eps_estimado": r2(raw(ce, "earningsAverage")), "ingresos_estimados": r2(raw(ce, "revenueAverage"), 0),
                "sector": f["sector"],
            })
        m.update({"ticker": row.ticker, "cedear": row.cedear, "ratio": float(row.ratio), "sym": sym})
        # símbolos para TradingView (prefijo de bolsa obligatorio al importar listas)
        mk = str(getattr(row, "mercado", "") or "").upper()
        pref = "NASDAQ" if "NASDAQ" in mk else "AMEX" if "AMERICAN" in mk else "OTC" if "OTC" in mk else "NYSE"
        m["tv_us"] = f"{pref}:{row.ticker.replace('-', '.')}"
        m["tv_byma"] = "BCBA:TS" if row.cedear == "TEN" else f"BCBA:{row.cedear.replace('-', '.')}"
        uni.append(m)

        # historial compacto para el gráfico del detalle (últimas 180 ruedas)
        h = df.iloc[-180:]
        e200 = ema(df.c, 200).iloc[-180:]; s50 = df.c.rolling(50).mean().iloc[-180:]
        hist.append({"ticker": row.ticker, "inicio": h.index[0].strftime("%Y-%m-%d"),
                     "fechas": ",".join(x.strftime("%Y-%m-%d") for x in h.index),
                     "cierres": ",".join(f"{x:.2f}" for x in h.c),
                     "ema200": ",".join("" if np.isnan(x) else f"{x:.2f}" for x in e200),
                     "sma50": ",".join("" if np.isnan(x) else f"{x:.2f}" for x in s50),
                     "volumen": ",".join(str(int(x)) for x in h.v)})

    # ranking RS (percentil 1-99)
    rs = pd.Series({u["ticker"]: u["rs_raw"] for u in uni}).dropna()
    rk = (rs.rank(pct=True) * 98 + 1).round().astype(int).to_dict()
    for u in uni:
        u["rs_rank"] = rk.get(u["ticker"])
        st = u.pop("_setup")
        t, r_, k, s = puntajes(u, u["rs_rank"] if u["rs_rank"] is not None else float("nan"), st)
        u.update({"p_tendencia": t, "p_fuerza": r_, "p_contraccion": k, "p_setup": s,
                  "puntaje": round(t + r_ + k + s, 1) if u["tendencia_ok"] else 0,
                  "setup": (st["patron"] + " " + st["marco"].lower()) if st else None,
                  "setup_estado": st["estado"] if st else None,
                  "pivote": st["pivote"] if st else None, "stop": st["stop"] if st else None,
                  "volatil": bool(u["adr_pct"] > 6)})
        # precios en pesos
        px = u["precio"]; ratio = u["ratio"]
        u["ccl_implicito"] = (u["cedear_ars"] * ratio / px) if u["cedear_ars"] else None
        u["precio_teorico_ars"] = px * CCL / ratio if CCL else None
        u["prima_ccl"] = (u["cedear_ars"] / u["precio_teorico_ars"] - 1) * 100 if u["cedear_ars"] and u["precio_teorico_ars"] else None
        if u["pivote"]:
            ent = u["pivote"]; stop = u["stop"] if u["stop"] and u["stop"] < ent else ent * 0.92
            stop = max(stop, ent * 0.90)
        else:
            ent = max(px, u["ema200"]) if u["tendencia_ok"] else None; stop = ent * 0.92 if ent else None
        u["entrada_usd"] = ent; u["stop_usd"] = stop
        u["objetivo_usd"] = ent * 1.20 if ent else None
        u["entrada_ars"] = ent * CCL / ratio if ent and CCL else None
        u["entrada_max_ars"] = ent * 1.05 * CCL / ratio if ent and CCL else None
        u["stop_ars"] = stop * CCL / ratio if stop and CCL else None
        u["objetivo_ars"] = u["objetivo_usd"] * CCL / ratio if u["objetivo_usd"] and CCL else None
        u["riesgo_pct"] = (1 - stop / ent) * 100 if ent and stop else None
        u["upside_analistas"] = (u["precio_objetivo"] / px - 1) * 100 if u.get("precio_objetivo") else None
        u["dias_a_earnings"] = int((pd.Timestamp(u["proximo_earnings"]) - hoy).days) if u.get("proximo_earnings") else None
        u["sobrecomprada"] = bool(u["rsi14"] >= 70); u["sobrevendida"] = bool(u["rsi14"] <= 30)

    # limpiar números
    def clean(rows):
        out = []
        for r in rows:
            out.append({k: (r2(v, 4) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, np.bool_) else v)) for k, v in r.items()})
        return out

    rs_by = {u["ticker"]: u["rs_rank"] for u in uni}
    for p in pats:
        p["rs_rank"] = rs_by.get(p["ticker"])
    for s in sens:
        s["rs_rank"] = rs_by.get(s["ticker"])
    for e in earn:
        e["rs_rank"] = rs_by.get(e["ticker"])

    n = len(uni)
    sobre200 = sum(1 for u in uni if u["dist_ema200"] > 0)
    mercado = [{
        "fecha": spy.index[-1].strftime("%Y-%m-%d"),
        "generado": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spy": r2(spy.c.iloc[-1]), "spy_var_1d": r2((spy.c.iloc[-1] / spy.c.iloc[-2] - 1) * 100),
        "spy_var_ytd": r2((spy.c.iloc[-1] / spy.c[spy.index.year < spy.index[-1].year].iloc[-1] - 1) * 100),
        "spy_sobre_ema200": bool(spy.c.iloc[-1] > ema(spy.c, 200).iloc[-1]),
        "spy_rsi14": r2(rsi(spy.c).iloc[-1], 1),
        "ccl": r2(CCL), "ccl_implicito": r2(ccl_impl), "ccl_dolarapi": r2(ccl_api), "ccl_fuente": "implícito CEDEARs" if ccl_impl else "dolarapi",
        "universo": n, "en_tendencia": sum(1 for u in uni if u["tendencia_ok"]),
        "pct_sobre_ema200": r2(sobre200 / n * 100 if n else None, 1),
        "suben": sum(1 for u in uni if u["var_1d"] > 0), "bajan": sum(1 for u in uni if u["var_1d"] < 0),
    }]
    files = {"universo": clean(uni), "patrones": clean(pats), "senales": clean(sens), "earnings": clean(earn),
             "mercado": clean(mercado), "historial": hist}
    for k, v in files.items():
        with open(os.path.join(out, f"{k}.json"), "w") as fh:
            json.dump(v, fh, ensure_ascii=False, separators=(",", ":"))
        print(k, len(v), os.path.getsize(os.path.join(out, f"{k}.json")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
