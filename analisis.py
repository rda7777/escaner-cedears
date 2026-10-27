"""Cálculos técnicos del escáner de CEDEARs (sin red).

Cada función recibe un DataFrame OHLCV con índice de fechas y columnas
o, h, l, c, v (precios del subyacente en USD).
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- básicos
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def true_range(df):
    pc = df.c.shift()
    return pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)


def semanal(df):
    w = df.resample("W-FRI").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    return w.dropna(subset=["c"])


def ret(c, n):
    if len(c) <= n:
        return np.nan
    return c.iloc[-1] / c.iloc[-1 - n] - 1


def fecha(ix):
    return pd.Timestamp(ix).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- zigzag
def zigzag(h, l, th):
    """Pivotes alternados (idx, precio, 'P' pico | 'V' valle) con umbral th (fracción)."""
    n = len(h)
    if n < 3:
        return []
    piv = []
    # dirección inicial
    hi_i = lo_i = 0
    mode = None
    i = 1
    while i < n and mode is None:
        if h[i] > h[hi_i]:
            hi_i = i
        if l[i] < l[lo_i]:
            lo_i = i
        if hi_i > lo_i and h[hi_i] >= l[lo_i] * (1 + th):
            piv.append((lo_i, l[lo_i], "V")); mode = "up"
        elif lo_i > hi_i and l[lo_i] <= h[hi_i] * (1 - th):
            piv.append((hi_i, h[hi_i], "P")); mode = "down"
        i += 1
    if mode is None:
        return [(int(np.argmax(h)), h[int(np.argmax(h))], "P")]
    ext = hi_i if mode == "up" else lo_i
    for j in range(i, n):
        if mode == "up":
            if h[j] >= h[ext]:
                ext = j
            elif l[j] <= h[ext] * (1 - th):
                piv.append((ext, h[ext], "P")); mode = "down"; ext = j
        else:
            if l[j] <= l[ext]:
                ext = j
            elif h[j] >= l[ext] * (1 + th):
                piv.append((ext, l[ext], "V")); mode = "up"; ext = j
    piv.append((ext, h[ext], "P") if mode == "up" else (ext, l[ext], "V"))
    return piv


# ---------------------------------------------------------------- VCP
VCP_CFG = {
    "D": dict(win=130, th=0.035, min_len=15, first_max=0.40, first_min=0.08, last_max=0.10, near=0.08, brk=5),
    "W": dict(win=52, th=0.06, min_len=5, first_max=0.50, first_min=0.12, last_max=0.15, near=0.10, brk=2),
}


def detectar_vcp(df, tf="D"):
    k = VCP_CFG[tf]
    d = df.iloc[-k["win"]:]
    if len(d) < k["min_len"] + 5:
        return None
    h, l, c, v = d.h.values, d.l.values, d.c.values, d.v.values
    end = len(d) - 1
    # inicio de la base: máximo absoluto de la ventana sin contar las últimas barras
    start = int(np.argmax(h[: end - 2]))
    if end - start < k["min_len"]:
        return None
    piv = zigzag(h[start:], l[start:], k["th"])
    piv = [(i + start, p, t) for i, p, t in piv]
    if not piv or piv[0][2] != "P":
        piv = [(start, h[start], "P")] + [x for x in piv if x[0] > start]
    contr = []  # (idx_pico, pico, idx_valle, valle, prof)
    for j in range(len(piv) - 1):
        if piv[j][2] == "P" and piv[j + 1][2] == "V":
            i1, p1, _ = piv[j]
            i2, p2, _ = piv[j + 1]
            contr.append((i1, p1, i2, p2, (p1 - p2) / p1))
    if len(contr) < 2:
        return None
    depths = [x[4] for x in contr]
    if not (k["first_min"] <= depths[0] <= k["first_max"]):
        return None
    for a, b in zip(depths, depths[1:]):
        if b > a * 1.05:
            return None
    if depths[-1] > k["last_max"]:
        return None
    # pivote = máximo desde el último pico de contracción
    last_peak_i = contr[-1][0]
    pivot = float(contr[-1][1])
    last_low = float(contr[-1][3])
    if c[end] < last_low:
        return None
    # volumen: última contracción vs primera
    v1 = v[contr[0][0]: contr[0][2] + 1].mean()
    v2 = v[contr[-1][0]: max(contr[-1][2], end) + 1].mean()
    seca = bool(v2 < v1)
    dist = c[end] / pivot - 1
    estado = "En formación"
    vavg = v[max(0, end - 50): end].mean()
    brk_idx = [i for i in range(max(last_peak_i + 1, end - k["brk"] + 1), end + 1) if c[i] > pivot]
    if brk_idx:
        if dist > 0.07:
            return None  # ya extendida
        i0 = brk_idx[0]
        estado = "Ruptura con volumen" if v[i0] > 1.4 * vavg else "Ruptura"
    elif dist < -k["near"]:
        return None
    return {
        "patron": "VCP",
        "inicio_base": fecha(d.index[start]),
        "semanas_base": round((end - start) / (5 if tf == "D" else 1), 1),
        "contracciones": len(contr),
        "profundidades": " → ".join(f"{x*100:.0f}%" for x in depths),
        "ultima_contraccion": round(depths[-1] * 100, 1),
        "pivote": round(pivot, 2),
        "stop": round(last_low, 2),
        "dist_pivote": round(dist * 100, 1),
        "volumen_seco": seca,
        "estado": estado,
    }


# ---------------------------------------------------------------- taza con asa
CUP_CFG = {
    "D": dict(look=250, h_min=5, h_max=25, cup_min=30, dmin=0.12, dmax=0.40, hmax=0.15),
    "W": dict(look=65, h_min=1, h_max=6, cup_min=6, dmin=0.12, dmax=0.45, hmax=0.18),
}


def detectar_taza(df, tf="D"):
    k = CUP_CFG[tf]
    d = df.iloc[-k["look"]:]
    if len(d) < k["cup_min"] + k["h_max"]:
        return None
    h, l, c, v = d.h.values, d.l.values, d.c.values, d.v.values
    end = len(d) - 1
    for r in range(end - k["h_max"], end - k["h_min"] + 1):
        if r <= k["cup_min"]:
            continue
        R = h[r]
        if R < h[r: end + 1].max():
            continue
        hl_rel = int(np.argmin(l[r + 1: end + 1])) + r + 1 if r < end else r
        hlow = l[hl_rel]
        hdepth = (R - hlow) / R
        if not (0.03 <= hdepth <= k["hmax"]):
            continue
        left = h[: r - k["cup_min"] + 1]
        if len(left) == 0:
            continue
        li = int(np.argmax(left))
        L = h[li]
        bi = li + int(np.argmin(l[li: r + 1]))
        B = l[bi]
        depth = (L - B) / L
        if not (k["dmin"] <= depth <= k["dmax"]):
            continue
        if not (0.90 * L <= R <= 1.05 * L):
            continue
        w = r - li
        if not (0.2 <= (bi - li) / w <= 0.8):
            continue
        if hlow < B + 0.5 * (L - B):
            continue
        # tendencia previa
        pre = l[max(0, li - (120 if tf == "D" else 26)): li + 1]
        if len(pre) > 5 and L < 1.15 * pre.min():
            continue
        if c[end] > R * 1.05 or c[end] < hlow:
            continue
        vh = v[r + 1: end + 1].mean() if end > r else np.nan
        vc = v[li: r + 1].mean()
        estado = "Ruptura" if c[end] > R else "Formando asa"
        return {
            "patron": "Taza con asa",
            "inicio_base": fecha(d.index[li]),
            "semanas_base": round((end - li) / (5 if tf == "D" else 1), 1),
            "contracciones": None,
            "profundidades": f"taza {depth*100:.0f}% · asa {hdepth*100:.0f}%",
            "ultima_contraccion": round(hdepth * 100, 1),
            "pivote": round(float(R), 2),
            "stop": round(float(hlow), 2),
            "dist_pivote": round((c[end] / R - 1) * 100, 1),
            "volumen_seco": bool(vh < vc) if not np.isnan(vh) else None,
            "estado": estado,
        }
    return None


# ---------------------------------------------------------------- EMA200
EMA_CFG = {"D": dict(look=20, tol=0.015, gap=3, prev=30), "W": dict(look=8, tol=0.02, gap=1, prev=10)}


def senales_ema200(df, spy_c, tf="D"):
    """Rebotes y cruces de la EMA200. Devuelve lista de dicts."""
    k = EMA_CFG[tf]
    if len(df) < 230:
        return []
    e = ema(df.c, 200)
    rsl = (df.c / spy_c.reindex(df.index).ffill())
    vavg = df.v.rolling(50).mean()
    c, l, h, v = df.c.values, df.l.values, df.h.values, df.v.values
    ev = e.values
    n = len(df) - 1
    out = []
    base = dict()

    def rs_cambio(i):
        a, b = rsl.iloc[i], rsl.iloc[-1]
        return round((b / a - 1) * 100, 1) if a and not np.isnan(a) else None

    def clim(i):
        r = v[i] / vavg.iloc[i] if vavg.iloc[i] else np.nan
        if np.isnan(r):
            return None, None
        tag = "Alto volumen" if r >= 1.5 else ("Bajo volumen" if r <= 0.8 else "Volumen normal")
        return round(float(r), 2), tag

    # --- cruce (alcista o bajista) en las últimas barras
    for i in range(n - k["look"] + 1, n + 1):
        if np.isnan(ev[i - 1]):
            continue
        up = c[i - 1] < ev[i - 1] and c[i] > ev[i]
        dn = c[i - 1] > ev[i - 1] and c[i] < ev[i]
        if up or dn:
            last_cross = (i, "Cruce alcista" if up else "Cruce bajista")
    try:
        i, tipo = last_cross
        still = (c[n] > ev[n]) if tipo == "Cruce alcista" else (c[n] < ev[n])
        if still:
            # cuánto estuvo del otro lado
            j = i - 1
            while j > 0 and ((c[j] < ev[j]) if tipo == "Cruce alcista" else (c[j] > ev[j])):
                j -= 1
            vr, vt = clim(i)
            out.append({
                "tipo": tipo, "contacto": fecha(df.index[i]), "ultimo_contacto": fecha(df.index[i]),
                "contacto_previo": None, "barras_desde": n - i, "barras_otro_lado": i - 1 - j,
                "dist_ema": round((c[n] / ev[n] - 1) * 100, 1), "rs_cambio": rs_cambio(i),
                "vol_ratio": vr, "climax": vt,
                "ema_pendiente": round((ev[n] / ev[max(0, n - 20)] - 1) * 100, 2),
            })
    except NameError:
        pass

    # --- rebote: tocó la EMA desde arriba y hoy cierra por encima
    touch = [i for i in range(n - k["look"] + 1, n + 1)
             if not np.isnan(ev[i]) and l[i] <= ev[i] * (1 + k["tol"]) and h[i] >= ev[i] * (1 - k["tol"])]
    if touch and c[n] > ev[n] and n - touch[-1] >= 1:
        last = touch[-1]
        first = last
        for t in reversed(touch[:-1]):
            if first - t <= k["gap"] + 1:
                first = t
            else:
                break
        # antes del episodio venía por encima
        pre = range(max(0, first - k["prev"]), first)
        above = np.mean([c[j] > ev[j] for j in pre if not np.isnan(ev[j])]) if len(pre) else 0
        slope = ev[n] / ev[max(0, n - 20)] - 1
        if above >= 0.7 and slope >= -0.002 and c[n] > c[last]:
            # contacto previo (episodio anterior)
            prev_touch = None
            for j in range(first - k["prev"], max(200, 0), -1):
                if j < 0 or np.isnan(ev[j]):
                    break
                if l[j] <= ev[j] * (1 + k["tol"]) and h[j] >= ev[j] * (1 - k["tol"]):
                    prev_touch = fecha(df.index[j]); break
            ep = range(first, last + 1)
            imax = max(ep, key=lambda j: v[j])
            vr, vt = clim(imax)
            out.append({
                "tipo": "Rebote", "contacto": fecha(df.index[first]), "ultimo_contacto": fecha(df.index[last]),
                "contacto_previo": prev_touch, "barras_desde": n - last, "barras_otro_lado": None,
                "dist_ema": round((c[n] / ev[n] - 1) * 100, 1), "rs_cambio": rs_cambio(first),
                "vol_ratio": vr, "climax": vt, "ema_pendiente": round(slope * 100, 2),
                "rebote_pct": round((c[n] / min(l[j] for j in ep) - 1) * 100, 1),
            })
    for o in out:
        o.setdefault("rebote_pct", None)
    return out


# ---------------------------------------------------------------- métricas por ticker
def metricas(df, spy):
    c = df.c
    m = {}
    m["precio"] = float(c.iloc[-1])
    m["var_1d"] = ret(c, 1) * 100
    m["var_5d"] = ret(c, 5) * 100
    m["var_1m"] = ret(c, 21) * 100
    m["var_3m"] = ret(c, 63) * 100
    m["var_6m"] = ret(c, 126) * 100
    m["var_12m"] = ret(c, 252) * 100
    m["var_ytd"] = (c.iloc[-1] / c[c.index.year < c.index[-1].year].iloc[-1] - 1) * 100 if (c.index.year < c.index[-1].year).any() else np.nan
    e200 = ema(c, 200); s50 = c.rolling(50).mean(); s150 = c.rolling(150).mean(); s200 = c.rolling(200).mean()
    m["ema200"] = float(e200.iloc[-1]); m["sma50"] = float(s50.iloc[-1])
    m["sma150"] = float(s150.iloc[-1]); m["sma200"] = float(s200.iloc[-1])
    m["dist_ema200"] = (c.iloc[-1] / e200.iloc[-1] - 1) * 100
    m["rsi14"] = float(rsi(c).iloc[-1])
    hi52 = df.h.iloc[-252:].max(); lo52 = df.l.iloc[-252:].min()
    m["max52"] = float(hi52); m["min52"] = float(lo52)
    m["dist_max52"] = (c.iloc[-1] / hi52 - 1) * 100
    m["sobre_min52"] = (c.iloc[-1] / lo52 - 1) * 100
    tr = true_range(df)
    m["atr_pct"] = float(tr.rolling(14).mean().iloc[-1] / c.iloc[-1] * 100)
    m["adr_pct"] = float(((df.h / df.l - 1).rolling(20).mean().iloc[-1]) * 100)
    m["atr_ratio"] = float(tr.rolling(10).mean().iloc[-1] / tr.rolling(50).mean().iloc[-1])
    m["rango10"] = float((df.h.iloc[-10:].max() / df.l.iloc[-10:].min() - 1) * 100)
    m["vol_ratio"] = float(df.v.iloc[-10:].mean() / df.v.iloc[-50:].mean()) if df.v.iloc[-50:].mean() else np.nan
    m["vol_hoy"] = float(df.v.iloc[-1] / df.v.iloc[-51:-1].mean()) if df.v.iloc[-51:-1].mean() else np.nan
    m["vol_usd_20d"] = float((df.c * df.v).iloc[-20:].mean())

    # fuerza relativa vs SPY
    sc = spy.c.reindex(df.index).ffill()
    rel = {}
    for nm, n in (("3m", 63), ("6m", 126), ("9m", 189), ("12m", 252)):
        a, b = ret(c, n), ret(sc, n)
        rel[nm] = ((1 + a) / (1 + b) - 1) * 100 if not (np.isnan(a) or np.isnan(b)) else np.nan
    m["rs_vs_spy_3m"] = rel["3m"]; m["rs_vs_spy_6m"] = rel["6m"]; m["rs_vs_spy_12m"] = rel["12m"]
    vals = [rel[x] for x in ("3m", "3m", "6m", "9m", "12m")]
    m["rs_raw"] = np.nanmean(vals) if not all(np.isnan(vals)) else np.nan
    rsl = c / sc
    m["rs_linea_max"] = bool(rsl.iloc[-1] >= rsl.iloc[-252:].max() * 0.98)
    m["rs_linea_1m"] = float((rsl.iloc[-1] / rsl.iloc[-22] - 1) * 100) if len(rsl) > 22 else np.nan

    # plantilla de tendencia (Minervini)
    C = c.iloc[-1]
    s200_up = s200.iloc[-1] > s200.iloc[-22] if len(s200.dropna()) > 22 else False
    crit = [
        C > s150.iloc[-1] and C > s200.iloc[-1],
        s150.iloc[-1] > s200.iloc[-1],
        bool(s200_up),
        s50.iloc[-1] > s150.iloc[-1] and s50.iloc[-1] > s200.iloc[-1],
        C > s50.iloc[-1],
        C >= 1.25 * lo52,
        C >= 0.75 * hi52,
    ]
    m["tendencia_criterios"] = int(sum(bool(x) for x in crit))
    e_up = e200.iloc[-1] > e200.iloc[-21]
    m["tendencia_ok"] = bool(C > e200.iloc[-1] and s50.iloc[-1] > s200.iloc[-1] and e_up)
    return m


def puntajes(m, rs_rank, setup):
    """4 pilares de 0 a 25."""
    t = 25 * m["tendencia_criterios"] / 7 if m["tendencia_ok"] else 0
    r = 25 * (rs_rank / 99) if rs_rank == rs_rank else 0
    if m.get("rs_linea_max"):
        r = min(25, r + 2)
    s_ratio = np.clip((1.2 - m["atr_ratio"]) / 0.6, 0, 1)
    s_tight = np.clip((15 - m["rango10"]) / 10, 0, 1)
    pen = 1.0 if m["adr_pct"] <= 4 else (0.6 if m["adr_pct"] <= 6 else 0.25)
    k = 25 * (0.5 * s_ratio + 0.5 * s_tight) * pen
    s = 0.0
    if setup:
        s = 12
        if setup.get("volumen_seco"):
            s += 4
        dp = setup.get("dist_pivote") or -99
        if -5 <= dp <= 3:
            s += 6
        elif -10 <= dp < -5:
            s += 3
        if setup.get("contracciones") and setup["contracciones"] >= 3:
            s += 3
        if "Ruptura con volumen" in (setup.get("estado") or ""):
            s += 3
        s = min(25, s)
    else:
        # formando base: cerca del máximo, volumen seco, rango estrecho
        if m["dist_max52"] >= -12:
            s += 5
        if m["vol_ratio"] < 0.85:
            s += 3
        if m["rango10"] < 8:
            s += 3
    return round(t, 1), round(r, 1), round(k, 1), round(s, 1)
