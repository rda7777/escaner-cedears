"""Arma ratios.csv desde la lista oficial de Comafi (resuelve ticker de Yahoo por ISIN)."""
import sys, re, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
from escaner import S
src, out = sys.argv[1], sys.argv[2]
x = pd.read_excel(src, header=None)
hdr = x.index[x.apply(lambda r: r.astype(str).str.contains('Ratio Cedear').any(), axis=1)][0]
x.columns = [str(c).strip() for c in x.iloc[hdr]]
x = x.iloc[hdr + 1:]
x = x[pd.to_numeric(x.iloc[:, 0], errors='coerce').notna()]
cols = {c: c for c in x.columns}
def col(name): return [c for c in x.columns if name.lower() in c.lower()][0]
d = pd.DataFrame({
    'cedear': x[col('Identificación Mercado')].fillna('').astype(str).str.strip(),
    'nombre': x[col('DENOMINACION')].fillna('').astype(str).str.strip(),
    'isin': x[col('ISIN Subyacente')].fillna('').astype(str).str.strip(),
    'ratio_txt': x[col('Ratio Cedear')].fillna('').astype(str).str.strip(),
    'mercado': x[col('Mercado de Negociación')].fillna('').astype(str).str.strip(),
    'tipo': x[col('Valor Subyacente')].fillna('').astype(str).str.strip(),
    'pais': x[col('País de Origen')].fillna('').astype(str).str.strip(),
})
d = d[~d.tipo.str.upper().eq('ETF')]
def ratio(t):
    m = re.match(r'\s*([\d.,]+)\s*:\s*([\d.,]+)', t)
    if not m: return None
    a, b = float(m.group(1).replace(',', '.')), float(m.group(2).replace(',', '.'))
    return a / b
d['ratio'] = d.ratio_txt.map(ratio)
US = ('NYSE', 'NASDAQ', 'AMEX', 'CBOE', 'NYSE ARCA', 'NYSE AMERICAN', 'OTC')
def yahoo(row):
    try:
        r = S.get('https://query2.finance.yahoo.com/v1/finance/search', params={'q': row.isin, 'quotesCount': 10, 'newsCount': 0}, timeout=15)
        qs = [q for q in r.json().get('quotes', []) if q.get('quoteType') in ('EQUITY',)]
        # preferir listados de EE.UU.
        us = [q for q in qs if '.' not in q['symbol'] or q['symbol'].count('-') ]
        us = [q for q in qs if q.get('exchange') in ('NYQ', 'NMS', 'NGM', 'NCM', 'ASE', 'PCX', 'BTS', 'NAS', 'PNK', 'OQB', 'OQX')]
        if us: return us[0]['symbol']
    except Exception:
        pass
    return row.cedear.replace('.', '-') if row.mercado.upper().startswith(US) else None
with ThreadPoolExecutor(6) as ex:
    d['ticker'] = list(ex.map(yahoo, d.itertuples()))
print(d[d.ticker.isna()][['cedear', 'nombre', 'mercado']].to_string())
d = d.dropna(subset=['ticker', 'ratio'])
# fuera programas inhabilitados (marcados con **) y especies raras
d = d[~d.nombre.str.contains(r'\*\*', regex=True)]
d = d[~d.cedear.str.contains(r'-XD|DJNJ', regex=True)]
# códigos BYMA que difieren del ticker de origen
ALIAS = {'BBVA': 'BBV', 'BG': 'BNG', 'KOF': 'KOFM', 'DIS': 'DISN', 'PKX': 'PKS', 'TX': 'TXR', 'TRV': 'TRVV',
         'XRX': 'XROX', 'WB': 'WBO', 'TS': 'TEN', 'AKO.B': 'AKO-B'}
d['cedear'] = [ALIAS.get(c, c.replace('/', '')) for c in d.cedear]
d[['cedear', 'ticker', 'ratio', 'nombre', 'pais', 'mercado']].to_csv(out, index=False)
print(len(d), 'filas'); print(d[d.cedear != d.ticker][['cedear', 'ticker', 'nombre']].to_string())
