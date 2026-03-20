"""Core analysis logic — shared by Streamlit app and Telegram bot."""
import time, json, random, warnings
from datetime import date
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def explode_niche(model_json, nicchia):
    """Step 1: Expand niche into sub-niches via Gemini."""
    prompt = (
        f'Sei un esperto SEO italiano. Nicchia: "{nicchia}".\n'
        f'Scomponi in 10-15 sotto-nicchie (ricerche Google Italia reali).\n'
        f'Ogni sotto-nicchia: 2-3 parole.\n'
        f'JSON: {{"sotto_nicchie": ["n1", "n2", ...]}}'
    )
    try:
        r = model_json.generate_content(prompt)
        return json.loads(r.text).get('sotto_nicchie', [nicchia])
    except Exception:
        return [nicchia]


def generate_keywords(model_json, sotto_nicchie, nicchia):
    """Step 2: Generate keywords for each sub-niche."""
    all_seed_kws = [sn for sn in sotto_nicchie if len(sn.split()) <= 3]
    for sn in sotto_nicchie:
        prompt = (
            f'Genera 5 keyword per GOOGLE TRENDS (NON per SEO).\n'
            f'Sotto-nicchia: "{sn}" (macro: "{nicchia}")\n'
            f'MASSIMO 2-3 parole, ricerche POPOLARI.\n'
            f'JSON: {{"keywords": ["kw1", "kw2", ...]}}'
        )
        try:
            r = model_json.generate_content(prompt)
            kws = json.loads(r.text).get('keywords', [])
            for kw in kws:
                if kw not in all_seed_kws and 1 < len(kw.split()) <= 3:
                    all_seed_kws.append(kw)
        except Exception:
            pass
    return all_seed_kws


def fetch_trends(pytrends, keywords, timeframe, geo, pausa):
    """Build payload with retries."""
    for attempt in range(3):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            time.sleep(pausa + random.uniform(1, 4))
            return True
        except Exception:
            time.sleep(pausa * (attempt + 2))
    return False


def discover_rising(pytrends, all_seed_kws, timeframe, geo, pausa):
    """Step 3: Discover rising queries."""
    rising_queries = []
    query_viste = set()
    for kw in all_seed_kws[:12]:
        if not fetch_trends(pytrends, [kw], timeframe, geo, pausa):
            continue
        try:
            rq = pytrends.related_queries()
            if kw in rq:
                for tipo in ('rising', 'top'):
                    df_rel = rq[kw].get(tipo)
                    if df_rel is not None and not df_rel.empty:
                        for _, row in df_rel.iterrows():
                            q = str(row.get('query', '')).strip()
                            if q and q not in query_viste and len(q.split()) <= 4:
                                query_viste.add(q)
                                rising_queries.append({
                                    'query': q, 'tipo': tipo,
                                    'value': row.get('value', 0), 'origine': kw
                                })
        except Exception:
            pass
    return rising_queries


def fetch_time_series(pytrends, tutte_le_kw, timeframe, geo, pausa):
    """Step 4: Download and normalize time series."""
    anchor = tutte_le_kw[0]
    altre = tutte_le_kw[1:]
    dfs = []

    if fetch_trends(pytrends, [anchor], timeframe, geo, pausa):
        try:
            df_a = pytrends.interest_over_time()
            if df_a is not None and not df_a.empty:
                if 'isPartial' in df_a.columns:
                    df_a = df_a.drop(columns=['isPartial'])
                dfs.append(df_a)
        except Exception:
            pass

    for i in range(0, len(altre), 4):
        batch = [anchor] + altre[i:i + 4]
        if not fetch_trends(pytrends, batch, timeframe, geo, pausa):
            continue
        try:
            df = pytrends.interest_over_time()
            if df is not None and not df.empty:
                if 'isPartial' in df.columns:
                    df = df.drop(columns=['isPartial'])
                dfs.append(df)
        except Exception:
            pass

    df_trend = pd.DataFrame()
    if dfs:
        df_base = dfs[0]
        for df in dfs[1:]:
            if anchor not in df.columns or anchor not in df_base.columns:
                continue
            med_b = df_base[anchor].median()
            med_d = df[anchor].median()
            scale = med_b / med_d if med_b > 0 and med_d > 0 else 1.0
            cols = [c for c in df.columns if c != anchor]
            df_base = pd.merge(df_base, df[cols] * scale,
                               left_index=True, right_index=True, how='outer')
        df_trend = df_base.fillna(0)
    return df_trend


def momentum(serie):
    s = serie.astype(float)
    if len(s) < 4:
        return None
    n = len(s)
    recente = s.tail(max(1, n // 4)).mean()
    precedente = s.iloc[:max(1, n * 3 // 4)].mean()
    var = ((recente - precedente) / (precedente + 0.1)) * 100
    x = np.arange(n)
    sl = np.polyfit(x, s.values, 1)[0] if n > 1 else 0
    sn_val = sl / (s.mean() + 0.1) * 100
    ac = 0
    if n >= 8:
        m1, m2 = s.iloc[:n // 2].mean(), s.iloc[n // 2:].mean()
        ac = ((m2 - m1) / (m1 + 0.1)) * 100
    picco = s.values.argmax() >= n * 0.75
    return {'vr': round(recente, 1), 'var': round(var, 1), 'sl': round(sn_val, 2),
            'ac': round(ac, 1), 'picco': picco, 'media': round(s.mean(), 1)}


def compute_scores(df_trend):
    """Step 5: Compute momentum scores."""
    rows = []
    for kw in df_trend.columns:
        mom = momentum(df_trend[kw])
        if not mom:
            continue
        s1 = min(100, max(0, 50 + mom['var'] * 0.3 + mom['sl'] * 2))
        s2 = min(100, max(0, mom['ac'] * 0.5 + 50))
        s3 = min(100, mom['media'] * 1.5)
        s4 = 80 if mom['picco'] else 40
        score = s1 * 0.35 + s2 * 0.20 + s3 * 0.25 + s4 * 0.20
        if score > 70 and mom['var'] > 30:
            cl = 'EMERGENTE'
        elif score > 55 or mom['var'] > 15:
            cl = 'IN CRESCITA'
        elif score < 35 or mom['var'] < -15:
            cl = 'IN CALO'
        else:
            cl = 'STABILE'
        rows.append({'keyword': kw, 'score': round(score, 1), 'classif': cl,
                     'interesse_medio': mom['media'], 'interesse_recente': mom['vr'],
                     'var_pct': mom['var'], 'slope': mom['sl'], 'accel': mom['ac'],
                     'picco_recente': mom['picco']})
    return pd.DataFrame(rows).sort_values('score', ascending=False) if rows else pd.DataFrame()


def run_prophet(df_trend, df_score, timeframe, mesi_prev, top_n):
    """Step 6: Prophet / linear forecasts."""
    risultati = {}
    if df_score.empty:
        return risultati

    try:
        from prophet import Prophet
        has_prophet = True
    except ImportError:
        has_prophet = False

    freq, periods = ('D', mesi_prev * 30) if timeframe == '3m' else ('W', mesi_prev * 4)

    for kw in df_score.head(top_n)['keyword'].tolist():
        dp = pd.DataFrame({'ds': df_trend.index, 'y': df_trend[kw].values.astype(float)})
        if len(dp) < 6:
            continue
        try:
            if dp['y'].mean() < 2 or not has_prophet:
                x = np.arange(len(dp))
                coeffs = np.polyfit(x, dp['y'].values, 1)
                fx = np.arange(len(dp), len(dp) + periods)
                fd = pd.date_range(dp['ds'].max() + pd.Timedelta(days=7 if freq == 'W' else 1),
                                   periods=periods, freq=freq)
                yh = np.maximum(0, coeffs[0] * fx + coeffs[1])
                std = dp['y'].std()
                mr = dp['y'].tail(max(1, len(dp) // 6)).mean()
                mp = float(np.mean(yh))
                risultati[kw] = {
                    'mr': round(mr, 1), 'mp': round(mp, 1),
                    'd': round(((mp - mr) / (mr + 0.1)) * 100, 1), 'modello': 'lineare'}
            else:
                dp['floor'] = 0
                cap = max(dp['y'].max() * 3, 100)
                dp['cap'] = cap
                m = Prophet(growth='logistic', weekly_seasonality=(freq == 'D'),
                            daily_seasonality=False, yearly_seasonality=len(dp) >= 52,
                            changepoint_prior_scale=0.1, interval_width=0.80)
                m.fit(dp)
                fu = m.make_future_dataframe(periods=periods, freq=freq)
                fu['floor'] = 0
                fu['cap'] = cap
                pr = m.predict(fu)
                for c in ['yhat', 'yhat_lower', 'yhat_upper']:
                    pr[c] = pr[c].clip(lower=0)
                mr = dp['y'].tail(max(1, len(dp) // 6)).mean()
                mp = float(pr['yhat'].tail(periods).mean())
                risultati[kw] = {
                    'mr': round(mr, 1), 'mp': round(mp, 1),
                    'd': round(((mp - mr) / (mr + 0.1)) * 100, 1), 'modello': 'prophet'}
        except Exception:
            pass
    return risultati


def generate_strategy(model_json, df_score, risultati_prophet, nicchia, mesi_prev):
    """Step 7: AI strategy."""
    if df_score.empty:
        return None
    lines = []
    for _, r in df_score.head(20).iterrows():
        p = risultati_prophet.get(r['keyword'], {})
        l = f"- {r['keyword']}: {r['classif']}, score {r['score']:.0f}, var {r['var_pct']:+.1f}%"
        if p:
            l += f", prev {p['mp']:.0f} ({p['d']:+.1f}%)"
        lines.append(l)

    prompt = (
        f'Analista SEO Italia. Nicchia: "{nicchia}" | {mesi_prev}m\n'
        f'Dati:\n{chr(10).join(lines)}\n\n'
        f'JSON: {{"executive_summary":"3-4 frasi",'
        f'"opportunita":[{{"keyword":"","azione":"","titolo":"","priorita":"ALTA/MEDIA"}}],'
        f'"piano_editoriale":[{{"mese":"Mese 1","focus":"","contenuti":["t1","t2","t3"]}}]}}'
    )
    try:
        resp = model_json.generate_content(prompt)
        return json.loads(resp.text)
    except Exception:
        return None


def run_full_analysis(gemini_key, nicchia, timeframe='12m', geo='IT',
                      mesi_prev=3, top_n=10, pausa=6, usa_prophet=True,
                      on_progress=None):
    """Run the complete pipeline. Returns a dict with all results.

    on_progress(pct, message) is an optional callback for status updates.
    """
    import google.generativeai as genai
    from pytrends.request import TrendReq

    def progress(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    oggi = date.today().strftime('%Y%m%d')
    tf_map = {'3m': 'today 3-m', '12m': 'today 12-m', '5y': 'today 5-y'}
    tf = tf_map.get(timeframe, 'today 12-m')

    genai.configure(api_key=gemini_key)
    model_json = genai.GenerativeModel('gemini-3.1-flash-lite-preview',
                                       generation_config={'response_mime_type': 'application/json'})

    # 1 — Explode niche
    progress(5, f'Esplosione nicchia "{nicchia}"...')
    sotto_nicchie = explode_niche(model_json, nicchia)

    # 2 — Keywords
    progress(15, 'Generazione keyword...')
    all_seed_kws = generate_keywords(model_json, sotto_nicchie, nicchia)

    # 3 — PyTrends setup
    progress(25, 'Connessione Google Trends...')
    pytrends = TrendReq(hl='it-IT', tz=-60, timeout=(10, 25),
                        requests_args={
                            'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                            'verify': True
                        })

    # 4 — Rising queries
    progress(30, 'Scoperta rising queries...')
    rising_queries = discover_rising(pytrends, all_seed_kws, tf, geo, pausa)

    extra = [rq['query'] for rq in rising_queries
             if rq['tipo'] == 'rising' and rq['query'] not in all_seed_kws
             and len(rq['query'].split()) <= 3][:15]
    tutte_le_kw = list(dict.fromkeys(all_seed_kws + extra))
    tutte_le_kw = [kw for kw in tutte_le_kw if len(kw.split()) <= 3]

    if not tutte_le_kw:
        return {'error': 'Nessuna keyword trovata.'}

    # 5 — Time series
    progress(40, f'Scaricamento dati ({len(tutte_le_kw)} keyword)...')
    df_trend = fetch_time_series(pytrends, tutte_le_kw, tf, geo, pausa)

    if df_trend.empty:
        return {'error': 'Nessun dato ricevuto da Google Trends.'}

    # 6 — Scores
    progress(75, 'Calcolo score...')
    df_score = compute_scores(df_trend)

    # 7 — Prophet
    risultati_prophet = {}
    if usa_prophet and not df_score.empty:
        progress(80, 'Previsioni ML...')
        risultati_prophet = run_prophet(df_trend, df_score, timeframe, mesi_prev, top_n)

    # 8 — Strategy
    progress(90, 'Generazione strategia AI...')
    strategia = generate_strategy(model_json, df_score, risultati_prophet, nicchia, mesi_prev)

    progress(100, 'Completato!')

    return {
        'nicchia': nicchia, 'data': oggi, 'timeframe': timeframe,
        'sotto_nicchie': sotto_nicchie,
        'n_keywords': len(df_score),
        'df_score': df_score,
        'rising_queries': rising_queries,
        'risultati_prophet': risultati_prophet,
        'strategia': strategia,
    }
