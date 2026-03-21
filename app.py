"""Ronia Trend Predictor — Streamlit App."""
import streamlit as st
import os, time, json, warnings, random
from datetime import date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Ronia Trend Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === SIDEBAR ===
with st.sidebar:
    st.title("⚙️ Configurazione")
    
    # API Key
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", ""),
        help="Ottienila gratis su aistudio.google.com"
    )
    
    st.divider()
    
    nicchia = st.text_input(
        "🔍 Nicchia da analizzare",
        value="significato dei sogni",
        help="Scrivi un argomento. Il sistema lo espande automaticamente."
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox("Periodo", ["12m", "3m", "5y"], index=0)
        geo = st.text_input("Paese", value="IT", max_chars=2)
    with col2:
        mesi_prev = st.slider("Mesi previsione", 1, 6, 3)
        top_n = st.slider("Top keyword Prophet", 3, 20, 10)
    
    pausa = st.slider("Pausa anti-ban (sec)", 3, 15, 6)
    
    st.divider()
    
    usa_prophet = st.checkbox("Usa Prophet ML", value=True, 
                               help="Disattiva se il server è lento")
    
    avvia_sidebar = st.button("🚀 Avvia Analisi", use_container_width=True, type="primary", key="avvia_sidebar")
    if avvia_sidebar:
        st.session_state['avvia'] = True

# === MAIN ===
st.title("📊 Ronia Trend Predictor")
st.caption("Google Trends + AI • Analisi automatica • Previsioni ML • Strategia editoriale")

avvia = st.session_state.get('avvia', False)

if not avvia:
    st.info("Configura la nicchia e clicca **Avvia Analisi**")

    with st.expander("📖 Come funziona", expanded=True):
        st.markdown("""
**Ronia Trend Predictor** analizza i trend di Google per qualsiasi nicchia e genera previsioni + strategia editoriale in automatico.

**Il flusso in 6 step:**

1. **Esplosione della nicchia** — Inserisci un argomento (es. "significato dei sogni"). Gemini AI lo scompone in 10-15 sotto-nicchie realistiche, tipiche delle ricerche Google italiane.

2. **Generazione keyword** — Per ogni sotto-nicchia, Gemini genera 5 keyword brevi (2-3 parole) ottimizzate per Google Trends. Il sistema raccoglie 50-80 keyword candidate.

3. **Scoperta rising queries** — Il sistema interroga Google Trends per ciascuna keyword seed, estraendo le ricerche in forte crescita (rising) e le top queries. Vengono aggiunte al pool da analizzare.

4. **Scaricamento dati storici** — Tutte le keyword vengono inviate a Google Trends in batch (max 5 alla volta, con keyword "ancora" per normalizzare). Il risultato e una serie temporale di interesse nel periodo scelto.

5. **Scoring e classificazione** — Ogni keyword riceve uno score calcolato su: interesse medio, interesse recente, variazione %, slope e accelerazione. Viene classificata come EMERGENTE, IN CRESCITA, STABILE o IN CALO.

6. **Previsioni ML + Strategia AI** — Prophet (Meta) prevede l'andamento dei prossimi mesi. Gemini analizza tutto e genera una strategia editoriale completa con priorita e piano mese per mese.
        """)

    st.divider()
    col_start1, col_start2, col_start3 = st.columns([1, 2, 1])
    with col_start2:
        if st.button("🚀 Avvia Analisi", use_container_width=True, type="primary", key="avvia_main"):
            st.session_state['avvia'] = True
            st.rerun()
    st.stop()

st.session_state['avvia'] = False  # reset per il prossimo ciclo

if not gemini_key:
    st.error("Inserisci la Gemini API Key nel pannello laterale.")
    st.stop()

# === PIPELINE ===
import google.generativeai as genai
from pytrends.request import TrendReq

oggi = date.today().strftime('%Y%m%d')
tf_map = {'3m': 'today 3-m', '12m': 'today 12-m', '5y': 'today 5-y'}
tf = tf_map.get(timeframe, 'today 12-m')

genai.configure(api_key=gemini_key)
model_json = genai.GenerativeModel('gemini-3.1-flash-lite-preview',
    generation_config={'response_mime_type': 'application/json'})

progress = st.progress(0, text="Inizializzazione...")

# --- PASSO 1: Esplosione nicchia ---
progress.progress(5, text=f'🔍 Step 1/6 — Esplosione nicchia "{nicchia}" con Gemini AI...')

prompt1 = (
    f'Sei un esperto SEO italiano. Nicchia: "{nicchia}".\n'
    f'Scomponi in 10-15 sotto-nicchie (ricerche Google Italia reali).\n'
    f'Ogni sotto-nicchia: 2-3 parole.\n'
    f'JSON: {{"sotto_nicchie": ["n1", "n2", ...]}}'
)
try:
    r1 = model_json.generate_content(prompt1)
    sotto_nicchie = json.loads(r1.text).get('sotto_nicchie', [nicchia])
except Exception as e:
    st.warning(f"Errore Gemini: {e}")
    sotto_nicchie = [nicchia]

with st.expander(f"📂 {len(sotto_nicchie)} sotto-nicchie trovate", expanded=False):
    for i, sn in enumerate(sotto_nicchie, 1):
        st.write(f"{i}. {sn}")

# --- PASSO 2: Keyword ---
progress.progress(15, text=f'🔑 Step 2/6 — Generazione keyword per {len(sotto_nicchie)} sotto-nicchie...')

all_seed_kws = [sn for sn in sotto_nicchie if len(sn.split()) <= 3]

for sn in sotto_nicchie:
    prompt2 = (
        f'Genera 5 keyword per GOOGLE TRENDS (NON per SEO).\n'
        f'Sotto-nicchia: "{sn}" (macro: "{nicchia}")\n'
        f'MASSIMO 2-3 parole, ricerche POPOLARI.\n'
        f'JSON: {{"keywords": ["kw1", "kw2", ...]}}'
    )
    try:
        r2 = model_json.generate_content(prompt2)
        kws = json.loads(r2.text).get('keywords', [])
        for kw in kws:
            if kw not in all_seed_kws and 1 < len(kw.split()) <= 3:
                all_seed_kws.append(kw)
    except Exception:
        pass

st.success(f"🔑 {len(all_seed_kws)} keyword generate")

# --- PYTRENDS ---
progress.progress(25, text='📈 Connessione Google Trends...')

pytrends = TrendReq(hl='it-IT', tz=-60, timeout=(10, 25),
    requests_args={
        'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
        'verify': True
    })

def fetch_trends(keywords):
    for attempt in range(3):
        try:
            pytrends.build_payload(keywords, timeframe=tf, geo=geo)
            time.sleep(pausa + random.uniform(1, 4))
            return True
        except Exception:
            time.sleep(pausa * (attempt + 2))
    return False

# --- DISCOVERY ---
progress.progress(30, text=f'🔎 Step 3/6 — Scoperta rising queries da Google Trends ({len(all_seed_kws)} keyword seed)...')
rising_queries = []
query_viste = set()

for kw in all_seed_kws[:12]:
    if not fetch_trends([kw]):
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

extra = [rq['query'] for rq in rising_queries 
         if rq['tipo'] == 'rising' and rq['query'] not in all_seed_kws 
         and len(rq['query'].split()) <= 3][:15]
tutte_le_kw = list(dict.fromkeys(all_seed_kws + extra))
tutte_le_kw = [kw for kw in tutte_le_kw if len(kw.split()) <= 3]

# --- SERIE TEMPORALI ---
anchor = tutte_le_kw[0]
altre = tutte_le_kw[1:]
dfs = []

progress.progress(40, text=f'📊 Step 4/6 — Scaricamento serie temporali per {len(tutte_le_kw)} keyword...')

if fetch_trends([anchor]):
    try:
        df_a = pytrends.interest_over_time()
        if df_a is not None and not df_a.empty:
            if 'isPartial' in df_a.columns:
                df_a = df_a.drop(columns=['isPartial'])
            dfs.append(df_a)
    except Exception:
        pass

total_batches = (len(altre) + 3) // 4
for i in range(0, len(altre), 4):
    batch = [anchor] + altre[i:i+4]
    batch_num = i // 4 + 1
    pct = 40 + int(30 * batch_num / max(total_batches, 1))
    progress.progress(min(pct, 70), text=f'📊 Step 4/6 — Scaricamento batch {batch_num}/{total_batches} da Google Trends...')
    if not fetch_trends(batch):
        continue
    try:
        df = pytrends.interest_over_time()
        if df is not None and not df.empty:
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            dfs.append(df)
    except Exception:
        pass

# Normalizzazione
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
        df_base = pd.merge(df_base, df[cols] * scale, left_index=True, right_index=True, how='outer')
    df_trend = df_base.fillna(0)

if df_trend.empty:
    st.error("Nessun dato ricevuto da Google Trends.")
    st.stop()

# --- MOMENTUM + SCORE ---
progress.progress(75, text=f'📐 Step 5/6 — Calcolo score, momentum e classificazione per {len(df_trend.columns)} keyword...')

def momentum(serie):
    s = serie.astype(float)
    if len(s) < 4: return None
    n = len(s)
    recente = s.tail(max(1, n // 4)).mean()
    precedente = s.iloc[:max(1, n * 3 // 4)].mean()
    var = ((recente - precedente) / (precedente + 0.1)) * 100
    x = np.arange(n)
    sl = np.polyfit(x, s.values, 1)[0] if n > 1 else 0
    sn_val = sl / (s.mean() + 0.1) * 100
    ac = 0
    if n >= 8:
        m1, m2 = s.iloc[:n//2].mean(), s.iloc[n//2:].mean()
        ac = ((m2 - m1) / (m1 + 0.1)) * 100
    picco = s.values.argmax() >= n * 0.75
    return {'vr': round(recente,1), 'var': round(var,1), 'sl': round(sn_val,2),
            'ac': round(ac,1), 'picco': picco, 'media': round(s.mean(),1)}

rows = []
for kw in df_trend.columns:
    mom = momentum(df_trend[kw])
    if not mom: continue
    s1 = min(100, max(0, 50 + mom['var'] * 0.3 + mom['sl'] * 2))
    s2 = min(100, max(0, mom['ac'] * 0.5 + 50))
    s3 = min(100, mom['media'] * 1.5)
    s4 = 80 if mom['picco'] else 40
    score = s1 * 0.35 + s2 * 0.20 + s3 * 0.25 + s4 * 0.20
    if score > 70 and mom['var'] > 30: cl = 'EMERGENTE'
    elif score > 55 or mom['var'] > 15: cl = 'IN CRESCITA'
    elif score < 35 or mom['var'] < -15: cl = 'IN CALO'
    else: cl = 'STABILE'
    rows.append({'keyword': kw, 'score': round(score,1), 'classif': cl,
        'interesse_medio': mom['media'], 'interesse_recente': mom['vr'],
        'var_pct': mom['var'], 'slope': mom['sl'], 'accel': mom['ac'],
        'picco_recente': mom['picco']})

df_score = pd.DataFrame(rows).sort_values('score', ascending=False) if rows else pd.DataFrame()

# --- PROPHET ---
risultati_prophet = {}

if usa_prophet and not df_score.empty:
    progress.progress(80, text=f'🤖 Step 5/6 — Previsioni ML con Prophet per le top {top_n} keyword...')
    
    try:
        from prophet import Prophet
        has_prophet = True
    except ImportError:
        has_prophet = False
        st.warning("Prophet non disponibile. Uso modello lineare.")
    
    freq, periods = ('D', mesi_prev * 30) if timeframe == '3m' else ('W', mesi_prev * 4)
    
    for kw in df_score.head(top_n)['keyword'].tolist():
        dp = pd.DataFrame({'ds': df_trend.index, 'y': df_trend[kw].values.astype(float)})
        if len(dp) < 6: continue
        try:
            if dp['y'].mean() < 2 or not has_prophet:
                # Lineare
                x = np.arange(len(dp))
                coeffs = np.polyfit(x, dp['y'].values, 1)
                fx = np.arange(len(dp), len(dp) + periods)
                fd = pd.date_range(dp['ds'].max() + pd.Timedelta(days=7 if freq=='W' else 1), periods=periods, freq=freq)
                yh = np.maximum(0, coeffs[0] * fx + coeffs[1])
                std = dp['y'].std()
                mr = dp['y'].tail(max(1, len(dp)//6)).mean()
                mp = float(np.mean(yh))
                risultati_prophet[kw] = {
                    'fc': pd.DataFrame({'ds': list(dp['ds'])+list(fd), 'yhat': list(dp['y'])+list(yh),
                        'yhat_lower': list(dp['y'])+list(np.maximum(0,yh-std)), 'yhat_upper': list(dp['y'])+list(yh+std)}),
                    'st': dp[['ds','y']], 'mr': round(mr,1), 'mp': round(mp,1),
                    'd': round(((mp-mr)/(mr+0.1))*100,1), 'modello': 'lineare'}
            else:
                # Prophet
                dp['floor'] = 0; cap = max(dp['y'].max()*3, 100); dp['cap'] = cap
                m = Prophet(growth='logistic', weekly_seasonality=(freq=='D'),
                    daily_seasonality=False, yearly_seasonality=len(dp)>=52,
                    changepoint_prior_scale=0.1, interval_width=0.80)
                m.fit(dp)
                fu = m.make_future_dataframe(periods=periods, freq=freq)
                fu['floor'] = 0; fu['cap'] = cap
                pr = m.predict(fu)
                for c in ['yhat','yhat_lower','yhat_upper']: pr[c] = pr[c].clip(lower=0)
                mr = dp['y'].tail(max(1, len(dp)//6)).mean()
                mp = float(pr['yhat'].tail(periods).mean())
                risultati_prophet[kw] = {
                    'fc': pr[['ds','yhat','yhat_lower','yhat_upper']], 'st': dp[['ds','y']],
                    'mr': round(mr,1), 'mp': round(mp,1),
                    'd': round(((mp-mr)/(mr+0.1))*100,1), 'modello': 'prophet'}
        except Exception:
            pass

# --- STRATEGIA ---
progress.progress(90, text='🎯 Step 6/6 — Generazione strategia editoriale con Gemini AI...')
strategia = None

if not df_score.empty:
    lines = []
    for _, r in df_score.head(20).iterrows():
        p = risultati_prophet.get(r['keyword'], {})
        l = f"- {r['keyword']}: {r['classif']}, score {r['score']:.0f}, var {r['var_pct']:+.1f}%"
        if p: l += f", prev {p['mp']:.0f} ({p['d']:+.1f}%)"
        lines.append(l)
    
    prompt_s = (
        f'Analista SEO Italia. Nicchia: "{nicchia}" | {mesi_prev}m\n'
        f'Dati:\n{chr(10).join(lines)}\n\n'
        f'JSON: {{"executive_summary":"3-4 frasi",'
        f'"opportunita":[{{"keyword":"","azione":"","titolo":"","priorita":"ALTA/MEDIA"}}],'
        f'"piano_editoriale":[{{"mese":"Mese 1","focus":"","contenuti":["t1","t2","t3"]}}]}}'
    )
    try:
        resp = model_json.generate_content(prompt_s)
        strategia = json.loads(resp.text)
    except Exception:
        pass

progress.progress(100, text='✅ Analisi completata! Esplora i risultati nelle tab qui sotto.')
time.sleep(0.5)
progress.empty()

# ================================================================
# VISUALIZZAZIONE RISULTATI
# ================================================================

# KPI
n_em = len(df_score[df_score['classif']=='EMERGENTE']) if not df_score.empty else 0
n_cr = len(df_score[df_score['classif']=='IN CRESCITA']) if not df_score.empty else 0
n_ca = len(df_score[df_score['classif']=='IN CALO']) if not df_score.empty else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Keyword", len(df_score))
col2.metric("Emergenti", n_em)
col3.metric("In crescita", n_cr)
col4.metric("In calo", n_ca)
col5.metric("Previsioni", len(risultati_prophet))

st.divider()

# --- TAB LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Classifica", "📈 Previsioni", "🔥 Rising", "🎯 Strategia", "📥 Export"
])

# TAB 1: CLASSIFICA
with tab1:
    if not df_score.empty:
        # Filtro
        filtro = st.selectbox("Filtra per stato", ["Tutti", "EMERGENTE", "IN CRESCITA", "STABILE", "IN CALO"])
        df_vis = df_score if filtro == "Tutti" else df_score[df_score['classif'] == filtro]
        
        # Tabella
        st.dataframe(
            df_vis[['keyword', 'score', 'classif', 'interesse_medio', 'interesse_recente', 'var_pct', 'slope', 'accel']].style
                .format({'score': '{:.0f}', 'var_pct': '{:+.1f}%', 'interesse_medio': '{:.0f}', 'interesse_recente': '{:.0f}'})
                .background_gradient(subset=['score'], cmap='YlOrRd')
                .background_gradient(subset=['var_pct'], cmap='RdYlGn'),
            use_container_width=True, height=500
        )
        
        # Bar chart
        fig_bar = px.bar(df_score.head(20), x='score', y='keyword', orientation='h',
            color='var_pct', color_continuous_scale='RdYlGn', title='Top 20 per Score')
        fig_bar.update_layout(template='plotly_dark', height=550,
            yaxis={'categoryorder': 'total ascending'}, margin=dict(l=200))
        st.plotly_chart(fig_bar, use_container_width=True)

# TAB 2: PREVISIONI
with tab2:
    if risultati_prophet:
        # Confronto multi-keyword
        fig_all = go.Figure()
        for kw in list(risultati_prophet.keys())[:10]:
            d = risultati_prophet[kw]
            st_data = d['st']; fc = d['fc']; cut = st_data['ds'].max()
            fut = fc[fc['ds'] > cut] if 'ds' in fc.columns else pd.DataFrame()
            fig_all.add_trace(go.Scatter(
                x=list(st_data['ds']) + (list(fut['ds']) if len(fut) > 0 else []),
                y=list(st_data['y']) + (list(fut['yhat']) if len(fut) > 0 else []),
                mode='lines', name=kw))
        oggi_str = df_trend.index.max().strftime('%Y-%m-%d')
        fig_all.add_shape(type="line", x0=oggi_str, x1=oggi_str, y0=0, y1=1,
                          yref="paper", line=dict(dash="dash", color="white"))
        fig_all.add_annotation(x=oggi_str, y=1, yref="paper", text="Oggi",
                               showarrow=False, yshift=10)
        fig_all.update_layout(title='Confronto Trend + Previsioni', template='plotly_dark',
            hovermode='x unified', height=500)
        st.plotly_chart(fig_all, use_container_width=True)
        
        # Singoli
        kw_sel = st.selectbox("Dettaglio keyword", list(risultati_prophet.keys()))
        if kw_sel:
            d = risultati_prophet[kw_sel]
            st_data = d['st']; fc = d['fc']; cut = st_data['ds'].max()
            fut = fc[fc['ds'] > cut] if 'ds' in fc.columns else pd.DataFrame()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st_data['ds'], y=st_data['y'],
                mode='lines+markers', name='Reale', line=dict(color='#00d1ff', width=2)))
            if len(fut) > 0:
                fig.add_trace(go.Scatter(x=fut['ds'], y=fut['yhat'],
                    mode='lines+markers', name='Previsione',
                    line=dict(color='#ff6700', width=3, dash='dash')))
                fig.add_trace(go.Scatter(
                    x=list(fut['ds']) + list(fut['ds'])[::-1],
                    y=list(fut['yhat_upper']) + list(fut['yhat_lower'])[::-1],
                    fill='toself', fillcolor='rgba(255,103,0,.15)',
                    line=dict(color='rgba(0,0,0,0)'), name='IC 80%'))
            
            sc = df_score[df_score['keyword'] == kw_sel]
            cl = sc['classif'].values[0] if len(sc) > 0 else ''
            fig.update_layout(
                title=f"{kw_sel.upper()} | {cl} [{d['modello']}] | Prev: {d['mp']:.0f} ({d['d']:+.1f}%)",
                template='plotly_dark', hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Media recente", f"{d['mr']:.0f}")
            mc2.metric("Previsione", f"{d['mp']:.0f}", f"{d['d']:+.1f}%")
            mc3.metric("Modello", d['modello'])
    else:
        st.info("Nessuna previsione disponibile.")

# TAB 3: RISING
with tab3:
    if rising_queries:
        df_rq = pd.DataFrame(rising_queries).sort_values('value', ascending=False)
        
        col_r, col_t = st.columns(2)
        with col_r:
            st.subheader("Rising Queries")
            rising_only = df_rq[df_rq['tipo'] == 'rising'].head(20)
            if not rising_only.empty:
                st.dataframe(rising_only[['query', 'value', 'origine']], use_container_width=True, height=400)
        with col_t:
            st.subheader("Top Queries")
            top_only = df_rq[df_rq['tipo'] == 'top'].head(20)
            if not top_only.empty:
                st.dataframe(top_only[['query', 'value', 'origine']], use_container_width=True, height=400)
    else:
        st.info("Nessuna rising query trovata.")

# TAB 4: STRATEGIA
with tab4:
    if strategia:
        if strategia.get('executive_summary'):
            st.subheader("Executive Summary")
            st.info(strategia['executive_summary'])
        
        if strategia.get('opportunita'):
            st.subheader("Opportunita")
            for o in strategia['opportunita']:
                pri = o.get('priorita', '')
                icon = '🔴' if 'ALTA' in pri else '🟡'
                st.markdown(f"**{icon} [{pri}] {o.get('keyword', '')}**")
                st.write(f"→ {o.get('azione', '')}")
                st.write(f"📝 _{o.get('titolo', '')}_")
                st.divider()
        
        if strategia.get('piano_editoriale'):
            st.subheader("Piano Editoriale")
            for m in strategia['piano_editoriale']:
                with st.expander(f"📅 {m.get('mese', '')} — {m.get('focus', '')}"):
                    for c in m.get('contenuti', []):
                        st.write(f"• {c}")
    else:
        st.info("Strategia non disponibile.")

# TAB 5: EXPORT
with tab5:
    st.subheader("Scarica i dati")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        if not df_score.empty:
            csv = df_score.to_csv(index=False)
            st.download_button("📥 Classifica (CSV)", csv,
                file_name=f"classifica_{nicchia.replace(' ','_')}_{oggi}.csv",
                mime="text/csv")
    
    with col_e2:
        if not df_trend.empty:
            csv_t = df_trend.to_csv()
            st.download_button("📥 Serie temporali (CSV)", csv_t,
                file_name=f"serie_{nicchia.replace(' ','_')}_{oggi}.csv",
                mime="text/csv")
    
    if strategia:
        json_s = json.dumps(strategia, ensure_ascii=False, indent=2)
        st.download_button("📥 Strategia (JSON)", json_s,
            file_name=f"strategia_{nicchia.replace(' ','_')}_{oggi}.json",
            mime="application/json")
    
    # Report completo JSON
    report = {
        'nicchia': nicchia, 'data': oggi, 'timeframe': timeframe,
        'n_keywords': len(df_score), 'n_rising': len(rising_queries),
        'classifica': df_score.to_dict('records') if not df_score.empty else [],
        'strategia': strategia
    }
    st.download_button("📥 Report completo (JSON)",
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        file_name=f"report_{nicchia.replace(' ','_')}_{oggi}.json",
        mime="application/json")

    # Report HTML
    st.divider()
    st.subheader("Report HTML interattivo")

    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Trend Report — {nicchia} — {oggi}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:2rem auto;max-width:1200px;padding:0 1rem}}
h1,h2,h3{{color:#00d1ff}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #333;padding:8px 12px;text-align:left}}
th{{background:#222;color:#00d1ff}}
tr:nth-child(even){{background:#1a1a1a}}
.badge{{padding:2px 8px;border-radius:4px;font-size:.85em;font-weight:600}}
.EMERGENTE{{background:#ff4444;color:#fff}}.IN_CRESCITA{{background:#ff8800;color:#fff}}
.STABILE{{background:#44aa44;color:#fff}}.IN_CALO{{background:#666;color:#ccc}}
.section{{margin:2rem 0;padding:1rem;background:#1a1a1a;border-radius:8px}}
</style></head><body>
<h1>📊 Trend Report — {nicchia.upper()}</h1>
<p>Data: {oggi} | Timeframe: {timeframe} | Keywords: {len(df_score)}</p>
"""]

    # Classifica table
    if not df_score.empty:
        html_parts.append('<div class="section"><h2>Classifica Keywords</h2><table>')
        html_parts.append('<tr><th>#</th><th>Keyword</th><th>Score</th><th>Stato</th><th>Interesse medio</th><th>Interesse recente</th><th>Var %</th></tr>')
        for i, row in df_score.head(30).iterrows():
            cls = row.get('classif', '').replace(' ', '_')
            html_parts.append(
                f'<tr><td>{i+1}</td><td><b>{row["keyword"]}</b></td>'
                f'<td>{row["score"]:.0f}</td>'
                f'<td><span class="badge {cls}">{row.get("classif","")}</span></td>'
                f'<td>{row.get("interesse_medio",0):.0f}</td>'
                f'<td>{row.get("interesse_recente",0):.0f}</td>'
                f'<td>{row.get("var_pct",0):+.1f}%</td></tr>')
        html_parts.append('</table></div>')

        # Bar chart
        fig_html_bar = px.bar(df_score.head(20), x='score', y='keyword', orientation='h',
            color='var_pct', color_continuous_scale='RdYlGn', title='Top 20 per Score')
        fig_html_bar.update_layout(template='plotly_dark', height=550,
            yaxis={'categoryorder': 'total ascending'}, margin=dict(l=200))
        html_parts.append(f'<div class="section"><h2>Top 20 Score</h2>{fig_html_bar.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # Previsioni chart
    if risultati_prophet:
        fig_html_all = go.Figure()
        for kw in list(risultati_prophet.keys())[:10]:
            d = risultati_prophet[kw]
            st_data = d['st']; fc = d['fc']; cut = st_data['ds'].max()
            fut = fc[fc['ds'] > cut] if 'ds' in fc.columns else pd.DataFrame()
            fig_html_all.add_trace(go.Scatter(
                x=list(st_data['ds']) + (list(fut['ds']) if len(fut) > 0 else []),
                y=list(st_data['y']) + (list(fut['yhat']) if len(fut) > 0 else []),
                mode='lines', name=kw))
        if not df_trend.empty:
            oggi_s = df_trend.index.max().strftime('%Y-%m-%d')
            fig_html_all.add_shape(type="line", x0=oggi_s, x1=oggi_s, y0=0, y1=1,
                                   yref="paper", line=dict(dash="dash", color="white"))
            fig_html_all.add_annotation(x=oggi_s, y=1, yref="paper", text="Oggi",
                                        showarrow=False, yshift=10)
        fig_html_all.update_layout(title='Confronto Trend + Previsioni', template='plotly_dark',
            hovermode='x unified', height=500)
        html_parts.append(f'<div class="section"><h2>Previsioni</h2>{fig_html_all.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # Rising queries
    if rising_queries:
        df_rq_html = pd.DataFrame(rising_queries).sort_values('value', ascending=False)
        html_parts.append('<div class="section"><h2>Rising Queries</h2><table>')
        html_parts.append('<tr><th>Query</th><th>Valore</th><th>Tipo</th><th>Origine</th></tr>')
        for _, row in df_rq_html.head(30).iterrows():
            html_parts.append(f'<tr><td>{row["query"]}</td><td>{row["value"]}</td><td>{row["tipo"]}</td><td>{row["origine"]}</td></tr>')
        html_parts.append('</table></div>')

    # Strategia
    if strategia:
        html_parts.append('<div class="section"><h2>Strategia AI</h2>')
        if strategia.get('executive_summary'):
            html_parts.append(f'<p><em>{strategia["executive_summary"]}</em></p>')
        if strategia.get('opportunita'):
            html_parts.append('<h3>Opportunità</h3><ul>')
            for o in strategia['opportunita']:
                html_parts.append(f'<li><b>[{o.get("priorita","")}] {o.get("keyword","")}</b> — {o.get("azione","")} <br><em>{o.get("titolo","")}</em></li>')
            html_parts.append('</ul>')
        if strategia.get('piano_editoriale'):
            html_parts.append('<h3>Piano Editoriale</h3>')
            for m in strategia['piano_editoriale']:
                html_parts.append(f'<h4>📅 {m.get("mese","")} — {m.get("focus","")}</h4><ul>')
                for c in m.get('contenuti', []):
                    html_parts.append(f'<li>{c}</li>')
                html_parts.append('</ul>')
        html_parts.append('</div>')

    html_parts.append('<footer style="text-align:center;color:#666;margin-top:3rem;padding:1rem">Generato da Ronia Trend Predictor</footer></body></html>')
    html_report = '\n'.join(html_parts)

    st.download_button("📥 Report completo (HTML)", html_report,
        file_name=f"report_{nicchia.replace(' ','_')}_{oggi}.html",
        mime="text/html")
