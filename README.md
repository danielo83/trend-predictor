# Trend Predictor Pro — Streamlit App

Analisi trend automatica: Google Trends + Gemini AI + Prophet ML.

## Deploy su Streamlit Cloud (gratis, 5 minuti)

### Passo 1: GitHub
1. Vai su [github.com/new](https://github.com/new) e crea un repo (es. `trend-predictor`)
2. Carica tutti i file di questa cartella nel repo

### Passo 2: Streamlit Cloud
1. Vai su [share.streamlit.io](https://share.streamlit.io)
2. Accedi con GitHub
3. Clicca **New app**
4. Seleziona il repo, branch `main`, file `app.py`
5. In **Advanced settings > Secrets**, incolla:
   ```
   GEMINI_API_KEY = "AIzaSy..."
   ```
6. Clicca **Deploy**

### Fatto!
Dopo 2-3 minuti avrai un URL tipo:
```
https://trend-predictor.streamlit.app
```

Condividi il link con chiunque. Chiunque puo usarlo.

## Uso locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Come funziona

1. L'utente scrive una nicchia (es. "significato dei sogni")
2. Gemini AI la espande in sotto-nicchie e keyword
3. PyTrends scarica dati reali da Google Trends
4. Il sistema calcola score, momentum e classificazione
5. Prophet ML prevede i prossimi mesi
6. Gemini genera una strategia editoriale
7. L'utente vede grafici interattivi e puo scaricare tutto in CSV/JSON
