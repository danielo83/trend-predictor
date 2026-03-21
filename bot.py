"""Telegram Bot — Ronia Trend Predictor."""
import os
import json
import logging
from datetime import date
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters,
)

from analyzer import run_full_analysis

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config from env ---
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# Conversation states
NICHE = 0


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 *Ronia Trend Predictor*\n\n"
        "Comandi:\n"
        "/analizza — avvia analisi completa\n"
        "/quick `<nicchia>` — analisi veloce\n"
        "/help — mostra aiuto\n",
        parse_mode='Markdown')


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Come funziona*\n\n"
        "1. Scrivi /analizza\n"
        "2. Invia la nicchia da analizzare (es. \"significato dei sogni\")\n"
        "3. Attendi l'analisi (2-5 minuti)\n"
        "4. Ricevi classifica, previsioni e strategia\n\n"
        "🔹 /quick `nicchia` — analisi con parametri default\n"
        "🔹 /impostazioni — mostra parametri attuali\n"
        "🔹 /set `parametro valore` — modifica parametro\n\n"
        "Parametri: timeframe (3m/12m/5y), geo (IT/US/...), "
        "mesi (1-6), top_n (3-20), pausa (3-15)",
        parse_mode='Markdown')


async def impostazioni(update: Update, context: ContextTypes.DEFAULT_TYPE):
    p = context.user_data
    await update.message.reply_text(
        "⚙️ *Impostazioni attuali*\n\n"
        f"• Timeframe: `{p.get('timeframe', '12m')}`\n"
        f"• Paese: `{p.get('geo', 'IT')}`\n"
        f"• Mesi previsione: `{p.get('mesi_prev', 3)}`\n"
        f"• Top keyword Prophet: `{p.get('top_n', 10)}`\n"
        f"• Pausa anti-ban: `{p.get('pausa', 6)}` sec\n"
        f"• Prophet ML: `{'sì' if p.get('usa_prophet', True) else 'no'}`\n\n"
        "Usa /set `parametro valore` per modificare.",
        parse_mode='Markdown')


async def set_param(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Uso: /set `parametro valore`\n"
                                        "Es: /set timeframe 5y")
        return

    param = context.args[0].lower()
    value = context.args[1]
    p = context.user_data

    mapping = {
        'timeframe': ('timeframe', str, ('3m', '12m', '5y')),
        'geo': ('geo', str, None),
        'mesi': ('mesi_prev', int, range(1, 7)),
        'top_n': ('top_n', int, range(3, 21)),
        'pausa': ('pausa', int, range(3, 16)),
        'prophet': ('usa_prophet', lambda v: v.lower() in ('si', 'sì', 'true', '1', 'yes'), None),
    }

    if param not in mapping:
        await update.message.reply_text(f"Parametro sconosciuto: {param}\n"
                                        f"Disponibili: {', '.join(mapping.keys())}")
        return

    key, converter, valid = mapping[param]
    try:
        val = converter(value)
        if valid and val not in valid:
            await update.message.reply_text(f"Valore non valido. Valori ammessi: {list(valid) if not isinstance(valid, range) else f'{valid.start}-{valid.stop - 1}'}")
            return
        p[key] = val
        await update.message.reply_text(f"✅ {param} = `{val}`", parse_mode='Markdown')
    except (ValueError, TypeError):
        await update.message.reply_text(f"Valore non valido: {value}")


async def analizza_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Quale nicchia vuoi analizzare?\n"
                                    "(Scrivi il tema, es. \"significato dei sogni\")")
    return NICHE


async def analizza_niche(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nicchia = update.message.text.strip()
    if not nicchia:
        await update.message.reply_text("Scrivi una nicchia valida.")
        return NICHE

    await _run_analysis(update, context, nicchia)
    return ConversationHandler.END


async def quick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso: /quick `nicchia`\nEs: /quick significato dei sogni")
        return
    nicchia = ' '.join(context.args)
    await _run_analysis(update, context, nicchia)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Analisi annullata.")
    return ConversationHandler.END


async def _run_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, nicchia: str):
    """Execute full analysis and send results."""
    if not GEMINI_API_KEY:
        await update.message.reply_text("❌ GEMINI_API_KEY non configurata sul server.")
        return

    p = context.user_data
    status_msg = await update.message.reply_text(f"⏳ Analisi di \"{nicchia}\" in corso...\n"
                                                  "Questo può richiedere 2-5 minuti.")

    async def on_progress(pct, msg):
        try:
            await status_msg.edit_text(f"⏳ [{pct}%] {msg}")
        except Exception:
            pass

    try:
        # Run in executor to not block the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: run_full_analysis(
            gemini_key=GEMINI_API_KEY,
            nicchia=nicchia,
            timeframe=p.get('timeframe', '12m'),
            geo=p.get('geo', 'IT'),
            mesi_prev=p.get('mesi_prev', 3),
            top_n=p.get('top_n', 10),
            pausa=p.get('pausa', 6),
            usa_prophet=p.get('usa_prophet', True),
        ))
    except Exception as e:
        await status_msg.edit_text(f"❌ Errore durante l'analisi: {e}")
        return

    if 'error' in result:
        await status_msg.edit_text(f"❌ {result['error']}")
        return

    # --- Format and send results ---
    df_score = result.get('df_score')
    rising = result.get('rising_queries', [])
    prophet = result.get('risultati_prophet', {})
    strategia = result.get('strategia')
    sotto_nicchie = result.get('sotto_nicchie', [])

    # 1) Summary
    n_em = len(df_score[df_score['classif'] == 'EMERGENTE']) if not df_score.empty else 0
    n_cr = len(df_score[df_score['classif'] == 'IN CRESCITA']) if not df_score.empty else 0
    n_ca = len(df_score[df_score['classif'] == 'IN CALO']) if not df_score.empty else 0

    summary = (
        f"📊 *Analisi completata: \"{nicchia}\"*\n\n"
        f"📂 Sotto-nicchie: {len(sotto_nicchie)}\n"
        f"🔑 Keyword analizzate: {len(df_score)}\n"
        f"🚀 Emergenti: {n_em}\n"
        f"📈 In crescita: {n_cr}\n"
        f"📉 In calo: {n_ca}\n"
        f"🤖 Previsioni ML: {len(prophet)}\n"
    )
    await status_msg.edit_text(summary, parse_mode='Markdown')

    # 2) Top keywords
    if not df_score.empty:
        lines = ["*🏆 Top 15 Keyword per Score:*\n"]
        for i, (_, r) in enumerate(df_score.head(15).iterrows(), 1):
            icon = {'EMERGENTE': '🚀', 'IN CRESCITA': '📈',
                    'STABILE': '➡️', 'IN CALO': '📉'}.get(r['classif'], '•')
            p_info = ''
            if r['keyword'] in prophet:
                pr = prophet[r['keyword']]
                p_info = f" → prev {pr['mp']:.0f} ({pr['d']:+.1f}%)"
            lines.append(
                f"{i}. {icon} *{r['keyword']}*\n"
                f"   Score: {r['score']:.0f} | {r['classif']} | var: {r['var_pct']:+.1f}%{p_info}"
            )
        await update.message.reply_text('\n'.join(lines), parse_mode='Markdown')

    # 3) Rising queries
    if rising:
        rising_sorted = sorted([r for r in rising if r['tipo'] == 'rising'],
                                key=lambda x: x['value'], reverse=True)[:10]
        if rising_sorted:
            lines = ["*🔥 Top Rising Queries:*\n"]
            for r in rising_sorted:
                lines.append(f"• *{r['query']}* (da: {r['origine']})")
            await update.message.reply_text('\n'.join(lines), parse_mode='Markdown')

    # 4) Strategy
    if strategia:
        msg_parts = []
        if strategia.get('executive_summary'):
            msg_parts.append(f"*🎯 Executive Summary*\n{strategia['executive_summary']}\n")

        if strategia.get('opportunita'):
            msg_parts.append("*💡 Opportunità:*")
            for o in strategia['opportunita'][:8]:
                pri = o.get('priorita', '')
                icon = '🔴' if 'ALTA' in pri else '🟡'
                msg_parts.append(
                    f"{icon} *[{pri}] {o.get('keyword', '')}*\n"
                    f"   → {o.get('azione', '')}\n"
                    f"   📝 _{o.get('titolo', '')}_"
                )

        if strategia.get('piano_editoriale'):
            msg_parts.append("\n*📅 Piano Editoriale:*")
            for m in strategia['piano_editoriale']:
                contenuti = '\n'.join(f"   • {c}" for c in m.get('contenuti', []))
                msg_parts.append(f"\n*{m.get('mese', '')}* — {m.get('focus', '')}\n{contenuti}")

        if msg_parts:
            strategy_text = '\n'.join(msg_parts)
            # Telegram has a 4096 char limit per message
            if len(strategy_text) > 4000:
                for i in range(0, len(strategy_text), 4000):
                    await update.message.reply_text(strategy_text[i:i + 4000],
                                                     parse_mode='Markdown')
            else:
                await update.message.reply_text(strategy_text, parse_mode='Markdown')

    # 5) Send JSON report as file
    report = {
        'nicchia': nicchia, 'data': result['data'],
        'timeframe': p.get('timeframe', '12m'),
        'sotto_nicchie': sotto_nicchie,
        'classifica': df_score.to_dict('records') if not df_score.empty else [],
        'rising_queries': rising,
        'previsioni': {k: {'mr': v['mr'], 'mp': v['mp'], 'd': v['d'], 'modello': v['modello']}
                       for k, v in prophet.items()},
        'strategia': strategia,
    }
    oggi = date.today().strftime('%Y%m%d')
    filename = f"report_{nicchia.replace(' ', '_')}_{oggi}.json"
    report_json = json.dumps(report, ensure_ascii=False, indent=2, default=str)

    from io import BytesIO
    buf = BytesIO(report_json.encode('utf-8'))
    buf.name = filename
    await update.message.reply_document(document=buf, filename=filename,
                                         caption="📥 Report completo (JSON)")


def main():
    if not TELEGRAM_TOKEN:
        print("❌ Imposta TELEGRAM_TOKEN come variabile d'ambiente.")
        print("   export TELEGRAM_TOKEN='il-tuo-token'")
        return

    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY non impostata. Impostala prima di usare /analizza.")
        print("   export GEMINI_API_KEY='la-tua-chiave'")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Conversation handler for /analizza
    conv = ConversationHandler(
        entry_points=[CommandHandler('analizza', analizza_start)],
        states={NICHE: [MessageHandler(filters.TEXT & ~filters.COMMAND, analizza_niche)]},
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('impostazioni', impostazioni))
    app.add_handler(CommandHandler('set', set_param))
    app.add_handler(CommandHandler('quick', quick))
    app.add_handler(conv)

    print("🤖 Bot avviato! Premi Ctrl+C per fermare.")
    app.run_polling()


if __name__ == '__main__':
    main()
