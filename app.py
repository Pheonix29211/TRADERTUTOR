import os
from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler
from bot import cmd_start, cmd_scan, cmd_status, cmd_help, cmd_daily_now, cmd_weekly_now
from config import TOKEN, ADMIN_CHAT_ID, PORT

app = Flask(__name__)
bot = Bot(TOKEN)
dp = Dispatcher(bot, None, workers=1, use_context=True)
dp.add_handler(CommandHandler("start", cmd_start))
dp.add_handler(CommandHandler("scan", cmd_scan))
dp.add_handler(CommandHandler("status", cmd_status))
dp.add_handler(CommandHandler("help", cmd_help))
dp.add_handler(CommandHandler("daily_now", cmd_daily_now))
dp.add_handler(CommandHandler("weekly_now", cmd_weekly_now))

@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dp.process_update(update)
    return "ok"

@app.route("/")
def index():
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
