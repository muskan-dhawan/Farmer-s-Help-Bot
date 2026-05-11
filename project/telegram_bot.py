import os
import logging
import pytz

# 🔥 Fix APScheduler issue (Windows + Python 3.13)
import apscheduler.util
def patched_astimezone(obj):
    if obj is None: return pytz.utc
    if isinstance(obj, str): return pytz.timezone(obj)
    if not hasattr(obj, 'localize'): return pytz.utc
    return obj
apscheduler.util.astimezone = patched_astimezone

import sys
import io

# 🔥 Fix encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    filters,
    CommandHandler,
    ContextTypes
)
from dotenv import load_dotenv

# 🔥 Import bot logic + DB
from bot.bot_logic import chatbot_response
from db.database_util import DatabaseManager

# Load env
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize DB
db = DatabaseManager(os.getenv("BOT_DB_PATH", "farmer_bot.sqlite3"))


# 🟢 /start COMMAND (ONBOARDING START
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)

    logger.info(f"/start from {user_id}")

    # 🔥 Reset user flow
    db.upsert_user(user_id, mode="new")

    await update.message.reply_text(
        "🙏 Namaste!\nI will help you with farming.\n\nLet’s start..."
    )


# 🟢 HANDLE TEXT MESSAGES
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)
    user_text = update.message.text

    logger.info(f"[TEXT] {user_id}: {user_text}")

    try:
        reply = chatbot_response(user_id=user_id, text=user_text)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        reply = "⚠️ Something went wrong. Please try again."

    await update.message.reply_text(reply)


# 🟢 HANDLE LOCATION
async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)
    location = update.message.location

    lat = location.latitude
    lon = location.longitude

    logger.info(f"[LOCATION] {user_id}: {lat}, {lon}")

    try:
        reply = chatbot_response(user_id=user_id, lat=lat, lon=lon)
    except Exception as e:
        logger.error(f"Error processing location: {e}")
        reply = "⚠️ Failed to process location."

    await update.message.reply_text(reply)

# 🟢 MAIN FUNCTION

def main():
    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env file!")
        return

    print("🤖 Starting Telegram Farmer Bot...")

    app = ApplicationBuilder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))

    # Text messages
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    # Location messages
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))

    print("🤖 Bot is running...")
    app.run_polling()



if __name__ == "__main__":
    main()