import telegram
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram.ext import MessageHandler, Filters

TOKEN = '6182809550:AAFTsLl8I_aVIHG2BpK8kLO_cjyDbLErqJs'
bot = telegram.Bot(token=TOKEN)


def reply(update, context):
    message = update.message
    chat_id = message.chat_id
    bot.send_message(chat_id=chat_id, text='Training still goes')


if __name__ == '__main__':
    updater = Updater(token=TOKEN, use_context=True)

    updater.dispatcher.add_handler(MessageHandler(Filters.all, reply))

    updater.start_polling()
    updater.idle()
