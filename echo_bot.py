import telebot
import numpy as np

bot = telebot.TeleBot("")


@bot.message_handler(commands=['start'])
def send_welcome(message):
    name = message.from_user.first_name
    bot.reply_to(message, f"")


@bot.message_handler(commands=['help'])
def send_welcome(message):
    name = message.from_user.first_name
    print('help - ', name)
    bot.reply_to(message, f"")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, f'')


bot.infinity_polling()
