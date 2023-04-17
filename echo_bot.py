import telebot
import numpy as np
from infer import detect_spoken_language_or_accent
from dataflow.utils import format_audio
from dataflow.utils import read_yaml


bot = telebot.TeleBot('')


@bot.message_handler(commands=['start'])
def send_welcome(message):
    name = message.from_user.first_name
    bot.reply_to(message, f"allooo")


@bot.message_handler(commands=['help'])
def send_welcome(message):
    name = message.from_user.first_name
    print('help - ', name)
    bot.reply_to(message, f"kjsfa")


# @bot.message_handler(content_types=['audio', 'voice'])
# def detect_spoken_language_or_accent(message):
#     message = format_audio(message, 16000, 5, True)
#     label = detect_spoken_language_or_accent(message, '', '')


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    text = aaa()
    config = read_yaml('./configs/model/config.yaml')
    t = detect_spoken_language_or_accent(config['encodings']['language_detection'])
    bot.reply_to(message, t)


bot.infinity_polling()
