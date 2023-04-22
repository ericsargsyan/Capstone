import os
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram.ext import MessageHandler, Filters
from list_of_replies import *
from infer import detect_spoken_language_or_accent
from utils import ogg_to_wav
from dataflow.utils import read_yaml, format_audio
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True)
    return parser.parse_args()


TOKEN = '6069729257:AAGxHF5C19MEuHTp30RO3eBYwai8XdeDISg'
bot = telegram.Bot(token=TOKEN)


def start(update, context):
    chat_id = update.message.chat_id
    user_name = update.message.from_user.first_name

    message = f"Hello, {user_name}! My name is VoiceSense."
    keyboard = [
        [InlineKeyboardButton(languages[context.user_data.get('language', 'en')], callback_data='change_language')],
        [InlineKeyboardButton(about[context.user_data.get('language', 'en')], callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)


def help(update, context):
    language = context.user_data.get('language', 'en')
    update.message.reply_text(help_text[language])


def language_callback(update, context):
    query = update.callback_query
    language = query.data

    context.user_data['language'] = language

    if language == 'en':
        greeting = "Hello, {}! My name is VoiceSense."
    elif language == 'es':
        greeting = "¡Hola, {}! Mi nombre es VoiceSense."
    elif language == 'hy':
        greeting = 'Բարեւ Ձեզ, {}! Իմ անունը VoiceSense է:'
    elif language == 'fr':
        greeting = "Bonjour, {}! Je m'appelle VoiceSense."

    user_name = query.from_user.first_name
    message = greeting.format(user_name)

    keyboard = [
        [InlineKeyboardButton(languages[language], callback_data='change_language')],
        [InlineKeyboardButton(about[language], callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.edit_message_text(chat_id=query.message.chat_id, message_id=query.message.message_id,
                                  text=message, reply_markup=reply_markup)


def change_language_callback(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')

    message = change_language[language]
    keyboard = [
        [InlineKeyboardButton("English 🇺🇸", callback_data='en'),
         InlineKeyboardButton("Español 🇪🇸", callback_data='es')],
        [InlineKeyboardButton("Հայերեն 🇦🇲", callback_data='hy'),
         InlineKeyboardButton("Français 🇫🇷", callback_data='fr')],
        [InlineKeyboardButton(back_texts[language], callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.edit_message_text(chat_id=query.message.chat_id, message_id=query.message.message_id,
                                  text=message, reply_markup=reply_markup)


def back_callback(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')

    if language == 'en':
        message = f"Hello, {query.from_user.first_name}! My name is VoiceSense."
    elif language == 'es':
        message = f"Hola, {query.from_user.first_name}! Mi nombre es VoiceSense."
    elif language == 'hy':
        message = f'Բարեւ Ձեզ, {query.from_user.first_name}! Իմ անունը VoiceSense է:'
    elif language == 'fr':
        message = f"Bonjour, {query.from_user.first_name}! Je m'appelle VoiceSense."

    keyboard = [
        [InlineKeyboardButton(languages[language], callback_data='change_language')],
        [InlineKeyboardButton(about[language], callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.edit_message_text(chat_id=query.message.chat_id, message_id=query.message.message_id,
                                  text=message,
                                  reply_markup=reply_markup)


def about_callback(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')
    message = intro[language]

    keyboard = [
        [InlineKeyboardButton(back_texts[language], callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.edit_message_text(chat_id=query.message.chat_id, message_id=query.message.message_id,
                                  text=message, reply_markup=reply_markup)


def handle_audio(update, context):
    audio_file = context.bot.getFile(update.message.audio.file_id)
    language = context.user_data.get('language', 'en')
    # update.message.reply_text(voice_received[language])


def handle_voice(update, context, path, samplerate, duration):
    voice_file = context.bot.getFile(update.message.voice.file_id)
    language = context.user_data.get('language', 'en')
    update.message.reply_text(voice_received[language])

    name = update.message.from_user.first_name
    last_name_or_username = update.message.from_user.last_name or update.message.from_user.username
    last_name_or_username = '' if last_name_or_username is None else f'_{last_name_or_username}'

    ogg_file = f"{name}_{last_name_or_username}.ogg"
    wav_file = os.path.join(path, f"{name}{last_name_or_username}.wav")

    # if os.path.exists(wav_file):
    #     wav_file = f'{wav_file}_{1}'

    voice_file.download(ogg_file)
    ogg_to_wav(ogg_file, wav_file)

    data = format_audio(wav_file,
                        self_samplerate=samplerate,
                        resample=False,
                        self_duration=duration)

    # detect_spoken_language_or_accent()

    # context.bot.send_audio(chat_id=update.message.chat_id, audio=open(wav_file, 'rb'))

    os.remove(ogg_file)


def handle_message(update, context):
    language = context.user_data.get('language', 'en')

    if update.message.text.startswith('/'):
        pass
        # handle_command(update, context)
    else:
        response = handle_messages[language]
        context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def trained_languages_of_model(update, context):
    language = context.user_data.get('language', 'en')
    update.message.reply_text(f"{train_reply[language]} \n{trained_languages[language]}")


def trained_accents_of_model(update, context):
    language = context.user_data.get('language', 'en')
    update.message.reply_text(f"{train_reply[language]} \n{trained_accents[language]}")


if __name__ == '__main__':
    parser = arg_parser()
    config = read_yaml(parser.config_path)
    path_to_voices = config['voices_path']

    updater = Updater(token=TOKEN, use_context=True)
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help', help))
    updater.dispatcher.add_handler(CommandHandler("languages", trained_languages_of_model))
    updater.dispatcher.add_handler(CommandHandler("accents", trained_accents_of_model))
    updater.dispatcher.add_handler(CallbackQueryHandler(language_callback, pattern='^(en|es|fr|hy)$'))
    updater.dispatcher.add_handler(CallbackQueryHandler(change_language_callback, pattern='^change_language$'))
    updater.dispatcher.add_handler(CallbackQueryHandler(back_callback, pattern='^back$'))
    updater.dispatcher.add_handler(CallbackQueryHandler(about_callback, pattern='^help$'))
    updater.dispatcher.add_handler(MessageHandler(Filters.audio, handle_audio))
    updater.dispatcher.add_handler(MessageHandler(Filters.voice,
                                                  lambda update, context:
                                                  handle_voice(update, context,
                                                               path=path_to_voices,
                                                               samplerate=config['samplerate'],
                                                               duration=config['duration']
                                                               )
                                                  )
                                   )
    updater.dispatcher.add_handler(MessageHandler(Filters.text, handle_message))

    updater.start_polling()
    updater.idle()