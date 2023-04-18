import os
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram.ext import MessageHandler, Filters
from infer import detect_spoken_language_or_accent
from utils import ogg_to_wav


TOKEN = ''
bot = telegram.Bot(token=TOKEN)

languages = {'en': 'Change Language 🌐',
             'es': 'Cambiar Idioma 🌐',
             'fr': 'Changer de Langue 🌐',
             'hy': 'Փոխել Լեզուն 🌐'}

back_texts = {'en': 'Back 🔙',
              'es': 'Atrás 🔙',
              'fr': 'Retour 🔙',
              'hy': 'Ետ 🔙'}

helps = {'en': 'Help ❓',
         'es': 'Ayuda ❓',
         'fr': 'Aider ❓',
         'hy': 'Օգնություն ❓'}


def start(update, context):
    chat_id = update.message.chat_id
    user_name = update.message.from_user.first_name

    message = f"Hello, {user_name}! My name is VoiceSense."
    keyboard = [
        [InlineKeyboardButton(languages[context.user_data.get('language', 'en')], callback_data='change_language')],
        [InlineKeyboardButton(helps[context.user_data.get('language', 'en')], callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)


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
        [InlineKeyboardButton(helps[language], callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.edit_message_text(chat_id=query.message.chat_id, message_id=query.message.message_id,
                                  text=message, reply_markup=reply_markup)


def change_language_callback(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')

    message = "Please select your language:"
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
        [InlineKeyboardButton(helps[language], callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.edit_message_text(chat_id=query.message.chat_id, message_id=query.message.message_id,
                                  text=message, reply_markup=reply_markup)


def help_callback(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')

    if language == 'en':
        message = "Let me introduce myself.\n" \
                  "I am a Telegram Bot created by the students of the Faculty of Mathematics and Mechanics at YSU," \
                  "Eric Sargsyan and Gor Piliposyan." \
                  "I am capable of detecting the language and words from human speech."
    elif language == 'es':
        message = "Este es el mensaje de ayuda. ¿En qué puedo ayudarte aún más?"
    elif language == 'hy':
        message = 'Միքիչ պատմեմ իմ մասին:\n' \
                  'Ես Telegram Bot եմ, որը ստեծվել է ԵՊՀ Մաթեմատիկայի և մեխանիկայի ֆակուլտետի ուսանողներ ' \
                  'Էրիկ Սարգսյանի և Գոռ Փիլիպոսյանի կողմից։ Ես կարողանում եմ ճանաչել լեզուն և բարբառը մարդու խոսքից։'
    elif language == 'fr':
        message = "Ceci est le message d'aide. Comment puis-je vous aider davantage ?"

    keyboard = [
        [InlineKeyboardButton(back_texts[language], callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.edit_message_text(chat_id=query.message.chat_id, message_id=query.message.message_id,
                                  text=message, reply_markup=reply_markup)


def handle_audio(update, context):
    audio_file = context.bot.getFile(update.message.audio.file_id)
    audio_file.download(f"audio_{update.message.chat_id}.ogg")
    update.message.reply_text("Audio message received.")


def handle_voice(update, context):
    voice_file = context.bot.getFile(update.message.voice.file_id)
    update.message.reply_text("Voice message received.")

    name = update.message.from_user.first_name
    last_name_or_username = update.message.from_user.last_name or update.message.from_user.username
    last_name_or_username = '' if last_name_or_username is None else last_name_or_username

    ogg_file = f"{name}_{last_name_or_username}.ogg"
    wav_file = f"{name}_{last_name_or_username}.wav"

    if os.path.exists(wav_file):
        wav_file = f'{wav_file}_{1}'

    voice_file.download(ogg_file)
    ogg_to_wav(ogg_file, wav_file)

    # Check if the WAV file was created
    if not os.path.isfile(wav_file):
        update.message.reply_text("Error converting voice message.")
        return

    # Send the WAV file back to the user
    context.bot.send_audio(chat_id=update.message.chat_id, audio=open(wav_file, 'rb'))

    os.remove(ogg_file)


updater = Updater(token=TOKEN, use_context=True)
updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CallbackQueryHandler(language_callback, pattern='^(en|es|fr|hy)$'))
updater.dispatcher.add_handler(CallbackQueryHandler(change_language_callback, pattern='^change_language$'))
updater.dispatcher.add_handler(CallbackQueryHandler(back_callback, pattern='^back$'))
updater.dispatcher.add_handler(CallbackQueryHandler(help_callback, pattern='^help$'))
updater.dispatcher.add_handler(MessageHandler(Filters.audio, handle_audio))
updater.dispatcher.add_handler(MessageHandler(Filters.voice, handle_voice))
# updater.dispatcher.add_handler(MessageHandler(Filters.voice, lambda update, context: handle_voice(update, context, path="/path/to/save/files")))


updater.start_polling()
updater.idle()
