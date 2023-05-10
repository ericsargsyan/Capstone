import os
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram.ext import MessageHandler, Filters
from telegramBot.list_of_replies import *
from infer import detect_spoken_language_and_dialect
from model import AudioModel
from utils import ogg_to_wav, handle_same_filename
from dataflow.utils import read_yaml, format_audio
# from telegram.utils.helpers import escape
import argparse
from telegram import ParseMode


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)

    return parser.parse_args()


TOKEN = '6069729257:AAEv27ZPk18AQ8MOPMrjSxilXi3wbD6laQs'
bot = telegram.Bot(token=TOKEN)


def get_user_info(update):
    try:
        user = update.message.from_user
        name = user.first_name
        last_name = user.last_name if user.last_name is not None else ''
        last_name = f'_{last_name}' if last_name is not None else last_name
        username = user.username if user.username is not None else ''
    except AttributeError:
        user = update.effective_user
        name = user.first_name
        last_name = user.last_name if user.last_name is not None else ''
        last_name = f'_{last_name}' if last_name is not None else last_name
        username = user.username if user.username is not None else ''

    return name, last_name, username


def start(update, context):
    chat_id = update.message.chat_id
    user_name = update.message.from_user.first_name
    language = context.user_data.get('language', 'en')

    message = f"Hello, {user_name}! My name is VoiceSense."
    keyboard = [
        [InlineKeyboardButton(languages[language], callback_data='change_language')],
        [InlineKeyboardButton(main_menu_train_languages[language], callback_data='trained_languages_of_model')],
        [InlineKeyboardButton(main_menu_train_dialects[language], callback_data='trained_dialects_of_model')],
        [InlineKeyboardButton(about[language], callback_data='about')],
        [InlineKeyboardButton(helps[language], callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)


def help_command(update, context):
    language = context.user_data.get('language', 'en')
    update.message.reply_text(help_text[language])


def help_callback(update, context):
    chat_id = update.callback_query.message.chat_id
    language = context.user_data.get('language', 'en')

    query = update.callback_query
    query.answer()
    message = help_text[language]

    keyboard = [
        [InlineKeyboardButton(back_texts[language], callback_data='back')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    context.bot.edit_message_text(chat_id=query.message.chat_id,
                                  message_id=query.message.message_id,
                                  text=message,
                                  reply_markup=reply_markup)


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
    elif language == 'ru':
        greeting = 'Привет, {}! Меня зовут VoiceSense.'

    user_name = query.from_user.first_name
    message = greeting.format(user_name)

    keyboard = [
        [InlineKeyboardButton(languages[language], callback_data='change_language')],
        [InlineKeyboardButton(main_menu_train_languages[language], callback_data='trained_languages_of_model')],
        [InlineKeyboardButton(main_menu_train_dialects[language], callback_data='trained_dialects_of_model')],
        [InlineKeyboardButton(about[language], callback_data='about')],
        [InlineKeyboardButton(helps[language], callback_data='help')]
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
        [InlineKeyboardButton("Русский 🇷🇺", callback_data='ru')],
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
    elif language == 'ru':
        message = f'Здравствуйте, {query.from_user.first_name}! Меня зовут VoiceSense.'

    keyboard = [
        [InlineKeyboardButton(languages[language], callback_data='change_language')],
        [InlineKeyboardButton(main_menu_train_languages[language], callback_data='trained_languages_of_model')],
        [InlineKeyboardButton(main_menu_train_dialects[language], callback_data='trained_dialects_of_model')],
        [InlineKeyboardButton(about[language], callback_data='about')],
        [InlineKeyboardButton(helps[language], callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.edit_message_text(chat_id=query.message.chat_id,
                                  message_id=query.message.message_id,
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


def handle_audio(update, context, model, path, samplerate, duration, encodings):
    audio_file = context.bot.getFile(update.message.audio.file_id)
    language = context.user_data.get('language', 'en')

    update.message.reply_text(voice_received[language])
    audio_file_extension = audio_file.file_path.split('.')[-1]

    name, last_name, username = get_user_info(update)
    telegram_user = username if username is not None else f'{name}{last_name}'

    location = os.path.join(path, f'{telegram_user}.{audio_file_extension}')
    audio_file.download(location)
    final_destination = handle_same_filename(f'{location.split(".")[0]}.wav')

    os.system(f'ffmpeg -y -hwaccel cuda -i {location} -acodec pcm_s16le -ac 1 -ar 16000 {final_destination}')
    os.remove(location)
    data = format_audio(final_destination, self_samplerate=samplerate, resample=False, self_duration=duration)

    spoken_language = detect_spoken_language_and_dialect(data, model, encodings)
    reply = detected_language[language][spoken_language]

    update.message.reply_text(reply)


def handle_voice(update, context, path, language_model, dialect_model, samplerate, duration,
                 language_encodings, dialect_encodings):
    voice_file = context.bot.getFile(update.message.voice.file_id)
    language = context.user_data.get('language', 'en')
    mode = context.user_data.get('detection_mode', 'language')
    update.message.reply_text(voice_received[language])

    name, last_name, username = get_user_info(update)
    telegram_user = username if username != '' else f'{name}{last_name}'
    print('--------------')
    print('username', username)
    print('name', name)
    print('last_name', last_name)
    print('telegram_user', telegram_user)
    print(username)

    location = handle_same_filename(os.path.join(path, f'{telegram_user}.wav'))
    voice_file.download(os.path.join(path, f'{telegram_user}.ogg'))
    ogg_to_wav(voice_file.download(os.path.join(path, f'{telegram_user}.ogg')), location)

    data = format_audio(location, self_samplerate=samplerate, resample=False, self_duration=duration)

    if mode == 'language':
        spoken_language = detect_spoken_language_and_dialect(data, language_model, language_encodings)
        reply = detected_language[language][spoken_language]

        update.message.reply_text(reply)
    elif mode == 'dialect':
        spoken_dialect = detect_spoken_language_and_dialect(data, dialect_model, dialect_encodings)
        # reply = detected_dialect[language][spoken_dialect]

        update.message.reply_text(spoken_dialect)


def handle_document(update, context, model, path, samplerate, duration, encodings):
    doc_file = context.bot.getFile(update.message.document.file_id)
    language = context.user_data.get('language', 'en')

    update.message.reply_text(voice_received[language])
    document_file_extension = doc_file.file_path.split('.')[-1]

    audios_extensions = ['wav', 'mp3', 'mp4', 'm4a', 'ogg', 'oga', 'flac']

    name, last_name, username = get_user_info(update)
    telegram_user = username if username is not None else f'{name}{last_name}'

    if document_file_extension not in audios_extensions:
        update.message.reply_text(not_audio_message[language])
        return

    location = os.path.join(path, f'{telegram_user}.{document_file_extension}')
    doc_file.download(location)
    final_destination = handle_same_filename(f'{location.split(".")[0]}.wav')

    os.system(f'ffmpeg -y -hwaccel cuda -i {location} -acodec pcm_s16le -ac 1 -ar 16000 {final_destination}')

    if location[-3:] != 'wav':
        os.remove(location)

    data = format_audio(location, self_samplerate=samplerate, resample=False, self_duration=duration)

    spoken_language = detect_spoken_language_and_dialect(data, model, encodings)
    reply = detected_language[language][spoken_language]

    update.message.reply_text(reply)


def handle_message(update, context):
    language = context.user_data.get('language', 'en')

    if update.message.text.startswith('/') and update.message.text[1:] not in ['start', 'help']:
        context.bot.send_message(chat_id=update.effective_chat.id, text=invalid_command[language])
    else:
        response = handle_messages[language]
        context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def trained_languages_of_model(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')

    trained_languages = pretrained_languages[language]

    keyboard = []
    row = []
    for lang in trained_languages:
        row.append(InlineKeyboardButton(lang, callback_data=f'selected_train_language:{lang}'))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    keyboard.append([InlineKeyboardButton(back_texts[language], callback_data='back')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    query.edit_message_text(text=train_lang_reply[language], reply_markup=reply_markup)


def trained_dialects_of_model(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')
    trained_dialects = pretrained_dialects[language]

    keyboard = []
    for lang in trained_dialects:
        keyboard.append([InlineKeyboardButton(lang, callback_data=lang)])

    keyboard.append([InlineKeyboardButton(back_texts[language], callback_data='back')])

    reply_markup = InlineKeyboardMarkup(keyboard)
    query.edit_message_text(text=train_dialect_reply[language], reply_markup=reply_markup)


def selected_train_language(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')
    selected_language = query.data.split(':')[1]

    lang_info = languages_information[language][selected_language]
    back_button = InlineKeyboardButton(back_texts[language], callback_data='back')
    reply_markup = InlineKeyboardMarkup([[back_button]])

    query.edit_message_text(text=lang_info, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)


def selected_train_dialect(update, context):
    query = update.callback_query
    language = context.user_data.get('language', 'en')
    selected_language = query.data.split(':')[1]

    dialect_info = dialects_information[language][selected_language]
    back_button = InlineKeyboardButton(back_texts[language], callback_data='back')
    reply_markup = InlineKeyboardMarkup([[back_button]])

    query.edit_message_text(text=dialect_info, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)


def model_callback(update, context):
    language = context.user_data.get('language', 'en')

    keyboard = [
        [InlineKeyboardButton("Detect Language", callback_data='language')],
        [InlineKeyboardButton("Detect Dialect", callback_data='dialect')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text(text=detection_mode[language], reply_markup=reply_markup)


def detect_language(update, context):
    context.user_data['detection_mode'] = 'language'
    language = context.user_data.get('language', 'en')

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=detection_mode_language[language])


def detect_dialect(update, context):
    context.user_data['detection_mode'] = 'dialect'
    language = context.user_data.get('language', 'en')

    update.callback_query.answer()
    update.callback_query.edit_message_text(text=detection_mode_dialect[language])


if __name__ == '__main__':
    parser = arg_parser()
    config = read_yaml(parser.config_path)
    path_to_voices = config['telegram_voices']
    model_config = read_yaml(config['model_config_path'])

    language_checkpoint = model_config['checkpoint_path']['language']
    dialect_checkpoint = model_config['checkpoint_path']['accent']
    language_labels = max(model_config['encodings']['language_detection'].values()) + 1
    dialect_labels = max(model_config['encodings']['accent_detection'].values()) + 1
    language_encodings = model_config['encodings']['language_detection']
    dialect_encodings = model_config['encodings']['accent_detection']

    model_configs_dict = {'model_config': read_yaml(model_config['model_config_path']),
                          'processor_config': model_config['audio_processor'],
                          'sr': model_config['sr'], 'learning_rate': model_config['learning_rate']}

    os.makedirs(config['telegram_voices'], exist_ok=True)

    language_model = AudioModel.load_from_checkpoint(checkpoint_path=language_checkpoint,
                                                     number_of_labels=language_labels,
                                                     encodings=language_encodings,
                                                     **model_configs_dict)
    language_model.eval()

    dialect_model = AudioModel.load_from_checkpoint(checkpoint_path=dialect_checkpoint,
                                                    number_of_labels=dialect_labels,
                                                    encodings=dialect_encodings,
                                                    **model_configs_dict)
    dialect_model.eval()

    parameters = {'path': path_to_voices, 'samplerate': config['samplerate'], 'duration': config['duration'],
                  'language_model': language_model, 'dialect_model': dialect_model,
                  'language_encodings': language_encodings, 'dialect_encodings': dialect_encodings}

    updater = Updater(token=TOKEN, use_context=True)
    telegram = updater.dispatcher

    telegram.add_handler(CommandHandler('start', start))
    telegram.add_handler(CommandHandler('help', help_command))
    telegram.add_handler(CommandHandler('model', model_callback))
    telegram.add_handler(CallbackQueryHandler(detect_language, pattern='language'))
    telegram.add_handler(CallbackQueryHandler(detect_dialect, pattern='dialect'))
    telegram.add_handler(CallbackQueryHandler(language_callback, pattern='^(en|es|fr|hy|ru)$'))
    telegram.add_handler(CallbackQueryHandler(change_language_callback, pattern='^change_language$'))
    telegram.add_handler(CallbackQueryHandler(trained_languages_of_model, pattern='trained_languages_of_model'))
    telegram.add_handler(CallbackQueryHandler(trained_dialects_of_model, pattern='trained_dialects_of_model'))
    telegram.add_handler(CallbackQueryHandler(back_callback, pattern='^back$'))
    telegram.add_handler(CallbackQueryHandler(about_callback, pattern='^about$'))
    telegram.add_handler(CallbackQueryHandler(help_callback, pattern='^help$'))
    telegram.add_handler(MessageHandler(Filters.audio,
                                        lambda update, context: handle_audio(update, context, **parameters)))
    telegram.add_handler(MessageHandler(Filters.voice,
                                        lambda update, context: handle_voice(update, context, **parameters)))
    telegram.add_handler(MessageHandler(Filters.document,
                                        lambda update, context: handle_document(update, context, **parameters)))
    telegram.add_handler(MessageHandler(Filters.text, handle_message))
    telegram.add_handler(CallbackQueryHandler(selected_train_language, pattern='selected_train_language:*'))
    telegram.add_handler(CallbackQueryHandler(selected_train_dialect, pattern='selected_train_dialect:*'))

    updater.start_polling()
    updater.idle()
