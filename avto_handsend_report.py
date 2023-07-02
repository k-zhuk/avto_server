import io
import glob
import os
from telegram.ext import CommandHandler
from telegram.ext import Updater
from datetime import datetime
from datetime import timedelta
from dotenv import dotenv_values


def send_last_report(update, context):

    envs = dotenv_values('/home/server_bot/.env')
    report_filenames = glob.glob(pathname=f'{envs["HOME_AVTOBOT"]}/report_backups/*')
    report_filenames.sort(key=os.path.getctime)

    with open(f'{report_filenames[-1]}', 'rb') as file:
        pdf_bytes = io.BytesIO(file.read())

    pdf_bytes.seek(0)
    pdf_bytes.name = f'handsend_report_{datetime.date(datetime.today() - timedelta(days=1))}.pdf'
    context.bot.send_document(chat_id=envs['AVTOBOT_CHANNEL_ID'],
                              document=pdf_bytes)


envs = dotenv_values('/home/server_bot/.env')

updater = Updater(token=envs['AVTOBOT_TELEGRAM_TOKEN'])
dispatcher = updater.dispatcher
report_handler = CommandHandler(envs['HANDSEND_REPORT_COMMAND'],
                                send_last_report)
dispatcher.add_handler(report_handler)
updater.start_polling()
