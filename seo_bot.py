import os
import io
import json
import logging
import asyncio
import aiohttp
from dotenv import load_dotenv

import pandas as pd
from docx import Document


from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, BufferedInputFile
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup


# Настройка логирования

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_seo.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения

load_dotenv()
TG_TOKEN = os.getenv("TG_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PIXEL_TOOLS_KEY = os.getenv("PIXEL_TOOLS_KEY")

if not TG_TOKEN or not OPENROUTER_API_KEY:
    logger.error("TG_TOKEN или OPENROUTER_API_KEY не найдены. Установите их в .env файле.")
    exit(1)

# Настройка OpenRouter
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterChat:
    """Чат-сессия с историей сообщений через OpenRouter API."""

    def __init__(self):
        self.history = []

    async def send_message(self, content: str) -> str:
        self.history.append({"role": "user", "content": content})

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "google/gemini-2.5-flash",
            "max_tokens": 4096,
            "messages": self.history,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
                data = await resp.json()

                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    raise Exception(f"OpenRouter API error: {error_msg}")

                assistant_msg = data["choices"][0]["message"]["content"]
                self.history.append({"role": "assistant", "content": assistant_msg})

                # Возвращаем объект с .text для совместимости с остальным кодом
                class Response:
                    def __init__(self, text):
                        self.text = text
                return Response(assistant_msg)

# Инициализация бота и диспетчера
bot = Bot(token=TG_TOKEN)
dp = Dispatcher()


# FSM (Конечный автомат)

class SeoProcess(StatesGroup):
    waiting_for_file = State()
    step0_structure = State()
    step1_tz = State()
    step2_text = State()

# Хранилище сессий Gemini (для сохранения контекста history)
USER_CHATS = {}

# Хранилище временных данных пользователя (ТЗ, Текст)
USER_DATA = {}


# Вспомогательные функции

def get_or_create_chat(user_id: int):
    """Создает или возвращает существующую сессию чата OpenRouter для пользователя."""
    if user_id not in USER_CHATS:
        USER_CHATS[user_id] = OpenRouterChat()
    return USER_CHATS[user_id]

def clear_user_session(user_id: int):
    """Очищает данные пользователя после завершения процесса."""
    if user_id in USER_CHATS:
        del USER_CHATS[user_id]
    if user_id in USER_DATA:
        del USER_DATA[user_id]

async def safe_llm_request(chat, prompt, max_retries=3, delay=5):
    """Обертка для запросов к OpenRouter с обработкой ошибок и ретраями."""
    for attempt in range(max_retries):
        try:
            return await chat.send_message(prompt)
        except Exception as e:
            if ("503" in str(e) or "429" in str(e) or "rate" in str(e).lower()) and attempt < max_retries - 1:
                logger.warning(f"OpenRouter ошибка (попытка {attempt + 1}/{max_retries}) через {delay} сек: {e}")
                await asyncio.sleep(delay)
                continue
            raise e

async def send_long_text(status_msg: Message, text: str, reply_markup=None):
    """Разбивает длинный текст на части и отправляет их."""
    max_len = 4000
    parts = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    
    for i, part in enumerate(parts):
        markup = reply_markup if i == len(parts) - 1 else None
        if i == 0:
            await status_msg.edit_text(part, reply_markup=markup)
        else:
            await status_msg.answer(part, reply_markup=markup)

async def get_top_competitors(keys):
    """
    Получение списка конкурентов через PixelPlus API (метод top10).
    Постановка задачи, ожидание выполнения (polling) и парсинг результатов.
    """
    if not PIXEL_TOOLS_KEY:
        logger.error("PIXEL_TOOLS_KEY не установлен.")
        return []

    # Константы для PixelPlus
    API_URL = "https://tools.pixelplus.ru/api/top10"
    REGION = "213"  # Москва
    SEARCH_ENGINE = "yandex"
    
    logger.info(f"Запуск задачи PixelPlus для {len(keys)} ключей.")
    
    async with aiohttp.ClientSession() as session:
        # 1. Постановка задачи
        payload = {
            'queries': json.dumps(keys),
            'lr': REGION,
            'ss': SEARCH_ENGINE,
            'deep': 10
        }
        
        # Передаем ключ в params (URL), а данные в data (body)
        params = {'key': PIXEL_TOOLS_KEY}
        
        try:
            logger.info(f"Отправка запроса в PixelPlus... Ключ: {PIXEL_TOOLS_KEY[:4]}***")
            async with session.post(API_URL, params=params, data=payload) as resp:
                result = await resp.json()
                
                if 'report_id' not in result:
                    logger.error(f"Ошибка постановки задачи PixelPlus: {result}")
                    return []
                
                report_id = result['report_id']
                logger.info(f"Задача PixelPlus создана. ID: {report_id}")
        except Exception as e:
            logger.error(f"Ошибка при обращении к PixelPlus (постановка): {e}")
            return []


        # 2. Ожидание результата (Polling)
        max_attempts = 30 # ~5 минут макс
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            await asyncio.sleep(10) # Проверяем каждые 10 секунд
            
            try:
                params = {'key': PIXEL_TOOLS_KEY, 'report_id': report_id}
                async with session.get(API_URL, params=params) as resp:
                    data = await resp.json()
                    
                    # Обработка ответов
                    if data.get('status') == 'success' and data.get('msg') == 'completed':
                        logger.info(f"Задача PixelPlus {report_id} завершена успешно.")
                        
                        # Извлекаем URL
                        urls_found = set()
                        # Структура: response -> response -> result -> {query: {yandex_urls: [...]}}
                        results_data = data.get('response', {}).get('response', {}).get('result', {})
                        
                        for query_val in results_data.values():
                            platform_key = 'yandex_urls' if SEARCH_ENGINE.startswith('yandex') else 'google_urls'
                            raw_urls = query_val.get(platform_key, [])
                            for item in raw_urls:
                                if 'url' in item:
                                    urls_found.add(item['url'])
                        
                        return list(urls_found)[:20] # Возвращаем топ-20 уникальных URL
                    
                    elif data.get('code') == -50 or data.get('error') == 'In progress':
                        # В процессе
                        if attempt % 3 == 0:
                            logger.info(f"Задача {report_id} еще выполняется (попытка {attempt})...")
                        continue
                    else:
                        logger.error(f"Ошибка при получении задачи PixelPlus {report_id}: {data}")
                        break
                        
            except Exception as e:
                logger.error(f"Ошибка при обращении к PixelPlus (результат): {e}")
                break
                
        logger.warning(f"Превышено время ожидания задачи PixelPlus {report_id}.")
        return []


# Обработчики команд
@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.set_state(SeoProcess.waiting_for_file)
    clear_user_session(message.from_user.id)
    await message.answer(
        "👋 Привет! Я бот для генерации SEO-статей.\n\n"
        "Отправьте мне файл (.xlsx или .csv) со столбцами <b>'Фраза'</b> и <b>'WS'</b>.",
        parse_mode="HTML"
    )

@dp.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    clear_user_session(message.from_user.id)
    await message.answer("Процесс отменен. Если хотите начать заново, отправьте /start.")

# ШАГ 0: Получение Excel и составление структуры
@dp.message(SeoProcess.waiting_for_file, F.document)
async def process_excel_file(message: Message, state: FSMContext):
    document = message.document
    is_xlsx = document.file_name.lower().endswith('.xlsx')
    is_csv = document.file_name.lower().endswith('.csv')
    
    if not (is_xlsx or is_csv):
        await message.answer("❌ Пожалуйста, отправьте файл в формате .xlsx или .csv!")
        return

    status_msg = await message.answer("📥 Скачиваю файл... ⏳")
    user_id = message.from_user.id

    try:
        # Скачиваем файл в память
        file_in_memory = io.BytesIO()
        await bot.download(document, destination=file_in_memory)
        file_in_memory.seek(0)
        
        # Парсим через pandas
        if is_xlsx:
            df = pd.read_excel(file_in_memory)
        else:
            df = pd.read_csv(file_in_memory, sep=None, engine='python') # sep=None determines separator automatically
        
        
        # Проверяем наличие столбцов
        if 'Фраза' not in df.columns or 'WS' not in df.columns:
            await status_msg.edit_text("❌ Ошибка: В файле отсутствуют нужные столбцы 'Фраза' и/или 'WS'.")
            return

        keys = df['Фраза'].astype(str).tolist()
        ws_values = df['WS'].tolist()
        
        logger.info(f"[{user_id}] Загружен файл. Количество ключей: {len(keys)}")
        await status_msg.edit_text("🔍 Парсим конкурентов через Pixel Tools... ⏳")
        
        top_urls = await get_top_competitors(keys)
        
        await status_msg.edit_text("🧠 Анализирую ключи и генерирую структуру... ⏳\n(Это может занять немного времени)")

        # Формируем промпт
        keys_str = "\n".join([f"- {k} (WS: {w})" for k, w in zip(keys, ws_values)])
        prompt = (
            "Проанализируй эти ключи, оцени нишу, рассчитай оптимальный объем текста в символах "
            "и составь SEO-структуру (H1, H2, H3).\n\n"
            f"Ключи:\n{keys_str}"
        )
        
        if top_urls:
            urls_str = "\n".join([f"- {url}" for url in top_urls[:20]])
            prompt += f"\n\nПопулярные URL конкурентов в ТОП-10 (для анализа интента и ниши):\n{urls_str}"

        chat = get_or_create_chat(user_id)
        
        # Запуск запроса к Gemini с ретраями
        response = await safe_llm_request(chat, prompt)
        
        # Сохраняем структуру
        USER_DATA[user_id] = {'structure': response.text}
        
        # Создаем Inline-клавиатуру
        markup = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Дальше", callback_data="step0_continue")],
            [InlineKeyboardButton(text="Пересобрать", callback_data="step0_regenerate")]
        ])

        await send_long_text(status_msg, response.text, markup)
        await state.set_state(SeoProcess.step0_structure)
        logger.info(f"[{user_id}] Успешно сгенерирована структура.")

    except Exception as e:
        logger.error(f"[{user_id}] Ошибка при обработке файла: {e}", exc_info=True)
        await status_msg.edit_text("❌ Произошла ошибка при обработке файла или обращении к API. Попробуйте еще раз.")


@dp.callback_query(SeoProcess.step0_structure, F.data.in_(["step0_continue", "step0_regenerate"]))
async def step0_callback(call: CallbackQuery, state: FSMContext):
    user_id = call.from_user.id
    chat = USER_CHATS.get(user_id)
    
    if not chat:
        await call.message.edit_text("❌ Сессия устарела. Напишите /start для перезапуска.")
        return

    # Убираем кнопки со старого сообщения
    await call.message.edit_reply_markup(reply_markup=None)

    if call.data == "step0_regenerate":
        status_msg = await call.message.answer("🔄 Пересобираю структуру... ⏳")
        try:
            prompt = "Предложи другой вариант структуры, предыдущий не подошел."
            response = await safe_llm_request(chat, prompt)
            
            USER_DATA[user_id]['structure'] = response.text
            
            markup = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Дальше", callback_data="step0_continue")],
                [InlineKeyboardButton(text="Пересобрать", callback_data="step0_regenerate")]
            ])
            await send_long_text(status_msg, response.text, markup)
            logger.info(f"[{user_id}] Структура пересобрана.")
        except Exception as e:
            logger.error(f"[{user_id}] Ошибка пересоздания: {e}")
            await status_msg.edit_text("❌ Произошла ошибка API. Попробуйте еще раз: /start")

    elif call.data == "step0_continue":
        await process_step1_tz(call.message, user_id, chat, state)
        await call.answer()



# ШАГ 1: Формирование ТЗ

async def process_step1_tz(message: Message, user_id: int, chat, state: FSMContext):
    status_msg = await message.answer("📝 Формирую Техническое Задание... ⏳")
    logger.info(f"[{user_id}] Старт генерации ТЗ.")
    
    try:
        prompt = (
            "Отлично. Теперь на основе утвержденной структуры составь строгое Техническое Задание (ТЗ) по шаблону:\n"
            "- Общие требования: разбить на блоки, списки на расстоянии 300 символов, между списками 450.\n"
            "- Ключевые фразы в точных вхождениях (указать количество).\n"
            "- Слова, задающие тематику (LSI).\n"
            "- Доп. требования: 1 картинка, 1 список.\n"
            "- Объем: [укажи рассчитанный ранее]."
        )
        response = await safe_llm_request(chat, prompt)
        
        USER_DATA[user_id]['tz'] = response.text
        
        markup = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Дальше", callback_data="step1_continue")],
            [InlineKeyboardButton(text="Пересобрать", callback_data="step1_regenerate")]
        ])
        
        await send_long_text(status_msg, response.text, markup)
        await state.set_state(SeoProcess.step1_tz)
        logger.info(f"[{user_id}] Успешно сгенерировано ТЗ.")
        
    except Exception as e:
        logger.error(f"[{user_id}] Ошибка генерации ТЗ: {e}")
        await status_msg.edit_text("❌ Произошла ошибка API при генерации ТЗ.")


@dp.callback_query(SeoProcess.step1_tz, F.data.in_(["step1_continue", "step1_regenerate"]))
async def step1_callback(call: CallbackQuery, state: FSMContext):
    user_id = call.from_user.id
    chat = USER_CHATS.get(user_id)
    
    if not chat:
        await call.message.edit_text("❌ Сессия устарела. Напишите /start")
        return

    await call.message.edit_reply_markup(reply_markup=None)

    if call.data == "step1_regenerate":
        status_msg = await call.message.answer("🔄 Переписываю ТЗ... ⏳")
        try:
            prompt = "Перепиши Техническое Задание по шаблону. Сделай его более подходящим."
            response = await safe_llm_request(chat, prompt)
            
            USER_DATA[user_id]['tz'] = response.text
            
            markup = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Дальше", callback_data="step1_continue")],
                [InlineKeyboardButton(text="Пересобрать", callback_data="step1_regenerate")]
            ])
            await send_long_text(status_msg, response.text, markup)
            logger.info(f"[{user_id}] ТЗ пересобрано.")
        except Exception as e:
            logger.error(f"[{user_id}] Ошибка пересоздания ТЗ: {e}")
            await status_msg.edit_text("❌ Произошла ошибка API.")

    elif call.data == "step1_continue":
        await process_step2_text(call.message, user_id, chat, state)
        await call.answer()


# ШАГ 2: Написание текста
async def process_step2_text(message: Message, user_id: int, chat, state: FSMContext):
    status_msg = await message.answer("✍️ Пишу итоговый текст (это может занять около минуты)... ⏳")
    logger.info(f"[{user_id}] Старт генерации текста.")
    
    try:
        prompt = "Теперь, строго соблюдая это ТЗ, напиши итоговый SEO-текст."
        # Запуск запроса к Gemini с ретраями
        response = await safe_llm_request(chat, prompt)
        
        USER_DATA[user_id]['text'] = response.text
        
        markup = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Сформировать Word-файл", callback_data="step2_continue")],
            [InlineKeyboardButton(text="Пересобрать текст", callback_data="step2_regenerate")]
        ])
        
        await send_long_text(status_msg, response.text, markup)
        await state.set_state(SeoProcess.step2_text)
        logger.info(f"[{user_id}] Успешно сгенерирован текст.")
        
    except Exception as e:
        logger.error(f"[{user_id}] Ошибка генерации текста: {e}")
        await status_msg.edit_text("❌ Произошла ошибка генерации текста. Попробуйте еще раз.")


@dp.callback_query(SeoProcess.step2_text, F.data.in_(["step2_continue", "step2_regenerate"]))
async def step2_callback(call: CallbackQuery, state: FSMContext):
    user_id = call.from_user.id
    chat = USER_CHATS.get(user_id)
    
    if not chat:
        await call.message.edit_text("❌ Сессия устарела. Напишите /start")
        return

    await call.message.edit_reply_markup(reply_markup=None)

    if call.data == "step2_regenerate":
        status_msg = await call.message.answer("🔄 Переписываю SEO-текст... ⏳")
        try:
            prompt = "Напиши другой вариант текста, строго соблюдая утвержденное ТЗ."
            response = await safe_llm_request(chat, prompt)
            
            USER_DATA[user_id]['text'] = response.text
            
            markup = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Сформировать Word-файл", callback_data="step2_continue")],
                [InlineKeyboardButton(text="Пересобрать текст", callback_data="step2_regenerate")]
            ])
            await send_long_text(status_msg, response.text, markup)
            logger.info(f"[{user_id}] Текст пересобран.")
        except Exception as e:
            logger.error(f"[{user_id}] Ошибка пересоздания текста: {e}")
            await status_msg.edit_text("❌ Произошла ошибка API.")

    elif call.data == "step2_continue":
        await process_step3_word(call.message, user_id, state)
        await call.answer()


# ШАГ 3: Создание Word-документа
async def process_step3_word(message: Message, user_id: int, state: FSMContext):
    status_msg = await message.answer("📄 Формирую Word-документ... ⏳")
    logger.info(f"[{user_id}] Старт генерации DOCX.")
    
    try:
        data = USER_DATA.get(user_id, {})
        tz_content = data.get('tz', 'ТЗ отсутствует')
        text_content = data.get('text', 'Текст отсутствует')
        
        # Работаем с python-docx
        doc = Document()
        
        # Страница 1: Техническое Задание
        doc.add_heading("Техническое Задание (ТЗ)", level=1)
        doc.add_paragraph(tz_content)
        
        doc.add_page_break()
        
        # Страница 2: Итоговый текст
        doc.add_heading("SEO-текст", level=1)
        doc.add_paragraph(text_content)
        
        # Сохранение в байтовый поток
        file_stream = io.BytesIO()
        doc.save(file_stream)
        file_stream.seek(0)
        
        # Отправка файла пользователю
        document = BufferedInputFile(file_stream.read(), filename="seo_article.docx")
        await message.answer_document(document=document, caption="✅ Ваш файл успешно сформирован!")
        await status_msg.delete()
        
        logger.info(f"[{user_id}] Успешно отправлен DOCX. Процесс завершен.")
        
        # Сброс состояния
        await state.clear()
        clear_user_session(user_id)
        
    except Exception as e:
        logger.error(f"[{user_id}] Ошибка создания Word-файла: {e}", exc_info=True)
        await status_msg.edit_text("❌ Возникла ошибка при формировании документа. Попробуйте снова.")

# Запуск бота
async def main():
    logger.info("Запуск бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка бота...")
