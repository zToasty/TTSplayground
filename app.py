import streamlit as st
import torch
import time
import numpy as np
import re
import io

# Заголовок
st.title("TTS Playground")
st.write("Тестирование скорости и качества генерации речи")

# Загрузка модели TTS (кешируем, чтобы не качать каждый раз)
@st.cache_resource
def load_tts_model():
    try:
        # Используем Silero TTS - качественная модель для русского языка
        # Она хорошо учитывает ударения, паузы и знаки препинания
        import torch
        
        # Загружаем модель Silero TTS v5 для русского языка
        language = 'ru'
        device = torch.device('cpu')  # Silero работает быстро даже на CPU
        
        # Загружаем модель v5_ru (правильный идентификатор модели)
        model, example_text = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker='v5_ru'  # Правильный идентификатор модели v5 для русского языка
        )
        model.to(device)
        
        # Доступные спикеры для русской модели v5: aidar, baya, kseniya, xenia, eugene, random
        default_speaker = 'xenia'  # Женский голос по умолчанию
        
        return model, language, default_speaker, device, None
    except Exception as e:
        error_msg = str(e)
        return None, None, None, None, error_msg

# Загрузка модели для расстановки ударений
@st.cache_resource
def load_stress_model():
    try:
        # Используем библиотеку ruaccent для расстановки ударений
        from ruaccent import RUAccent
        
        accentizer = RUAccent()
        # Загружаем модель (можно выбрать 'turbo3.1', 'big', 'medium', 'small')
        # turbo3.1 - самая быстрая и легкая модель
        accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)
        return accentizer, None
    except ImportError:
        error_msg = "Библиотека 'ruaccent' не установлена. Установите её командой: pip install ruaccent"
        return None, error_msg
    except Exception as e:
        # Сохраняем информацию об ошибке для отображения
        error_msg = str(e)
        return None, error_msg

# Функция для расстановки ударений
def add_stress_marks(text, accentizer=None):
    """
    Расставляет ударения в русском тексте.
    Использует библиотеку RUAccent если доступна, иначе возвращает исходный текст.
    """
    if accentizer is not None:
        try:
            # Обрабатываем текст с помощью RUAccent
            stressed_text = accentizer.process_all(text)
            return stressed_text
        except Exception as e:
            # Если произошла ошибка, возвращаем исходный текст
            return text
    
    # Если модель не загружена, возвращаем исходный текст
    return text

st.write("Загрузка моделей...")
tts_model, tts_language, tts_speaker, tts_device, tts_error = load_tts_model()

if tts_model is not None:
    st.success("✅ Модель TTS (Silero) загружена!")
else:
    st.error(f"Ошибка загрузки TTS модели: {tts_error}")
    st.info("💡 Убедитесь, что у вас установлен PyTorch и есть подключение к интернету для загрузки модели.")

# Загрузка модели ударений
accentizer, error_msg = load_stress_model()

if accentizer is not None:
    st.success("✅ Модель ударений (RUAccent) загружена!")
else:
    with st.expander("ℹ️ Информация о модели ударений", expanded=True):
        st.warning("⚠️ Модель ударений не загружена. Обработка ударений будет отключена.")
        if error_msg:
            st.error(f"**Детали ошибки:**")
            st.code(error_msg, language=None)
            if "ruaccent" in error_msg.lower():
                st.info("💡 **Решение:** Установите библиотеку ruaccent командой: `pip install ruaccent`")
        st.write("""
        **Возможные причины:**
        - Библиотека 'ruaccent' не установлена
        - Проблемы с подключением к Hugging Face
        - Модель временно недоступна
        - Недостаточно места на диске
        
        **Что делать:**
        - Установите библиотеку: `pip install ruaccent`
        - Проверьте подключение к интернету
        - Убедитесь, что у вас достаточно места на диске
        - Попробуйте перезапустить приложение
        - Приложение продолжит работать без обработки ударений
        """)
        
        # Кнопка для очистки кеша и повторной попытки
        if st.button("🔄 Попробовать загрузить модель снова"):
            st.cache_resource.clear()
            st.rerun()

# Настройки
st.sidebar.header("⚙️ Настройки")
use_stress = st.sidebar.checkbox("Использовать обработку ударений", value=True, 
                                  help="Расставляет ударения для более естественного произношения")

# Выбор голоса (только если модель загружена)
if tts_model is not None:
    available_speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
    speaker_names = {
        'aidar': 'Айдар (мужской)',
        'baya': 'Бая (женский)',
        'kseniya': 'Ксения (женский)',
        'xenia': 'Ксения (женский)',
        'eugene': 'Евгений (мужской)',
        'random': 'Случайный'
    }
    selected_speaker = st.sidebar.selectbox(
        "Выберите голос",
        options=available_speakers,
        index=3,  # xenia по умолчанию
        format_func=lambda x: speaker_names.get(x, x)
    )
else:
    selected_speaker = tts_speaker if tts_speaker else 'xenia'

# Ввод текста
text_input = st.text_area("Введите текст для озвучки", 
                          value="Привет! Как слышно?")

# Показываем обработанный текст с ударениями (только если включена обработка и модель доступна)
if use_stress and text_input and accentizer is not None:
    with st.spinner('Обрабатываю ударения...'):
        processed_text = add_stress_marks(text_input, accentizer)
        if processed_text != text_input:
            st.info(f"📝 Текст с ударениями: `{processed_text}`")
        else:
            processed_text = text_input
elif use_stress and accentizer is None:
    processed_text = text_input
    # Не показываем предупреждение здесь, так как оно уже показано выше
else:
    processed_text = text_input

# Кнопка генерации
if st.button("🎤 Сгенерировать голос"):
    if text_input and tts_model is not None:
        start_time = time.time()
        
        # Генерация аудио с помощью Silero TTS
        # Silero автоматически учитывает ударения (символ +), паузы и знаки препинания
        sample_rate = 48000  # Высокое качество
        
        with st.spinner('Генерирую речь...'):
            # Используем обработанный текст с ударениями (только если обработка включена и модель доступна)
            text_to_synthesize = processed_text if (use_stress and accentizer is not None) else text_input
            
            with torch.no_grad():
                # Используем правильный API для Silero TTS
                # Используем выбранный спикер из настроек
                current_speaker = selected_speaker if tts_model is not None else tts_speaker
                audio = tts_model.apply_tts(
                    text=text_to_synthesize,
                    speaker=current_speaker,
                    sample_rate=sample_rate
                )
            
            # Преобразуем в numpy array для Streamlit
            if isinstance(audio, torch.Tensor):
                audio_data = audio.cpu().numpy()
            else:
                audio_data = np.array(audio)
            
            # Убеждаемся, что аудио одномерное
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Нормализуем аудио, если нужно
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ограничиваем значения в диапазоне [-1, 1]
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

        end_time = time.time()
        inference_time = end_time - start_time
        
        st.audio(audio_data, sample_rate=sample_rate)
        emoji = "🚀" if inference_time < 1 else "🐢"
        st.info(f"**{emoji} ⏱️ Время инференса: {inference_time:.4f} сек.**")
        
    elif not text_input:
        st.warning("Сначала введите текст!")
    else:
        st.error("Модель TTS не загружена. Проверьте ошибки выше.")