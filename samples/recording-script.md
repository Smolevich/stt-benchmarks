# STT bench — что наговорить

20 голосовых: 10 RU + 10 EN, по 15–60 секунд каждое. Цель — покрыть разные сценарии, чтобы bench показал не только «модель X точнее на 0.3 %», а где она ломается.

Записывай в **iPhone Voice Memos** (или Mac QuickTime / Audacity). После каждой записи в `Диктофон.app` можно её переименовать в `ru-01-clean`, `ru-02-fast` и так далее — это сильно упростит мне сборку manifest-файла позже.

Под каждой строкой пометка какой кейс тестируем + параметр (микрофон, обстановка). Текст можешь не зубрить — главное произнести своими словами близко к смыслу. Бенч идёт по фактической ground-truth расшифровке, не по плановому тексту, так что отклонения нормальны.

---

## RU

### ru-01-clean — Чистая комната, ровный темп
> Сегодня прогоняю бенч на двадцати голосовых, сравниваю Whisper, Parakeet и GigaAM. Хочу понять, кто быстрее на ноуте без видеокарты и где архитектура решает больше, чем железо.

_Тихая комната, телефон в руке, ровно._

### ru-02-fast — Быстро, разговорно, с «эээ» и самоперебиваниями
> Слушай, я тут думаю, может быть, ну вот этот бот мой, он в принципе работает, но… короче, надо посмотреть, может Parakeet прикрутить, потому что Groq, ну, в целом ок, но русский у него такой себе.

_Скоростная разговорная речь, естественные «эээ», обрывы._

### ru-03-mix — Mix языков
> Вчера деплоил новую версию через GitHub Actions, и у меня pipeline упал на step build, потому что Node-восемнадцатой не было в runner-е. Пришлось matrix-у поправить.

_Английские термины в потоке русской речи — типичный девелоперский регистр._

### ru-04-digits — Цифры и домены
> Сервер двести один точка сто два точка тридцать восемь точка семь, порт восемьдесят три. Конфиг лежит в slash opt slash voice dash ai slash bot slash dot env, проверь GROQ_API_KEY.

_Цифры словами, технические идентификаторы — самое больное место для STT._

### ru-05-long — Длинная фраза с придаточными
> Если посмотреть на разницу между autoregressive и non-autoregressive моделями, особенно в контексте того, как они декодируют аудио — становится понятно, почему Parakeet, который работает по фреймам, на процессоре может обогнать Whisper, который генерирует токены последовательно.

_Одно длинное предложение с придаточными — проверка контекстного декодирования._

### ru-06-whisper — Тихо, почти шёпотом
> Сейчас ночь, дома все спят, и я тестирую как бот распознаёт тихую речь, говорю практически шёпотом, посмотрим что выдаст модель.

_Очень тихо, телефон близко ко рту, минимум воздуха._

### ru-07-names — Имена и термины (русские + английские)
> Стас Шупилкин, Telegram бот smolevich_voice_bot, репозиторий voice-ai, бенч скрипт stt landscape bench точка пай. Запускается через mlx_whisper, NeMo, transformers.

_Имена собственные, библиотеки, mix RU/EN — стресс-тест на пропс-нунах._

### ru-08-street — На улице или с шумом
> Вышел погулять, проверяю как бот справляется с уличным шумом, ветром и проезжающими машинами. Думаю, это самый честный тест для real-world кейса.

_Реальный outdoor: ветер / машины / шаги. Самое важное real-world условие._

### ru-09-short — Короткие команды, отрывисто
> Перезагрузи сервер. Сначала останови бота, потом редис, потом telegram bot api. Подожди пять секунд, запускай обратно.

_Короткие императивы, паузы между, без модальных слов._

### ru-10-podcast — Подкаст-стиль, спокойный монолог
> Когда мы выбираем STT-стек, первое что я делаю — смотрю не на бенчмарки, а на свою задачу. Real-time или batch? Один язык или много? Privacy критична или нет? От этих трёх вопросов всё дальше.

_Размеренный монолог, среднестудийный темп._

---

## EN

### en-01-clean — Clean room, steady pace
> I'm running a benchmark across ten speech-to-text models tonight, comparing local Whisper variants against cloud APIs from Groq and ElevenLabs. The goal is to see how architecture beats hardware in real-world scenarios.

_Quiet room, even pace, phone close._

### en-02-fast — Fast, colloquial, with filler
> So like, the thing is, I've been running this bot on Groq for three weeks, right, and it's, you know, fine for English, but the moment I send a Russian voice, it kind of, eh, falls apart on rare words.

_Casual fast speech with natural fillers._

### en-03-codeswitch — Mixed accent and code-switching
> The pipeline в общем работает, but I want to add a fallback to local Whisper if the Groq API returns a five-hundred error. That way we don't lose transcriptions during outages.

_Code-switching EN/RU mid-sentence — stresses language auto-detect._

### en-04-digits — Numbers and technical strings
> The endpoint is api dot groq dot com slash openai slash v one slash audio slash transcriptions. Model ID is whisper dash large dash v three dash turbo. Set the timeout to six hundred seconds.

_Dotted technical strings, numbers as words._

### en-05-long — Long sentence with subordinate clauses
> If you look at how non-autoregressive transducers work, especially the ones based on TDT like Parakeet, you'll notice that latency stays roughly constant regardless of output length, which makes them ideal for streaming use cases.

_One long sentence with nested clauses._

### en-06-whisper — Quiet, near-whisper
> Late night recording, everyone's asleep, I'm testing how the model handles low-volume speech, so I'm keeping my voice very quiet on purpose to see what comes out.

_Near-whisper, mic close._

### en-07-names — Proper nouns and technical terms
> Stas Shupilkin, Telegram handle smolevich underscore voice underscore bot, repo name voice dash ai. Bench loads MLX Whisper, NVIDIA NeMo, Hugging Face transformers, and the Sber GigaAM model.

_Proper nouns, mixed-script project names, library names._

### en-08-street — Outdoor / noisy environment
> I'm walking outside right now to test how the bot handles street noise, wind, and the occasional car driving past. This is the kind of audio real users actually send.

_Real outdoor recording with ambient noise._

### en-09-short — Short, punchy commands
> Restart the bot. Stop redis first. Then the Telegram API container. Wait five seconds. Bring everything back up.

_Short imperatives, gaps between._

### en-10-podcast — Podcast-style monologue
> When you pick a speech recognition stack in twenty twenty-six, the first question isn't which model is the most accurate. It's whether you need real-time or batch, one language or many, and whether your data ever leaves the device.

_Calm monologue pace._

---

## После записи

1. На Mac mini открой `Voice Memos.app` (или Finder, если файлы у тебя в Finder-папке).
2. По возможности переименуй файлы в `ru-01-clean`, `ru-02-fast`, … `en-10-podcast` — так я сразу пойму что есть что. Если лень переименовывать — оставь как есть, я разберусь по таймштампам или по содержимому.
3. Скажи мне путь к папке (например `~/Library/Application Support/com.apple.voicememos/Recordings/` если включён iCloud sync, или любая твоя папка типа `~/voice-memos-bench/` куда экспортировал).
4. Я через rsync заберу всё в `experiments/live-samples/`, соберу `experiments/stt_samples.json` с reference text (текстами из этого файла + ручная коррекция если ты сильно отклонишься), и прогоню полный bench: mlx-whisper × 4 локально + Groq + ElevenLabs Scribe, на RU и EN.

PII в текстах нет — ничего личного, серверный IP здесь публичный из примера, токенов и ключей нет.
