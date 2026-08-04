# Deepgram runner

Cloud runner for [Deepgram](https://deepgram.com) pre-recorded transcription, added the same
way as the Groq / ElevenLabs / Fish Audio runners: an entry in `MODEL_REGISTRY` plus a
`deepgram` branch in `build_runner()` inside [`bench/stt_landscape_bench.py`](../bench/stt_landscape_bench.py).

## Getting a key

1. Sign up at <https://console.deepgram.com/signup>. **No credit card required.**
2. New accounts get **$200 of free credit**, and per Deepgram's pricing page it carries
   "no minimums, no expiration" — enough for this whole benchmark thousands of times over
   (a full 3-set sweep of all three model IDs costs a few cents).
3. Console → *API Keys* → *Create a New API Key*. Any scope that allows `usage:write`
   (the default "Member" role) is enough.

## Environment variable

`DEEPGRAM_API_KEY` — read from the process environment, exactly like `GROQ_API_KEY`,
`ELEVENLABS_API_KEY` and `FISH_AUDIO_API_KEY`. `build_runner()` fails fast with a clear
message when it is missing (`requires_env` on the `ModelSpec`).

Never commit the key. `.env` and `.env.*` are gitignored; the repo itself has no dotenv
loader, so pass the key inline or `export` it:

```bash
export DEEPGRAM_API_KEY=...
python bench/stt_landscape_bench.py \
    --samples samples/manifest.json \
    --out-dir results/ \
    --models deepgram-nova-3,deepgram-nova-3-multi,deepgram-nova-2 \
    --warmup 1 --runs 3 --cloud-sleep-s 0.5
```

## Registered models

| Model id | API `model` | `language` sent | Why it's here |
|---|---|---|---|
| `deepgram-nova-3` | `nova-3` | per-sample code from the manifest (`ru` / `en`) | Deepgram's current flagship. |
| `deepgram-nova-3-multi` | `nova-3` | always `multi` | Deepgram's docs point non-English users at the multilingual mode; this row isolates whether it actually helps. |
| `deepgram-nova-2` | `nova-2` | per-sample code from the manifest | Previous generation, same vendor — the "did upgrading matter" control, mirroring `scribe_v1` vs `scribe_v1_experimental`. |

The fixed-language behaviour comes from the optional `language_override` field on
`ModelSpec` (empty for every other model, so nothing else changes).

## API specifics

- **Endpoint**: `POST https://api.deepgram.com/v1/listen`.
- **Auth header**: `Authorization: Token <key>` — `Token`, not `Bearer`.
- **Body is raw audio bytes, not multipart.** This is the one structural difference from
  Groq / ElevenLabs / Fish Audio, which all post a `files={...}` form. The runner sends
  `content=<bytes>` with `Content-Type` from `mimetypes.guess_type()` (`audio/x-m4a` for the
  live `.m4a` set, `audio/mpeg` for the ElevenLabs `.mp3` set, `audio/x-wav` for the Silero
  `.wav` set). Deepgram sniffs the container itself, so the header is advisory.
- **Every knob is a query parameter**: `model`, `language`, `smart_format=true`,
  `punctuate=true`. `diarize=true` is available but not used — the bench samples are
  single-speaker and diarization would only add latency.
- **Transcript path**: `response["results"]["channels"][0]["alternatives"][0]["transcript"]`.
  The runner returns `""` when either list is empty instead of raising.
- **`smart_format` is not free, and not what it looks like.** Measured A/B on the live set,
  3 runs, clean subset:

  | Model | EN clean, `smart_format=on` | off | delta |
  |---|---|---|---|
  | `deepgram-nova-3-multi` | 17.91% | **14.58%** | −3.33 pp |
  | `deepgram-nova-2` | 18.51% | 15.69% | −2.81 pp |
  | `deepgram-nova-3` (`language=en`) | 20.87% | 20.66% | −0.21 pp |

  On **Russian the flag makes no difference at all** — on `ru-04-digits` the transcript is
  byte-identical with it on and off, at 80.0% WER either way for Nova-3. Deepgram renders
  Russian numbers as digits regardless of the flag, so this is a property of the model, not
  of `smart_format`, and the `*-04-digits` / `*-07-names` exclusion already absorbs it.

  The English penalty is a different story: it is ~3 points and it lands on the *clean*
  subset, so it is not filtered out. The runner keeps `smart_format=true` because it is the
  vendor's documented production posture, but that choice costs Deepgram a place in the
  English ranking — with it off, `deepgram-nova-3-multi` at 14.58% moves ahead of Fish Audio
  (14.96%). Flip it if your downstream consumer does its own formatting.
  Only the `smart_format=on` sweep is committed to `results/`.
- Rate limits are generous (50 concurrent REST requests on pay-as-you-go), so
  `--cloud-sleep-s 0.5` is plenty — no need for the 4 s Groq free-tier pause.

## Pricing (Aug 2026, pay-as-you-go)

| Model | $/min | $/audio hour |
|---|---|---|
| Nova-3 monolingual | $0.0077 | ~$0.46 |
| Nova-3 multilingual | $0.0092 | ~$0.55 |
| Nova-2 | $0.0043 | ~$0.26 |

For reference: Fish Audio `transcribe-1` is $0.36/audio hour, Groq Whisper Large v3 Turbo
is ~$0.04/audio hour.
