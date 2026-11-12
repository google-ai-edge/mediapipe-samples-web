# Release validation fixtures

Inputs with a known ground truth, fed to the sample app through its UI by `runner.mjs`.

| File | Source | SHA-256 | Ground truth |
|---|---|---|---|
| `speech_16000_hz_mono.wav` | MediaPipe test asset, `https://storage.googleapis.com/mediapipe-assets/speech_16000_hz_mono.wav` (listed in `mediapipe/tasks/testdata/audio/BUILD`) | `71caf50b8757d6ab9cad5eae4d36669d3c20c225a51660afd7fe0dc44cdb74f6` | 4.27 s of speech, 16 kHz mono. MediaPipe's own `audio_classifier_test.py` expects YAMNet to label it `Speech`. There is no published transcript; `SPEECH_TRANSCRIPT` in `expectations.mjs` holds what the Audio Transcriber of release 1.2.2 heard ("Send a text to Mike, I'll be home late tomorrow."), which a person should confirm once by listening. |
