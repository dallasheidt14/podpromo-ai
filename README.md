# PodPromo AI (Clip-only MVP)

## Quickstart
1) Install: Python 3.10+, Node 18+, ffmpeg
2) Backend
   ```bash
   cd backend
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3) Frontend
   ```bash
   cd ../frontend
   export NEXT_PUBLIC_API_URL=http://localhost:8000
   npm install
   npm run dev
   ```

4) Open http://localhost:3000
   → Upload → Find Candidates → Nudge → Render.

## Secret Sauce

Scoring = Hook (0.35) + Prosody (0.20) + Emotion (0.15) + Q/List (0.10) + Payoff (0.10) + Info (0.05) + Loop (0.05)

Edit `backend/config/secret_config.json` or load a preset via `POST /config/load-preset`.

### Hot-Swappable Configuration
- **Live Reload**: `POST /config/reload` to refresh config without restart
- **Genre Presets**: Business, Comedy presets available
- **Runtime Tuning**: Adjust weights and lexicons on-the-fly

## Tests
```bash
cd backend
pytest -q
```

## Notes

- Replace Whisper stub with faster-whisper for real timestamps
- Clips served at `GET /files/clip_*.mp4`
- Configuration hot-reloadable via API endpoints
- Genre-specific presets for different content types
