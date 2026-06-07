# Deploy: Hugging Face Space (backend) + GitHub Pages (frontend)

Your GitHub Pages site is static, so the PyTorch model runs on a Hugging Face Space
and your Pages site calls it over HTTP.

```
GitHub Pages (static)                 Hugging Face Space (Docker)
  chess.html  ──POST /move {fen}──►   server.py + model  ──► { move }
```

---

## Part 1 — Deploy the backend to a Hugging Face Space

1. **Create the Space.** Sign in at <https://huggingface.co> → **New → Space**.
   - Name: `alan-aumaton-chess` (anything is fine)
   - **SDK: Docker** → template **Blank**
   - Visibility: Public → **Create Space**

2. **Clone the (empty) Space repo** locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-hf-username>/alan-aumaton-chess
   cd alan-aumaton-chess
   ```
   (If prompted to authenticate, run `huggingface-cli login` or use an access token
   from huggingface.co/settings/tokens as the git password.)

3. **Copy these files** from this project into the cloned Space folder, keeping paths:
   - `Dockerfile`
   - `server.py`, `engine.py`, `model.py`, `encoding.py`
   - `models/policy_alanzhang.pt`  ← the trained model (keep it under `models/`)
   - `deploy/SPACE_README.md` → save it as **`README.md`** in the Space root
     (the YAML header tells HF it's a Docker app on port 7860)

4. **Commit & push** (the 38 MB model goes through Git LFS):
   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes Dockerfile server.py engine.py model.py encoding.py README.md models/policy_alanzhang.pt
   git commit -m "Deploy alan-aumaton move API"
   git push
   ```

5. **Wait for the build.** On the Space page, the **Logs** tab shows the Docker build,
   then "Running". Your API is now at:
   ```
   https://<your-hf-username>-alan-aumaton-chess.hf.space
   ```
   Test it: open `…/health` in a browser → you should see
   `{"status":"ok","username":"AlanZhang","model_loaded":false}`.

6. *(Optional, recommended)* **Lock down CORS.** Space → **Settings → Variables and
   secrets → New variable**: `BOT_ORIGINS = https://<your-github-username>.github.io`.
   (Default allows all origins, which also works.)

---

## Part 2 — Add the page to your existing GitHub Pages site

1. Copy **both** files into your Pages repo (keep them in the same folder — e.g. repo
   root, or a `chess/` subfolder):
   - `web/github-pages/chess.html`
   - `web/github-pages/chess-0.10.3.min.js`

2. Edit `chess.html` — set the API URL near the top to your Space:
   ```js
   const API_BASE = "https://<your-hf-username>-alan-aumaton-chess.hf.space";
   ```

3. Link it from your site’s navigation, e.g.:
   ```html
   <a href="chess.html">Play my chess clone</a>
   ```

4. Commit & push. It’ll be live at
   `https://<your-github-username>.github.io/<repo>/chess.html`
   (or under your custom domain).

---

## Notes

- **Cold starts:** free Spaces sleep after inactivity; the first move wakes the server
  (~10–30 s). The page shows a "waking up" message — later moves are instant.
- **HTTPS:** both GitHub Pages and HF Spaces are HTTPS, so there’s no mixed-content block.
- **Updating the model:** retrain locally, then repeat Part 1 step 4 (commit the new
  `.pt`, push). The Space rebuilds and serves the new weights automatically.
