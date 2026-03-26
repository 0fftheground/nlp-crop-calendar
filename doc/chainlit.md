# Agronomy Assistant

This UI connects to the FastAPI backend and routes your questions to tools or workflows.

You can ask about:
- Crop variety traits
- Weather and time ranges for a specific region
- Growth stage prediction
- Full planting plans

Note: growth-stage prediction and crop calendar workflows currently use historical weather only; future sowing dates will trigger a prompt to provide a valid historical date.

If information is incomplete, the assistant will ask follow-up questions. To switch to a new question, just ask a new request. To clear memory, simply say so.

Tip: to change the backend address, set `BACKEND_URL`.

## Chat history & authentication
Chat history is enabled via Chainlit data persistence (`chainlit.toml`). Authentication is enabled using the password callback in `chainlit_app.py`. The username is used as the stable identity (`user_id`) for backend memory, and the Chainlit thread id is reused as `session_id` so a resumed chat can continue pending follow-ups.

Authentication modes:
- If you set credentials in `.env`, only those users can log in:
  - `CHAINLIT_AUTH_USERS` as `user:pass,user2:pass2`
  - or `CHAINLIT_AUTH_USERNAME` / `CHAINLIT_AUTH_PASSWORD` for a single user
- If no credentials are configured, any non-empty username/password will be accepted (open mode).
Authentication requires a JWT secret. Set `CHAINLIT_AUTH_SECRET` (or run `chainlit create-secret` to generate one) before starting Chainlit.

Once configured, your chat threads will persist across page reloads for the same login.
