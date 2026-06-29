#!/usr/bin/env bash
# Khởi động Streamlit + ngrok cùng lúc và in ra URL public.
# Dùng: ./run_share.sh   (Ctrl-C để dừng cả hai)
set -euo pipefail

PORT="${STREAMLIT_PORT:-8501}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- Kích hoạt venv nếu có ---
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# --- Kiểm tra Ollama đang chạy (bộ não cục bộ) ---
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "⚠️  Ollama chưa chạy ở localhost:11434 — hãy mở Ollama trước khi hỏi đáp."
fi

# --- Dọn dẹp tiến trình con khi thoát ---
PIDS=()
cleanup() {
  echo ""
  echo "🛑 Đang dừng..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# --- 1) Streamlit ---
echo "🚀 Khởi động Streamlit ở cổng $PORT..."
streamlit run apps/streamlit_app.py \
  --server.port "$PORT" \
  --server.headless true &
PIDS+=($!)

# Chờ Streamlit lắng nghe cổng
until curl -sf "http://localhost:$PORT" >/dev/null 2>&1; do
  sleep 0.5
done
echo "✅ Streamlit sẵn sàng ở http://localhost:$PORT"

# --- 2) ngrok ---
echo "🌐 Mở ngrok tunnel..."
ngrok http "$PORT" --log=stdout > /tmp/ngrok_streamlit.log 2>&1 &
PIDS+=($!)

# Chờ tunnel sẵn sàng rồi lấy URL public
until curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -q public_url; do
  sleep 0.5
done
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])")

echo ""
echo "==================================================================="
echo "  🔗 URL public:  $PUBLIC_URL"
echo "  📊 Dashboard:   http://localhost:4040"
echo "  (Ctrl-C để dừng cả Streamlit và ngrok)"
echo "==================================================================="

# Giữ script chạy cho tới khi người dùng Ctrl-C
wait
