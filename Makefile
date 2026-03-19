# QBOT Makefile — one command for everything
.PHONY: install run dev ui test clean reset deploy-railway deploy-render

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	mkdir -p data/cache dashboard
	@echo "\n  ✓ QBOT installed. Run: make run\n"

# ── Local run ─────────────────────────────────────────────────────────────
run:
	@echo "\n  Starting QBOT on http://localhost:5000 ...\n"
	python api_server.py

dev:
	DEBUG=true python api_server.py

# ── Backtest (CLI) ────────────────────────────────────────────────────────
backtest:
	python main.py --mode backtest --symbols AAPL MSFT BTC-USD --period 2y

optimize:
	python main.py --mode optimize --symbols AAPL --period 5y

paper:
	python main.py --mode paper --symbols AAPL BTC-USD --poll 60

# ── Tests ─────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -f paper_state.json

reset:
	rm -f paper_state.json settings.json
	rm -rf data/cache/*.parquet
	@echo "  ✓ State reset"

# ── Deploy ────────────────────────────────────────────────────────────────
deploy-railway:
	@echo "\n  Deploy to Railway:"
	@echo "  1. railway login"
	@echo "  2. railway init"
	@echo "  3. railway up\n"
	railway up 2>/dev/null || echo "  Run: npm install -g @railway/cli  then  railway login"

deploy-render:
	@echo "\n  Deploy to Render:"
	@echo "  1. Push this folder to GitHub"
	@echo "  2. Go to https://render.com → New Web Service"
	@echo "  3. Connect your repo — render.yaml handles the rest\n"

# ── Docker ────────────────────────────────────────────────────────────────
docker-build:
	docker build -t qbot .

docker-run:
	docker run -p 5000:5000 -e PORT=5000 qbot
