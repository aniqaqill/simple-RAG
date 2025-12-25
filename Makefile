v-up:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
	fi
	@if [ -f "requirements.txt" ]; then \
		echo "Installing requirements..."; \
		bash -c "source venv/bin/activate && pip install -r requirements.txt"; \
	fi
	@echo "Activating virtual environment..."
	@bash -c "source venv/bin/activate"

v-down:
	@echo "Deactivating virtual environment..."
	@echo "Type 'exit' or press Ctrl+D to leave the virtual environment shell"