install:
	uv sync

run:
	uv run src/ $(FUNCTION_DEF) $(INPUT) $(OUTPUT)

debug:
	uv run -m pdb -m src/ $(FUNCTION_DEF) $(INPUT) $(OUTPUT)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache graph_pac/.mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint