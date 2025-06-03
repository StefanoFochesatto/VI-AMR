.PHONY: clean

clean:
	@rm -rf __pycache__/ .pytest_cache/ *.egg-info/
	@rm -rf tests/__pycache__/ viamr/__pycache__/
	@rm -rf .coverage htmlcov/
