# MYCO Examples

This folder contains example projects demonstrating MYCO capabilities.

## Getting Started

1. **Start MYCO** from the project root:
   ```bash
   python -m cli.main
   ```
2. **Navigate to an example folder** in the interactive prompt:
   ```
   You> cd examples/simple_project
   ```
3. **Give MYCO a task** — just type it in the prompt

## Examples

### simple_project/
A basic Python project to test MYCO fundamentals.

**What to try:**
```
Add type hints to all functions
Add docstrings to all classes
```

### ecommerce_api/
A REST API example with models, services, and routes.

**What to try:**
```
Create authentication service with JWT
Add caching to database queries
Add input validation to all endpoints
```

## Tips

- Start with `simple_project` if new to MYCO
- Ask MYCO to "check entropy" or "substrate health" to see code health
- Check `.myco/world.json` for session history
- Review stigmergic annotations in modified files

## Contributing Examples

Want to contribute an example? See [CONTRIBUTING.md](../CONTRIBUTING.md)
