# MYCO Examples

This folder contains example projects demonstrating MYCO capabilities.

## Examples

### simple_project/
A basic Python project to test MYCO fundamentals.

**What to try:**
```bash
cd examples/simple_project
python -m myco "Add type hints to all functions"
python -m myco "Add docstrings to all classes"
python -m myco report
```

### ecommerce_api/
A REST API example with models, services, and routes.

**What to try:**
```bash
cd examples/ecommerce_api
python -m myco "Create authentication service with JWT"
python -m myco "Add caching to database queries"
python -m myco "Add input validation to all endpoints"
```

## Getting Started

1. **Choose an example**
2. **Navigate to folder**
3. **Run MYCO task**
4. **Review changes**
5. **Check entropy report**

## Tips

- Start with `simple_project` if new to MYCO
- Use `python -m myco report` to see code health
- Check `.myco/world.json` for session history
- Review stigmergic annotations in modified files

## Contributing Examples

Want to contribute an example? See [CONTRIBUTING.md](../CONTRIBUTING.md)
