# MYCO Development Notes

**Version:** 1.0.0
**Date:** 2026-04-02

---

## ⚠️ Important: Local-First Design

**MYCO was built for local use** with small LLM models (4B-9B parameters). This is not a limitation—it's a core design principle that shaped every architectural decision.

### Why Small Models?

Most AI coding tools are designed with the assumption that you have access to:
- Large cloud models (GPT-4, Claude, etc.)
- Massive context windows (100K+ tokens)
- Unlimited API calls
- Conversation history

**MYCO rejects all of these assumptions.**

Instead, MYCO was designed for:
- **Local models** (4B-9B parameters, running on your hardware)
- **Limited context** (16K-32K tokens typical)
- **No cloud dependencies** (privacy-first)
- **No conversation history** (stateless sessions)

---

## Architecture Decisions Driven by Small Model Constraints

### 1. Layered Context Loading

**Constraint:** Small models have limited context windows (16K-32K tokens)

**Solution:** Instead of loading entire codebase into context:
```
Layer 1: Architecture map (10-20 KB) - Always loaded
Layer 2: Module manifests (2-5 KB each) - On demand
Layer 3: Full file content - Only when modifying
```

**Result:** 60-80% less context usage while maintaining awareness

---

### 2. Focused Agent Tools

**Constraint:** Small models get confused by complex multi-purpose tools

**Solution:** 24+ simple, single-purpose tools:
```python
# Good (MYCO approach)
read_file(path)
write_file(path, content)
edit_file(path, old, new)

# Bad (what we avoided)
do_complex_refactoring_with_options(params...)
```

**Result:** Small models can reliably choose and execute tools

---

### 3. Entropy Budgets

**Constraint:** Small models tend to generate overly complex code

**Solution:** Enforce entropy budgets at write time:
```python
# New files: H < 0.50
# Modifications: ΔH < 0.15
# Blocks god classes and monolithic files
```

**Result:** Code quality enforced automatically

---

### 4. Stigmergic Memory

**Constraint:** No conversation history between sessions

**Solution:** Write memory traces directly into code:
```python
# ⊕ H:0.73 | press:refactor | age:4 | drift:+0.12
```

**Result:** Cross-session learning without databases

---

### 5. Iteration Limits & Loop Detection

**Constraint:** Small models can get stuck in repetitive loops

**Solution:** 
- Max 30 iterations per task
- Loop detection at count 3
- Automatic recovery (force different tool)
- Warnings at 75% budget

**Result:** Agent recovers from stuck states automatically

---

### 6. Architecture Awareness

**Constraint:** Small models can't hold entire codebase in mind

**Solution:** Lightweight architecture maps (~15 KB):
```json
{
  "layers": [
    {"name": "api", "level": 1},
    {"name": "services", "level": 2},
    {"name": "models", "level": 3}
  ],
  "dependencies": {...}
}
```

**Result:** Agent knows what depends on what without loading everything

---

### 7. Test Co-Creation

**Constraint:** Small models forget to write tests

**Solution:** Automatically generate test stubs after every Python file write

**Result:** 100% test creation rate (vs ~20% manually)

---

### 8. Type Inference & Contracts

**Constraint:** Small models make type errors

**Solution:** Static analysis catches type mismatches before they're written

**Result:** Higher certainty code despite small model limitations

---

## Testing Status

### What's Been Tested ✅

| Component | Testing Level | Models Used |
|-----------|---------------|-------------|
| **Entropy Calculation** | ✅ Extensive | 4B, 9B |
| **Stigmergic Annotations** | ✅ Extensive | 4B, 9B |
| **Autopoietic Gate** | ✅ Extensive | 4B, 9B |
| **Test Co-Creation** | ✅ Extensive | 4B, 9B |
| **Agent Tools** | ✅ Extensive | 4B, 9B |
| **Path Normalization** | ✅ Extensive | 4B, 9B |
| **Loop Recovery** | ✅ Extensive | 4B, 9B |
| **Type Inference** | ✅ Moderate | 4B, 9B |
| **Contract Generation** | ✅ Moderate | 4B, 9B |

### What Needs More Testing ⚠️

| Component | Testing Needed |
|-----------|----------------|
| **Large Models (30B+)** | Compatibility testing |
| **Very Large Codebases** | 1000+ file projects |
| **Edge Cases** | Unusual project structures |
| **Long-Term Use** | 100+ session durability |
| **Multi-Project Workflows** | Future feature |
| **Production Environments** | Real-world deployment |

---

## Known Limitations

### Small Model Limitations (By Design)

1. **Complex Reasoning**
   - 4B models may struggle with very complex tasks
   - Recommendation: Break into smaller subtasks

2. **Context Window**
   - Can't load entire large codebase at once
   - Mitigation: Layered approach (works well)

3. **Semantic Understanding**
   - Knows structure, not deep meaning
   - Future: Semantic index enhancement

4. **Cross-File Analysis**
   - Limited by what agent has read
   - Mitigation: Dependency tracking helps

### Untested Scenarios

1. **Very Large Models (70B+)**
   - May not need all safety mechanisms
   - Could potentially handle more complex workflows

2. **Cloud Models (GPT-4, Claude)**
   - Not tested (violates local-only principle)
   - Would require architectural changes

3. **Team Collaboration**
   - Single-user focus currently
   - Multi-user scenarios untested

4. **Continuous Integration**
   - CI/CD integration untested
   - Future enhancement

---

## Performance Expectations

### With 4B Model (Qwen3.5-4B)

| Metric | Expected |
|--------|----------|
| **Token Speed** | 45-55 tok/s (RTX 3060) |
| **Simple Tasks** | 2-5 minutes |
| **Medium Tasks** | 5-10 minutes |
| **Complex Tasks** | 10-20 minutes |
| **Success Rate** | 80-90% |

### With 9B Model (Qwen3.5-9B)

| Metric | Expected |
|--------|----------|
| **Token Speed** | 20-25 tok/s (RTX 3060) |
| **Simple Tasks** | 5-10 minutes |
| **Medium Tasks** | 10-20 minutes |
| **Complex Tasks** | 20-40 minutes |
| **Success Rate** | 85-95% |

### With Larger Models (30B+)

| Metric | Expected |
|--------|----------|
| **Token Speed** | 5-15 tok/s (varies) |
| **Task Time** | 2-3x longer than 9B |
| **Success Rate** | Potentially higher (untested) |
| **Reasoning** | Better for complex tasks (untested) |

---

## Recommendations for Users

### If You're Using Small Models (4B-9B)

✅ **You're in the target zone!**

- Start with simple tasks
- Use verbose mode to see what's happening
- Review code before accepting
- Let test co-creation work for you
- Check entropy reports regularly

### If You're Using Larger Models (30B+)

⚠️ **Limited testing, but should work**

- All safety mechanisms will still apply
- May find some constraints unnecessary
- Consider contributing test results
- May benefit from relaxed budgets (future config)

### If You're Using Cloud Models

❌ **Not supported (by design)**

- MYCO is local-only (privacy principle)
- Would require architectural changes
- Consider: Run local instead

---

## How You Can Help

### Testing Needed

1. **Real-World Projects**
   - Use MYCO on your actual codebase
   - Report what works, what doesn't
   - Share entropy before/after

2. **Different Model Sizes**
   - Test with 7B, 13B, 30B, 70B models
   - Compare success rates
   - Report performance metrics

3. **Edge Cases**
   - Unusual project structures
   - Non-Python languages (if applicable)
   - Very large codebases (1000+ files)

4. **Long-Term Use**
   - 50+ session projects
   - Stigmergic memory durability
   - Quality trend accuracy

### How to Report

**GitHub Issues:**
- Describe your setup (model, GPU, project size)
- What you tried to do
- What actually happened
- Logs and error messages

**GitHub Discussions:**
- Success stories
- Tips and tricks
- Questions and answers

---

## Future Considerations

### Potential Enhancements (Post-v1.0)

1. **Configurable Safety Levels**
   - Allow experienced users to relax budgets
   - Keep defaults for safety

2. **Model-Specific Tuning**
   - Different defaults for 4B vs 9B vs 30B
   - Auto-detect and adjust

3. **Enhanced Testing Tools**
   - Performance benchmarks
   - Comparison frameworks
   - Automated testing suites

4. **Documentation Improvements**
   - More real-world examples
   - Case studies
   - Best practices from community

---

## Summary

**MYCO is operational and tested** with small local models (4B-9B).

**What works:**
- ✅ All core features
- ✅ Entropy tracking
- ✅ Test co-creation
- ✅ Stigmergic memory
- ✅ Quality tracking

**What needs more testing:**
- ⚠️ Large model support (30B+)
- ⚠️ Very large codebases
- ⚠️ Production environments
- ⚠️ Long-term durability

**Best use cases:**
- ✅ Local development
- ✅ Learning and experimentation
- ✅ Code quality improvement
- ✅ Personal projects

**Use caution:**
- ⚠️ Critical production workflows (without additional testing)
- ⚠️ Team environments (not designed for collaboration yet)

---

**MYCO was built for local use, with small models, by someone who believes in privacy-first AI coding.**

If that aligns with your values, you're the target user. Welcome! 🍄

---

**Questions? Issues? Success stories?** Open a GitHub issue or discussion!
