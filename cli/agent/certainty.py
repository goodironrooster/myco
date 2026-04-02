# ⊕ H:0.25 | press:certainty | age:0 | drift:+0.00
"""MYCO Certainty - High-certainty code generation.

MYCO Vision:
- Type inference (catch type errors)
- Contract generation (pre/post conditions)
- Property testing (find edge cases)
- Integration verification (verify connections)

Architecture:
- Type inferencer (static analysis)
- Contract generator (DBC - Design by Contract)
- Property tester (Hypothesis-style)
- Integration verifier (connection checks)
"""

import ast
import inspect
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from pathlib import Path


# ============================================================================
# Type Inference
# ============================================================================

@dataclass
class TypeAnnotation:
    """Type annotation for a variable or function."""
    name: str
    inferred_type: str
    confidence: float  # 0.0 to 1.0
    source: str  # "annotation", "inference", "default"
    line_number: int = 0


@dataclass
class TypeInferenceResult:
    """Result of type inference for a file."""
    file_path: str
    function_types: Dict[str, List[TypeAnnotation]] = field(default_factory=dict)
    variable_types: Dict[str, TypeAnnotation] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TypeInferencer:
    """Infer types from Python code.
    
    MYCO Phase 3.1: Catch type errors before runtime.
    """
    
    # Common type mappings
    TYPE_DEFAULTS = {
        "int": 0,
        "float": 0.0,
        "str": "",
        "bool": False,
        "list": [],
        "dict": {},
        "set": set(),
        "tuple": (),
        "None": None
    }
    
    def infer_types(self, source_code: str, file_path: str = "") -> TypeInferenceResult:
        """Infer types from source code.
        
        Args:
            source_code: Python source code
            file_path: Optional file path
            
        Returns:
            TypeInferenceResult with inferred types
        """
        result = TypeInferenceResult(file_path=file_path)
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            result.errors.append(f"Syntax error: {e}")
            return result
        
        # Infer function types
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_types = self._infer_function_types(node)
                result.function_types[node.name] = func_types
                
                # Check for type inconsistencies
                inconsistencies = self._check_function_type_consistency(node, func_types)
                result.warnings.extend(inconsistencies)
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_type = self._infer_expression_type(node.value)
                        result.variable_types[target.id] = TypeAnnotation(
                            name=target.id,
                            inferred_type=var_type,
                            confidence=0.7,  # Inference confidence
                            source="inference",
                            line_number=node.lineno
                        )
        
        return result
    
    def _infer_function_types(self, func: ast.FunctionDef) -> List[TypeAnnotation]:
        """Infer types for function parameters and return."""
        annotations = []
        
        # Infer parameter types
        for arg in func.args.args:
            if arg.annotation:
                # Has explicit annotation
                type_str = self._annotation_to_string(arg.annotation)
                annotations.append(TypeAnnotation(
                    name=arg.arg,
                    inferred_type=type_str,
                    confidence=1.0,
                    source="annotation",
                    line_number=func.lineno
                ))
            else:
                # Infer from default value
                default_idx = len(func.args.args) - len(func.args.defaults)
                if default_idx >= 0 and default_idx < len(func.args.defaults):
                    default = func.args.defaults[default_idx]
                    inferred_type = self._infer_expression_type(default)
                    annotations.append(TypeAnnotation(
                        name=arg.arg,
                        inferred_type=inferred_type,
                        confidence=0.6,
                        source="default",
                        line_number=func.lineno
                    ))
                else:
                    # No annotation, no default
                    annotations.append(TypeAnnotation(
                        name=arg.arg,
                        inferred_type="Any",
                        confidence=0.3,
                        source="inference",
                        line_number=func.lineno
                    ))
        
        # Infer return type
        if func.returns:
            return_type = self._annotation_to_string(func.returns)
            annotations.append(TypeAnnotation(
                name="return",
                inferred_type=return_type,
                confidence=1.0,
                source="annotation",
                line_number=func.lineno
            ))
        
        return annotations
    
    def _infer_expression_type(self, expr: ast.expr) -> str:
        """Infer type from expression."""
        if isinstance(expr, ast.Constant):
            return type(expr.value).__name__
        
        elif isinstance(expr, ast.List):
            return "list"
        
        elif isinstance(expr, ast.Dict):
            return "dict"
        
        elif isinstance(expr, ast.Set):
            return "set"
        
        elif isinstance(expr, ast.Tuple):
            return "tuple"
        
        elif isinstance(expr, ast.Call):
            # Try to infer from function name
            if isinstance(expr.func, ast.Name):
                func_name = expr.func.id
                if func_name in ["int", "float", "str", "bool", "list", "dict", "set"]:
                    return func_name
            
            return "Any"
        
        elif isinstance(expr, ast.BinOp):
            # Binary operations preserve type
            left_type = self._infer_expression_type(expr.left)
            right_type = self._infer_expression_type(expr.right)
            
            # If both same type, result is that type
            if left_type == right_type:
                return left_type
            
            # Numeric promotion
            if left_type in ["int", "float"] and right_type in ["int", "float"]:
                return "float" if "float" in [left_type, right_type] else "int"
            
            return "Any"
        
        elif isinstance(expr, ast.Name):
            # Variable reference - unknown type
            return "Any"
        
        else:
            return "Any"
    
    def _annotation_to_string(self, annotation: ast.expr) -> str:
        """Convert AST annotation to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        
        elif isinstance(annotation, ast.Subscript):
            # e.g., List[int], Dict[str, Any]
            if isinstance(annotation.value, ast.Name):
                container = annotation.value.id
                if isinstance(annotation.slice, ast.Tuple):
                    # Multiple type parameters
                    params = ", ".join(
                        self._annotation_to_string(elt)
                        for elt in annotation.slice.elts
                    )
                    return f"{container}[{params}]"
                else:
                    # Single type parameter
                    param = self._annotation_to_string(annotation.slice)
                    return f"{container}[{param}]"
        
        elif isinstance(annotation, ast.Attribute):
            # e.g., typing.List
            return f"{self._annotation_to_string(annotation.value)}.{annotation.attr}"
        
        return "Any"
    
    def _check_function_type_consistency(
        self,
        func: ast.FunctionDef,
        annotations: List[TypeAnnotation]
    ) -> List[str]:
        """Check for type inconsistencies in function."""
        warnings = []
        
        # Check if return type matches actual returns
        if annotations and annotations[-1].name == "return":
            return_type = annotations[-1].inferred_type
            
            # Find all return statements
            for node in ast.walk(func):
                if isinstance(node, ast.Return) and node.value:
                    actual_type = self._infer_expression_type(node.value)
                    
                    if return_type != "Any" and actual_type != "Any":
                        if return_type != actual_type:
                            warnings.append(
                                f"Function '{func.name}': Return type {return_type} "
                                f"doesn't match actual return {actual_type} "
                                f"(line {node.lineno})"
                            )
        
        return warnings


# ============================================================================
# Contract Generation
# ============================================================================

@dataclass
class Contract:
    """Design by Contract for a function."""
    function_name: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "function_name": self.function_name,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "invariants": self.invariants
        }


class ContractGenerator:
    """Generate contracts (pre/post conditions) for functions.
    
    MYCO Phase 3.2: Design by Contract for AI-generated code.
    """
    
    def generate_contract(self, source_code: str, function_name: str) -> Optional[Contract]:
        """Generate contract for a function.
        
        Args:
            source_code: Python source code
            function_name: Name of function
            
        Returns:
            Contract or None
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return None
        
        # Find function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return self._generate_function_contract(node)
        
        return None
    
    def _generate_function_contract(self, func: ast.FunctionDef) -> Contract:
        """Generate contract for a function."""
        contract = Contract(function_name=func.name)
        
        # Generate preconditions from parameters
        for arg in func.args.args:
            if arg.annotation:
                type_str = self._annotation_to_string(arg.annotation)
                
                # Add type-based preconditions
                if type_str == "int":
                    contract.preconditions.append(f"{arg.arg} is not None")
                elif type_str == "str":
                    contract.preconditions.append(f"{arg.arg} is not None")
                    contract.preconditions.append(f"len({arg.arg}) > 0")
                elif type_str.startswith("List["):
                    contract.preconditions.append(f"{arg.arg} is not None")
                elif type_str.startswith("Dict["):
                    contract.preconditions.append(f"{arg.arg} is not None")
        
        # Generate postconditions from return type
        if func.returns:
            return_type = self._annotation_to_string(func.returns)
            
            if return_type != "None" and return_type != "Any":
                contract.postconditions.append(f"result is not None")
                
                if return_type == "bool":
                    contract.postconditions.append(f"isinstance(result, bool)")
                elif return_type == "int":
                    contract.postconditions.append(f"isinstance(result, int)")
                elif return_type == "str":
                    contract.postconditions.append(f"isinstance(result, str)")
                elif return_type.startswith("List["):
                    contract.postconditions.append(f"isinstance(result, list)")
                elif return_type.startswith("Dict["):
                    contract.postconditions.append(f"isinstance(result, dict)")
        
        # Generate invariants from class context (if method)
        # (Simplified - full implementation would check class context)
        
        return contract
    
    def _annotation_to_string(self, annotation: ast.expr) -> str:
        """Convert AST annotation to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                container = annotation.value.id
                if isinstance(annotation.slice, ast.Tuple):
                    params = ", ".join(
                        self._annotation_to_string(elt)
                        for elt in annotation.slice.elts
                    )
                    return f"{container}[{params}]"
                else:
                    param = self._annotation_to_string(annotation.slice)
                    return f"{container}[{param}]"
        return "Any"
    
    def generate_decorator_code(self, contract: Contract) -> str:
        """Generate Python decorator code from contract.
        
        Args:
            contract: Contract
            
        Returns:
            Python decorator code
        """
        lines = [
            "def contract_check(func):",
            "    def wrapper(*args, **kwargs):",
            "        import inspect",
            "        sig = inspect.signature(func)",
            "        bound = sig.bind(*args, **kwargs)",
            "        bound.apply_defaults()",
            "",
            "        # Preconditions",
        ]
        
        for precond in contract.preconditions:
            lines.append(f"        assert {precond}, 'Precondition failed: {precond}'")
        
        lines.extend([
            "",
            "        # Call function",
            "        result = func(*bound.args, **bound.kwargs)",
            "",
            "        # Postconditions",
        ])
        
        for postcond in contract.postconditions:
            lines.append(f"        assert {postcond}, 'Postcondition failed: {postcond}'")
        
        lines.extend([
            "",
            "        return result",
            "    return wrapper",
            ""
        ])
        
        return "\n".join(lines)


# ============================================================================
# Property Testing
# ============================================================================

@dataclass
class Property:
    """A property to test."""
    name: str
    description: str
    test_code: str
    property_type: str  # "always", "never", "sometimes"


class PropertyTester:
    """Generate and run property tests.
    
    MYCO Phase 3.3: Find edge cases automatically.
    """
    
    def generate_properties(self, source_code: str) -> List[Property]:
        """Generate properties to test.
        
        Args:
            source_code: Python source code
            
        Returns:
            List of properties
        """
        properties = []
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return properties
        
        # Generate properties for each function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_properties = self._generate_function_properties(node)
                properties.extend(func_properties)
        
        return properties
    
    def _generate_function_properties(self, func: ast.FunctionDef) -> List[Property]:
        """Generate properties for a function."""
        properties = []
        
        # Property 1: No exceptions for valid input
        properties.append(Property(
            name=f"{func.name}_no_exception",
            description=f"{func.name} should not raise exceptions for valid input",
            test_code=f"""
def test_{func.name}_no_exception():
    # TODO: Add valid test inputs
    try:
        {func.name}(...)  # Add arguments
    except Exception as e:
        assert False, f"Unexpected exception: {{e}}"
""",
            property_type="always"
        ))
        
        # Property 2: Deterministic (same input → same output)
        properties.append(Property(
            name=f"{func.name}_deterministic",
            description=f"{func.name} should be deterministic",
            test_code=f"""
def test_{func.name}_deterministic():
    # TODO: Add test inputs
    result1 = {func.name}(...)
    result2 = {func.name}(...)
    assert result1 == result2, "Function is not deterministic"
""",
            property_type="always"
        ))
        
        # Property 3: Type consistency
        properties.append(Property(
            name=f"{func.name}_type_consistent",
            description=f"{func.name} should return consistent types",
            test_code=f"""
def test_{func.name}_type_consistent():
    # TODO: Add multiple test inputs
    results = [
        {func.name}(...),  # Input 1
        {func.name}(...),  # Input 2
    ]
    types = set(type(r).__name__ for r in results)
    assert len(types) == 1, f"Inconsistent return types: {{types}}"
""",
            property_type="always"
        ))
        
        # Property 4: Pure function (if no side effects detected)
        if self._appears_pure(func):
            properties.append(Property(
                name=f"{func.name}_pure",
                description=f"{func.name} appears to be pure (no side effects)",
                test_code=f"""
def test_{func.name}_pure():
    # TODO: Add test inputs
    import copy
    state_before = ...  # Capture state
    result = {func.name}(...)
    state_after = ...  # Capture state
    assert state_before == state_after, "Function has side effects"
""",
                property_type="always"
            ))
        
        return properties
    
    def _appears_pure(self, func: ast.FunctionDef) -> bool:
        """Check if function appears pure (no side effects)."""
        # Look for side effects:
        # - Global variable access
        # - I/O operations
        # - Attribute assignment
        
        for node in ast.walk(func):
            # Check for global access
            if isinstance(node, ast.Global):
                return False
            
            # Check for I/O
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["print", "input", "open", "write"]:
                        return False
            
            # Check for attribute assignment
            if isinstance(node, ast.Attribute):
                if isinstance(node.ctx, ast.Store):
                    return False
        
        return True
    
    def generate_hypothesis_tests(self, properties: List[Property]) -> str:
        """Generate Hypothesis property tests.
        
        Args:
            properties: List of properties
            
        Returns:
            Hypothesis test code
        """
        lines = [
            "from hypothesis import given, strategies as st",
            "",
        ]
        
        for prop in properties:
            lines.append(prop.test_code)
            lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# Integration Verification
# ============================================================================

@dataclass
class VerificationResult:
    """Result of integration verification."""
    file1: str
    file2: str
    verified: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class IntegrationVerifier:
    """Verify integrations between components.
    
    MYCO Phase 3.4: Verify connections work.
    """
    
    def verify_integration(self, file1_path: str, file2_path: str) -> VerificationResult:
        """Verify integration between two files.
        
        Args:
            file1_path: Path to first file
            file2_path: Path to second file
            
        Returns:
            VerificationResult
        """
        result = VerificationResult(
            file1=file1_path,
            file2=file2_path,
            verified=True
        )
        
        try:
            # Read files
            source1 = Path(file1_path).read_text(encoding='utf-8')
            source2 = Path(file2_path).read_text(encoding='utf-8')
            
            # Parse ASTs
            tree1 = ast.parse(source1)
            tree2 = ast.parse(source2)
            
            # Check imports
            imports = self._extract_imports(tree1)
            exports = self._extract_exports(tree2)
            file2_module = Path(file2_path).stem
            
            # Verify imports match exports
            for module, names in imports:
                if module and (module == file2_module or module.endswith(f".{file2_module}")):
                    for name in names:
                        if name not in exports:
                            result.issues.append(
                                f"{Path(file1_path).name} imports '{name}' from {Path(file2_path).name}, "
                                f"but it's not exported"
                            )
                            result.verified = False
            
            # Check type compatibility
            type_issues = self._check_type_compatibility(tree1, tree2)
            result.issues.extend(type_issues)
            
            if type_issues:
                result.verified = False
            
            # Generate suggestions
            if not result.verified:
                result.suggestions = self._generate_fix_suggestions(result.issues)
            
        except Exception as e:
            result.issues.append(f"Verification error: {e}")
            result.verified = False
        
        return result
    
    def _extract_imports(self, tree: ast.AST) -> List[Tuple[str, List[str]]]:
        """Extract imports from AST.
        
        Returns:
            List of (module, [names]) tuples
        """
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
                imports.append((node.module, names))
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                imports.append((None, names))
        return imports
    
    def _extract_exports(self, tree: ast.AST) -> Set[str]:
        """Extract exports (public names) from AST."""
        exports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if not target.id.startswith('_'):
                                exports.add(target.id)
                else:
                    if not node.name.startswith('_'):
                        exports.add(node.name)
        
        return exports
    
    def _check_type_compatibility(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> List[str]:
        """Check type compatibility between two files."""
        issues = []
        
        # Simplified type checking
        # Full implementation would use type inferencer
        
        return issues
    
    def _generate_fix_suggestions(self, issues: List[str]) -> List[str]:
        """Generate suggestions to fix issues."""
        suggestions = []
        
        for issue in issues:
            if "not exported" in issue:
                suggestions.append(
                    "Add the missing export to __all__ or remove the underscore prefix"
                )
            elif "type" in issue.lower():
                suggestions.append(
                    "Check type annotations and ensure compatibility"
                )
        
        return suggestions


# ============================================================================
# Agent Tools
# ============================================================================

def infer_types(project_root: str, file_path: str) -> dict:
    """Infer types for file (agent tool)."""
    try:
        inferencer = TypeInferencer()
        source = Path(file_path).read_text(encoding='utf-8')
        result = inferencer.infer_types(source, file_path)
        
        return {
            "file": file_path,
            "function_types": {
                name: [a.to_dict() for a in anns]
                for name, anns in result.function_types.items()
            },
            "errors": result.errors,
            "warnings": result.warnings
        }
    except Exception as e:
        return {"error": str(e)}


def generate_contract(project_root: str, file_path: str, function_name: str) -> Optional[dict]:
    """Generate contract for function (agent tool)."""
    try:
        generator = ContractGenerator()
        source = Path(file_path).read_text(encoding='utf-8')
        contract = generator.generate_contract(source, function_name)
        
        if contract:
            return contract.to_dict()
        return None
    except Exception:
        return None


def generate_property_tests(project_root: str, file_path: str) -> list:
    """Generate property tests for file (agent tool)."""
    try:
        tester = PropertyTester()
        source = Path(file_path).read_text(encoding='utf-8')
        properties = tester.generate_properties(source)
        
        return [p.__dict__ for p in properties]
    except Exception:
        return []


def verify_integration(project_root: str, file1: str, file2: str) -> dict:
    """Verify integration between files (agent tool)."""
    try:
        verifier = IntegrationVerifier()
        result = verifier.verify_integration(file1, file2)
        
        return {
            "file1": file1,
            "file2": file2,
            "verified": result.verified,
            "issues": result.issues,
            "suggestions": result.suggestions
        }
    except Exception as e:
        return {"error": str(e), "verified": False}
