"""
Simple verification script for Phase 1 LRP implementation.
Checks that all modules can be imported and basic structure is correct.
"""

import sys
import os

# Add pyhealth to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("LRP PHASE 1 - VERIFICATION SCRIPT")
print("=" * 80)

# Test 1: Import base classes
print("\n[1/6] Testing base class imports...")
try:
    from pyhealth.interpret.methods.lrp_base import (
        LRPLayerHandler,
        LRPHandlerRegistry,
        ConservationValidator,
        stabilize_denominator,
        check_tensor_validity
    )
    print("✓ Successfully imported lrp_base classes")
except ImportError as e:
    print(f"✗ Failed to import lrp_base: {e}")
    sys.exit(1)

# Test 2: Import handlers
print("\n[2/6] Testing handler imports...")
try:
    from pyhealth.interpret.methods.lrp_handlers import (
        LinearLRPHandler,
        ReLULRPHandler,
        EmbeddingLRPHandler,
        create_default_registry
    )
    print("✓ Successfully imported lrp_handlers classes")
except ImportError as e:
    print(f"✗ Failed to import lrp_handlers: {e}")
    sys.exit(1)

# Test 3: Import unified LRP
print("\n[3/6] Testing UnifiedLRP import...")
try:
    from pyhealth.interpret.methods.lrp_unified import UnifiedLRP
    print("✓ Successfully imported UnifiedLRP")
except ImportError as e:
    print(f"✗ Failed to import lrp_unified: {e}")
    sys.exit(1)

# Test 4: Check package exports
print("\n[4/6] Testing package-level exports...")
try:
    from pyhealth.interpret.methods import (
        UnifiedLRP,
        LRPHandlerRegistry,
        ConservationValidator,
        LinearLRPHandler,
        create_default_registry
    )
    print("✓ All exports available from package")
except ImportError as e:
    print(f"✗ Failed to import from package: {e}")
    sys.exit(1)

# Test 5: Create registry
print("\n[5/6] Testing registry creation...")
try:
    registry = create_default_registry()
    handlers = registry.list_handlers()
    print(f"✓ Created registry with {len(handlers)} handlers:")
    for handler_name in handlers:
        print(f"    - {handler_name}")
except Exception as e:
    print(f"✗ Failed to create registry: {e}")
    sys.exit(1)

# Test 6: Create validator
print("\n[6/6] Testing validator creation...")
try:
    validator = ConservationValidator(tolerance=0.01)
    print(f"✓ Created validator with tolerance={validator.tolerance}")
except Exception as e:
    print(f"✗ Failed to create validator: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\n✓ All Phase 1 components successfully imported and instantiated!")
print("\nImplemented components:")
print("  • LRPLayerHandler abstract base class")
print("  • LRPHandlerRegistry with automatic layer detection")
print("  • ConservationValidator for property checking")
print("  • LinearLRPHandler (epsilon + alpha-beta rules)")
print("  • ReLULRPHandler")
print("  • EmbeddingLRPHandler")
print("  • UnifiedLRP main class")
print("\nTo run full tests (requires PyTorch):")
print("  python3 -m pytest tests/test_lrp_phase1.py -v")
print("\nTo see examples:")
print("  See examples/lrp_phase1_demo.md")
print("\n" + "=" * 80)
